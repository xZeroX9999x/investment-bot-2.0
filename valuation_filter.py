"""
valuation_filter.py
===================
FASE 2 (parte fundamental de valoración) — Descuento del PER vs histórico.

Calcula el PER actual y lo compara con la media de los últimos N años.
El activo solo aprueba si cotiza con al menos `per_discount_min` de descuento.

PER actual:
    Preferimos `trailingPE` de yfinance.info si está disponible.
    Si no, lo calculamos como Price / EPS_TTM.

PER histórico:
    Lo aproximamos como: precio_cierre_anual / EPS_anual del mismo año.
    Esta es una aproximación estándar usada en pantallas profesionales,
    suficiente para detectar disonancia significativa.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from models import FilterResult, FilterStage, FilterVerdict, RawMarketData, ValuationSnapshot

log = logging.getLogger(__name__)


class ValuationFilter:
    """Aplica el filtro de descuento del PER."""

    def __init__(self, discount_min: float, history_years: int) -> None:
        self._discount_min = discount_min
        self._history_years = history_years

    def evaluate(
        self,
        raw: RawMarketData,
        eps_dates: List[pd.Timestamp],
        eps_series: List[float],
    ) -> tuple[FilterResult, ValuationSnapshot]:
        """
        Recibimos `eps_dates` y `eps_series` ya alineadas desde el filtro
        fundamental. Esto garantiza que cada EPS está asociado a su fecha
        fiscal correcta sin riesgo de desfase.
        """
        current_pe = self._compute_current_pe(raw, eps_series)
        historical_mean = self._compute_historical_pe_mean(raw, eps_dates, eps_series)

        reasons: List[str] = []
        discount: Optional[float] = None
        passes = False

        if current_pe is None or current_pe <= 0:
            reasons.append(f"PER actual no calculable o no positivo ({current_pe})")
        elif historical_mean is None or historical_mean <= 0:
            reasons.append(
                f"Media histórica PER no calculable o no positiva ({historical_mean})"
            )
        else:
            # Descuento positivo significa que el PER actual está por
            # DEBAJO de la media histórica (más barato de lo habitual).
            discount = (historical_mean - current_pe) / historical_mean
            passes = discount >= self._discount_min
            if not passes:
                reasons.append(
                    f"Descuento PER {discount:.2%} < umbral {self._discount_min:.2%} "
                    f"(actual={current_pe:.2f}, media={historical_mean:.2f})"
                )

        snapshot = ValuationSnapshot(
            ticker=raw.ticker,
            current_pe=current_pe,
            historical_pe_mean=historical_mean,
            pe_discount=discount,
            pe_discount_passes=passes,
        )

        verdict = FilterVerdict.PASS if passes else FilterVerdict.FAIL
        result = FilterResult(
            ticker=raw.ticker,
            stage=FilterStage.VALUATION,
            verdict=verdict,
            reasons=reasons,
        )

        log.info(
            "Valuation evaluation completed",
            extra={
                "ticker": raw.ticker,
                "verdict": verdict.value,
                "current_pe": current_pe,
                "historical_pe": historical_mean,
                "discount": discount,
            },
        )
        return result, snapshot

    # -------------------------------------------------------------------------
    # Cálculos
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_current_pe(
        raw: RawMarketData,
        eps_series: List[float],
    ) -> Optional[float]:
        """
        Preferimos trailingPE explícito. Si no está, derivamos:
            current_pe = last_close / eps_ttm
        usando el EPS anual más reciente como proxy del TTM.
        """
        trailing_pe = raw.info.get("trailingPE")
        if trailing_pe is not None and trailing_pe > 0:
            return float(trailing_pe)

        if not eps_series:
            return None
        latest_eps = eps_series[-1]
        if latest_eps <= 0:
            return None

        if raw.history is None or raw.history.empty:
            return None
        last_price = float(raw.history["Close"].iloc[-1])
        return last_price / latest_eps

    def _compute_historical_pe_mean(
        self,
        raw: RawMarketData,
        eps_dates: List[pd.Timestamp],
        eps_series: List[float],
    ) -> Optional[float]:
        """
        Para cada año histórico (excluyendo el actual), tomamos el precio de
        cierre al final de ese periodo fiscal y dividimos por el EPS de ese
        año. Promediamos (con mediana, robusta a outliers).

        Inputs PRE-ALINEADOS: `eps_dates[i]` y `eps_series[i]` se refieren al
        MISMO periodo fiscal. Esto se garantiza en el filtro fundamental.

        Robustez:
          * Saltamos años con EPS<=0 (PER no significativo).
          * Saltamos años sin precio disponible en ese rango.
          * Si quedan < 2 puntos válidos, devolvemos None.
        """
        if not eps_series or len(eps_series) < 2:
            return None
        if len(eps_dates) != len(eps_series):
            # Invariante violado; defensa explícita
            log.error(
                "EPS dates/series length mismatch — invariant violated",
                extra={
                    "ticker": raw.ticker,
                    "dates_len": len(eps_dates),
                    "series_len": len(eps_series),
                },
            )
            return None
        if raw.history is None or raw.history.empty:
            return None

        # Limitamos al lookback configurado y excluimos el año en curso.
        # Construimos pares (date, eps) preservando alineación.
        paired = list(zip(eps_dates, eps_series))[-self._history_years:][:-1]
        if len(paired) < 2:
            return None

        # Normalizar índice de precios a tz-naive para comparaciones seguras
        closes = raw.history["Close"]
        if hasattr(closes.index, "tz") and closes.index.tz is not None:
            closes = closes.copy()
            closes.index = closes.index.tz_localize(None)

        pe_points: List[float] = []
        for fiscal_date, eps_value in paired:
            if eps_value is None or eps_value <= 0:
                continue
            target = pd.Timestamp(fiscal_date)
            if target.tz is not None:
                target = target.tz_localize(None)
            # Precio en el último día disponible <= fiscal_date
            mask = closes.index <= target
            if not mask.any():
                continue
            price_at_close = float(closes[mask].iloc[-1])
            if price_at_close <= 0:
                continue
            pe_points.append(price_at_close / eps_value)

        if len(pe_points) < 2:
            return None

        # Mediana > Media: robusta a años atípicos con EPS minúsculo que
        # pueden inflar la media artificialmente.
        return float(np.median(pe_points))
