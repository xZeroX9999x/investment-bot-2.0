"""
technical_filter.py
===================
FASE 2 (parte técnica) — Indicadores de Miedo Extremo.

Gatillo final: el precio debe estar POR DEBAJO de la MA200 **y** el RSI
debe marcar sobreventa. El umbral de sobreventa se endurece (cae a
`rsi_extreme_oversold`) cuando el ticker tiene un catalizador cercano:
en esas ventanas el ruido pre-evento eleva la probabilidad de falsos
positivos, así que exigimos sobreventa más profunda.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List

from models import (
    Catalyst,
    FilterResult,
    FilterStage,
    FilterVerdict,
    RawMarketData,
    TechnicalSnapshot,
)
from technical_indicators import (
    IndicatorError,
    close_series,
    last_close,
    last_rsi,
    last_sma,
)

log = logging.getLogger(__name__)


class TechnicalFilter:
    """Aplica los gatillos técnicos: MA200 + RSI con modulación por catalizador."""

    def __init__(
        self,
        ma_period: int,
        rsi_period: int,
        rsi_oversold_threshold: float,
        rsi_extreme_oversold: float,
    ) -> None:
        self._ma_period = ma_period
        self._rsi_period = rsi_period
        self._rsi_oversold = rsi_oversold_threshold
        self._rsi_extreme = rsi_extreme_oversold

    def evaluate(
        self,
        raw: RawMarketData,
        catalysts: List[Catalyst],
        today: date,
    ) -> tuple[FilterResult, TechnicalSnapshot]:
        reasons: List[str] = []

        try:
            closes = close_series(raw.history)
        except IndicatorError as exc:
            return (
                FilterResult(
                    ticker=raw.ticker,
                    stage=FilterStage.TECHNICAL,
                    verdict=FilterVerdict.FAIL,
                    reasons=[f"Serie de cierres inválida: {exc}"],
                ),
                TechnicalSnapshot(
                    ticker=raw.ticker,
                    last_close=0.0,
                    ma200=None,
                    below_ma200=False,
                    rsi=None,
                    rsi_oversold=False,
                ),
            )

        try:
            close_value = last_close(raw.history)
            ma_value = last_sma(closes, self._ma_period)
            rsi_value = last_rsi(closes, self._rsi_period)
        except IndicatorError as exc:
            return (
                FilterResult(
                    ticker=raw.ticker,
                    stage=FilterStage.TECHNICAL,
                    verdict=FilterVerdict.FAIL,
                    reasons=[f"Indicador no calculable: {exc}"],
                ),
                TechnicalSnapshot(
                    ticker=raw.ticker,
                    last_close=0.0,
                    ma200=None,
                    below_ma200=False,
                    rsi=None,
                    rsi_oversold=False,
                ),
            )

        # Decidir umbral RSI a aplicar
        applied_threshold = self._resolve_rsi_threshold(raw.ticker, catalysts, today)

        below_ma = close_value < ma_value
        is_oversold = rsi_value < applied_threshold

        if not below_ma:
            reasons.append(
                f"Precio {close_value:.2f} no está bajo MA{self._ma_period}={ma_value:.2f}"
            )
        if not is_oversold:
            reasons.append(
                f"RSI={rsi_value:.2f} no marca sobreventa "
                f"(umbral aplicado={applied_threshold})"
            )

        snapshot = TechnicalSnapshot(
            ticker=raw.ticker,
            last_close=close_value,
            ma200=ma_value,
            below_ma200=below_ma,
            rsi=rsi_value,
            rsi_oversold=is_oversold,
        )

        verdict = FilterVerdict.PASS if not reasons else FilterVerdict.FAIL
        result = FilterResult(
            ticker=raw.ticker,
            stage=FilterStage.TECHNICAL,
            verdict=verdict,
            reasons=reasons,
        )

        log.info(
            "Technical evaluation completed",
            extra={
                "ticker": raw.ticker,
                "verdict": verdict.value,
                "close": close_value,
                "ma": ma_value,
                "rsi": rsi_value,
                "rsi_threshold": applied_threshold,
            },
        )
        return result, snapshot

    # -------------------------------------------------------------------------
    # Lógica de modulación por catalizador
    # -------------------------------------------------------------------------

    def _resolve_rsi_threshold(
        self,
        ticker: str,
        catalysts: List[Catalyst],
        today: date,
    ) -> float:
        """
        Si HAY un catalizador activo (dentro de la ventana de sensibilidad),
        endurecemos el umbral de sobreventa a `rsi_extreme_oversold`.
        Caso contrario, usamos el umbral normal.
        """
        for c in catalysts:
            if c.ticker != ticker:
                continue
            window = c.sensitivity_window_days
            distance_days = abs((c.event_date - today).days)
            if distance_days <= window:
                log.info(
                    "Catalyst active; tightening RSI threshold",
                    extra={
                        "ticker": ticker,
                        "event_date": c.event_date.isoformat(),
                        "distance_days": distance_days,
                        "window": window,
                    },
                )
                return self._rsi_extreme
        return self._rsi_oversold
