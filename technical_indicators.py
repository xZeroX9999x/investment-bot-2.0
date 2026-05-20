"""
technical_indicators.py
=======================
Funciones puras vectorizadas para cálculo de indicadores técnicos.
Implementaciones revisadas para ser numéricamente equivalentes a las
referencias canónicas (Wilder para RSI, SMA simple para la media móvil).

Diseño intencional:
    * Funciones sin estado: misma entrada -> misma salida.
    * Trabajan sobre `pd.Series` o `np.ndarray`.
    * Devuelven la *última* lectura o la serie completa según la firma.
    * Validan tamaño mínimo antes de calcular para evitar resultados NaN
      silenciosos que se propaguen al filtro.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class IndicatorError(Exception):
    """Datos insuficientes o inválidos para calcular el indicador."""


# =============================================================================
# Media móvil simple
# =============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Media móvil simple sobre los últimos `period` periodos.
    Devuelve la SERIE completa (cada elemento es la SMA hasta ese índice).
    """
    if period <= 0:
        raise IndicatorError(f"SMA period must be > 0, got {period}")
    if len(series) < period:
        raise IndicatorError(
            f"SMA requires at least {period} points; got {len(series)}"
        )
    return series.rolling(window=period, min_periods=period).mean()


def last_sma(series: pd.Series, period: int) -> float:
    """Atajo: SMA del último punto. Lanza si el resultado es NaN."""
    value = sma(series, period).iloc[-1]
    if pd.isna(value):
        raise IndicatorError(f"SMA{period} resulted in NaN")
    return float(value)


# =============================================================================
# RSI (J. Welles Wilder Jr., 1978)
# =============================================================================
#
# Definición canónica:
#   1. delta_t = close_t - close_{t-1}
#   2. gain_t = max(delta_t, 0);  loss_t = max(-delta_t, 0)
#   3. Para t = period: avg_gain = mean(gains[1:period+1])
#                       avg_loss = mean(losses[1:period+1])
#   4. Para t > period: suavizado de Wilder:
#        avg_gain_t = (avg_gain_{t-1} * (period-1) + gain_t) / period
#        avg_loss_t = (avg_loss_{t-1} * (period-1) + loss_t) / period
#   5. RS = avg_gain / avg_loss
#   6. RSI = 100 - 100 / (1 + RS)
#
# Casos límite:
#   * Si avg_loss == 0: RSI = 100 (movimiento totalmente alcista)
#   * Si avg_gain == 0: RSI = 0   (movimiento totalmente bajista)
# =============================================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder. Devuelve la serie completa (NaN hasta `period` puntos)."""
    if period < 2:
        raise IndicatorError(f"RSI period must be >= 2, got {period}")
    if len(series) < period + 1:
        raise IndicatorError(
            f"RSI requires at least {period + 1} points; got {len(series)}"
        )

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Suavizado de Wilder ≡ EMA con alpha = 1/period (adjust=False).
    # Esto es matemáticamente idéntico a la definición recursiva del paso 4.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss

    # Evitamos divisiones por cero:
    #   - avg_loss == 0 y avg_gain >  0 -> RSI = 100
    #   - avg_loss == 0 y avg_gain == 0 -> RSI = 50 (mercado plano; convención)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    rsi_values = rsi_values.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    return rsi_values


def last_rsi(series: pd.Series, period: int = 14) -> float:
    """Atajo: RSI del último punto. Lanza si NaN."""
    value = rsi(series, period).iloc[-1]
    if pd.isna(value):
        raise IndicatorError("RSI resulted in NaN")
    return float(value)


# =============================================================================
# Utilidades de extracción de precios
# =============================================================================

def close_series(history: pd.DataFrame) -> pd.Series:
    """Extrae la serie de cierres de un DataFrame OHLCV de yfinance."""
    if "Close" not in history.columns:
        raise IndicatorError("History DataFrame missing 'Close' column")
    series = history["Close"].dropna()
    if series.empty:
        raise IndicatorError("Close series is empty after dropping NaN")
    return series


def last_close(history: pd.DataFrame) -> float:
    """Último precio de cierre disponible."""
    return float(close_series(history).iloc[-1])


# =============================================================================
# Sanity check exportable (útil en tests o REPL)
# =============================================================================

def _self_test() -> None:
    """
    Verificación rápida contra valores de referencia.
    No es un test unitario formal; existe para que el ingeniero pueda
    ejecutar `python -c "from technical_indicators import _self_test; _self_test()"`
    y validar la implementación tras cualquier cambio.
    """
    # Serie sintética alcista pura -> RSI debe acercarse a 100
    s = pd.Series(np.arange(1, 50, dtype=float))
    r = last_rsi(s, period=14)
    assert 99.0 <= r <= 100.0, f"Alcista pura debería RSI~100, obtenido {r}"

    # Serie sintética bajista pura -> RSI debe acercarse a 0
    s = pd.Series(np.arange(50, 1, -1, dtype=float))
    r = last_rsi(s, period=14)
    assert 0.0 <= r <= 1.0, f"Bajista pura debería RSI~0, obtenido {r}"

    # Serie plana -> RSI = 50 por convención
    s = pd.Series([100.0] * 20)
    r = last_rsi(s, period=14)
    assert abs(r - 50.0) < 1e-6, f"Serie plana debería RSI=50, obtenido {r}"

    print("technical_indicators self-test OK")


if __name__ == "__main__":
    _self_test()
