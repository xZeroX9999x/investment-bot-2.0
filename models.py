from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any

import pandas as pd


class FilterStage(str, Enum):
    FUNDAMENTAL = "FUNDAMENTAL"
    VALUATION = "VALUATION"
    TECHNICAL = "TECHNICAL"
    EVENT_DRIVEN = "EVENT_DRIVEN"

class FilterVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"

@dataclass(frozen=True)
class RawMarketData:
    ticker: str
    sec_facts: Dict[str, Any]
    history: pd.DataFrame
    fetched_at: datetime
    # Puente de compatibilidad silencioso para que valuation_filter.py no falle
    info: dict = field(default_factory=dict)

@dataclass(frozen=True)
class FundamentalSnapshot:
    ticker: str
    eps_series: List[float]
    eps_dates: List[pd.Timestamp]
    eps_trend_positive: bool
    gross_margin: Optional[float]
    gross_margin_passes: bool
    shares_outstanding_series: List[float]
    shares_decreasing: bool
    dividend_paid: bool
    dividend_no_recent_cuts: bool
    fcf_series: List[float]
    fcf_consistently_positive: bool
    fcf_covers_short_term_debt: bool

@dataclass(frozen=True)
class ValuationSnapshot:
    ticker: str
    current_pe: Optional[float]
    historical_pe_mean: Optional[float]
    pe_discount: Optional[float]
    pe_discount_passes: bool

@dataclass(frozen=True)
class TechnicalSnapshot:
    ticker: str
    last_close: float
    ma200: Optional[float]
    below_ma200: bool
    rsi: Optional[float]
    rsi_oversold: bool

@dataclass(frozen=True)
class Catalyst:
    ticker: str
    event_date: date
    description: str
    sensitivity_window_days: int = 30

@dataclass(frozen=True)
class FilterResult:
    ticker: str
    stage: FilterStage
    verdict: FilterVerdict
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == FilterVerdict.PASS

@dataclass(frozen=True)
class GoldenOpportunity:
    ticker: str
    detected_at: datetime
    gross_margin: float
    pe_discount: float
    rsi: float
    last_close: float
    ma200: float
    below_ma200: bool
    active_catalysts: List[Catalyst] = field(default_factory=list)

    def to_alert_text(self) -> str:
        catalyst_block = ""
        if self.active_catalysts:
            catalyst_lines = [
                f"   • {c.event_date.isoformat()} — {c.description}"
                for c in self.active_catalysts
            ]
            catalyst_block = "\n\nCatalizadores activos:\n" + "\n".join(catalyst_lines)

        return (
            f"🎯 OPORTUNIDAD DE ORO: {self.ticker}\n"
            f"Detectado: {self.detected_at.isoformat(timespec='seconds')}\n\n"
            f"• Margen Bruto:     {self.gross_margin:.1%}\n"
            f"• Descuento PER:    {self.pe_discount:.1%}\n"
            f"• RSI actual:       {self.rsi:.2f}\n"
            f"• Precio último:    {self.last_close:.2f}\n"
            f"• MA200:            {self.ma200:.2f}\n"
            f"• ¿Bajo MA200?:     {'Sí' if self.below_ma200 else 'No'}"
            f"{catalyst_block}"
        )