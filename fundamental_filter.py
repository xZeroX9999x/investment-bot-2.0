import logging
from typing import Tuple, List
import pandas as pd
from models import RawMarketData, FundamentalSnapshot, FilterResult, FilterStage, FilterVerdict

log = logging.getLogger(__name__)

class FundamentalFilter:
    def __init__(self, gross_margin_min: float, eps_years_lookback: int, shares_years_lookback: int):
        self._gross_margin_min = gross_margin_min
        self._eps_lookback = eps_years_lookback
        self._shares_lookback = shares_years_lookback

    def evaluate(self, raw: RawMarketData) -> Tuple[FilterResult, FundamentalSnapshot]:
        reasons = []
        us_gaap = raw.sec_facts.get("facts", {}).get("us-gaap", {})

        def _get_annual_series(tags: List[str]) -> pd.Series:
            for tag in tags:
                if tag in us_gaap:
                    units = us_gaap[tag].get("units", {})
                    usd_data = units.get("USD", units.get("shares", []))
                    annual = [d for d in usd_data if d.get("form") == "10-K" and d.get("fp") == "FY"]
                    if annual:
                        df = pd.DataFrame(annual)
                        df['end'] = pd.to_datetime(df['end'])
                        df = df.sort_values('end').drop_duplicates(subset='end', keep='last')
                        return pd.Series(df['val'].values, index=df['end'])
            return pd.Series(dtype=float)

        eps_series_pd = _get_annual_series(["EarningsPerShareDiluted", "EarningsPerShareBasic"])
        eps_dates = eps_series_pd.index.tolist()
        eps_vals = eps_series_pd.tolist()
        
        eps_trend_pos = len(eps_vals) >= 2 and (eps_vals[-1] > eps_vals[0])
        if not eps_trend_pos: reasons.append("Tendencia EPS no es positiva")

        revenues = _get_annual_series(["Revenues", "SalesRevenueNet"])
        cogs = _get_annual_series(["CostOfGoodsAndServicesSold", "CostOfRevenue"])
        gross_profit = _get_annual_series(["GrossProfit"])
        
        gross_margin = None
        if not gross_profit.empty and not revenues.empty:
            gross_margin = gross_profit.iloc[-1] / revenues.iloc[-1]
        elif not revenues.empty and not cogs.empty:
            gross_margin = (revenues.iloc[-1] - cogs.iloc[-1]) / revenues.iloc[-1]

        passes_margin = gross_margin is not None and gross_margin >= self._gross_margin_min
        if not passes_margin: reasons.append(f"Margen bruto menor a {self._gross_margin_min*100}%")

        shares = _get_annual_series(["WeightedAverageNumberOfDilutedSharesOutstanding"])
        shares_vals = shares.tolist()
        shares_dec = len(shares_vals) >= 2 and shares_vals[-1] < shares_vals[0]
        if not shares_dec: reasons.append("Acciones en circulación no decrecen")

        ocf = _get_annual_series(["NetCashProvidedByUsedInOperatingActivities"])
        capex = _get_annual_series(["PaymentsToAcquirePropertyPlantAndEquipment"])
        
        fcf_vals = []
        if not ocf.empty and not capex.empty:
            common_idx = ocf.index.intersection(capex.index)
            fcf_vals = (ocf[common_idx] - capex[common_idx].abs()).tolist()
            
        fcf_pos = len(fcf_vals) > 0 and all(v > 0 for v in fcf_vals[-self._eps_lookback:])
        if not fcf_pos: reasons.append("FCF no consistentemente positivo")

        snap = FundamentalSnapshot(
            ticker=raw.ticker,
            eps_series=eps_vals, eps_dates=eps_dates, eps_trend_positive=eps_trend_pos,
            gross_margin=gross_margin, gross_margin_passes=passes_margin,
            shares_outstanding_series=shares_vals, shares_decreasing=shares_dec,
            dividend_paid=True, dividend_no_recent_cuts=True,
            fcf_series=fcf_vals, fcf_consistently_positive=fcf_pos,
            fcf_covers_short_term_debt=True 
        )

        verdict = FilterVerdict.PASS if not reasons else FilterVerdict.FAIL
        return FilterResult(raw.ticker, FilterStage.FUNDAMENTAL, verdict, reasons), snap