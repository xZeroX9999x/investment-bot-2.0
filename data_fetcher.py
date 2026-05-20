import asyncio
import logging
import aiohttp
import yfinance as yf
from datetime import datetime, timezone
from typing import List, Optional, Dict
from models import RawMarketData

log = logging.getLogger(__name__)

class DataFetchError(Exception): pass
class InsufficientDataError(DataFetchError): pass

class AsyncDataFetcher:
    def __init__(self, concurrency_limit: int, sec_user_agent: str):
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self._sec_headers = {"User-Agent": sec_user_agent}
        self._cik_map: Dict[str, str] = {}

    async def _load_cik_map(self, session: aiohttp.ClientSession):
        if self._cik_map: return
        async with session.get("https://www.sec.gov/files/company_tickers.json") as resp:
            if resp.status == 200:
                data = await resp.json()
                for entry in data.values():
                    self._cik_map[entry["ticker"]] = str(entry["cik_str"]).zfill(10)

    async def fetch_batch(self, tickers: List[str]) -> List[RawMarketData]:
        async with aiohttp.ClientSession(headers=self._sec_headers) as session:
            await self._load_cik_map(session)
            coros = [self._fetch_single_safe(t, session) for t in tickers]
            results = await asyncio.gather(*coros)
            return [r for r in results if r is not None]

    async def _fetch_single_safe(self, ticker: str, session: aiohttp.ClientSession) -> Optional[RawMarketData]:
        try:
            return await self._do_fetch(ticker, session)
        except Exception as exc:
            log.warning(f"Error procesando {ticker}: {exc}")
            return None

    async def _do_fetch(self, ticker: str, session: aiohttp.ClientSession) -> RawMarketData:
        async with self._semaphore:
            cik = self._cik_map.get(ticker)
            if not cik:
                raise InsufficientDataError(f"CIK no encontrado para {ticker}")

            sec_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            async with session.get(sec_url) as resp:
                if resp.status != 200:
                    raise DataFetchError(f"SEC API falló con estado {resp.status}")
                sec_facts = await resp.json()

            history = await asyncio.to_thread(self._sync_fetch_prices, ticker)
            
            if history.empty or len(history) < 220:
                raise InsufficientDataError("Data de precios insuficiente para MA200")

            return RawMarketData(
                ticker=ticker,
                sec_facts=sec_facts,
                history=history,
                fetched_at=datetime.now(tz=timezone.utc)
            )

    @staticmethod
    def _sync_fetch_prices(ticker: str):
        return yf.Ticker(ticker).history(period="10y", interval="1d", auto_adjust=True)