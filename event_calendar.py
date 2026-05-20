from datetime import date
from models import Catalyst

# Catalizador por defecto para endurecer el filtro RSI a niveles extremos
DEFAULT_CATALYSTS = [
    Catalyst(
        ticker="TTWO",
        event_date=date(2026, 11, 19),
        description="Lanzamiento Anticipado GTA VI",
        sensitivity_window_days=45
    ),
]

class EventCalendar:
    def __init__(self, repository):
        self._repo = repository
        self._cache = {}

    async def bootstrap_defaults(self, defaults):
        for c in defaults:
            await self._repo.upsert(c)

    async def active_catalysts(self, ticker: str, today: date):
        all_cats = await self._repo.list_for(ticker)
        return [c for c in all_cats if abs((c.event_date - today).days) <= c.sensitivity_window_days]