"""
web_app.py
==========
Capa HTTP: FastAPI que expone la funcionalidad del Arquitecto vía REST y
sirve el dashboard estático. Reutiliza los mismos componentes que el modo
headless (filtros, fetcher, base de datos), garantizando que la UI nunca
diverja del comportamiento real del sistema.

Diseño:
    * Una sola instancia de cada componente, vivida en `app.state`.
    * El loop periódico corre como `asyncio.Task` lanzada en el lifespan.
    * Los endpoints son thin: validan, llaman al componente, devuelven JSON.
    * El dashboard se sirve desde `static/index.html`.

Seguridad:
    * Bind a 127.0.0.1 por defecto (uso local).
    * Sin autenticación: el supuesto es entorno de un único operador.
      Si se expone a red, montar detrás de un reverse proxy con auth.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from alerts import AlertManager
from config import get_settings
from data_fetcher import AsyncDataFetcher
from database import (
    AlertRepository,
    CatalystRepository,
    Database,
    HistoryRepository,
)
from event_calendar import DEFAULT_CATALYSTS, EventCalendar
from fundamental_filter import FundamentalFilter
from logger_setup import setup_logging
from main import Pipeline, build_notifiers
from models import Catalyst
from scanner import Scanner, ScanInProgressError
from technical_filter import TechnicalFilter
from valuation_filter import ValuationFilter

log = logging.getLogger(__name__)


# =============================================================================
# Modelos Pydantic de request/response
# =============================================================================

class CatalystCreateRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    event_date: date
    description: str = Field(min_length=1, max_length=500)
    sensitivity_window_days: int = Field(default=30, ge=1, le=365)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class CatalystResponse(BaseModel):
    id: int
    ticker: str
    event_date: str
    description: str
    sensitivity_window_days: int


class ScanTriggerResponse(BaseModel):
    accepted: bool
    message: str


# =============================================================================
# Lifespan: arranque y apagado
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Crea los componentes una sola vez al arranque, los publica en `app.state`,
    y lanza el loop periódico. Al cerrar, cancela el loop limpiamente.
    """
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_path)
    log.info("Web mode bootstrap starting")

    # --- DB ---
    db = Database(settings.database_path)
    await db.initialize()
    history_repo = HistoryRepository(db)
    alert_repo = AlertRepository(db)
    catalyst_repo = CatalystRepository(db)

    # --- Catalizadores por defecto ---
    calendar = EventCalendar(catalyst_repo)
    await calendar.bootstrap_defaults(DEFAULT_CATALYSTS)

    # --- Filtros ---
    fundamental = FundamentalFilter(
        gross_margin_min=settings.margen_bruto_minimo,
        eps_years_lookback=settings.eps_years_lookback,
        shares_years_lookback=settings.shares_years_lookback,
    )
    valuation = ValuationFilter(
        discount_min=settings.per_discount_min,
        history_years=settings.per_history_years,
    )
    technical = TechnicalFilter(
        ma_period=settings.ma_period,
        rsi_period=settings.rsi_period,
        rsi_oversold_threshold=settings.rsi_oversold_threshold,
        rsi_extreme_oversold=settings.rsi_extreme_oversold,
    )

    # --- Fetcher + notifiers ---
    fetcher = AsyncDataFetcher(
        concurrency_limit=settings.concurrency_limit,
        sec_user_agent=settings.sec_user_agent,
    )
    alert_manager = AlertManager(build_notifiers(settings))

    # --- Pipeline + Scanner ---
    pipeline = Pipeline(fundamental, valuation, technical, history_repo, calendar)
    scanner = Scanner(
        universe=list(settings.ticker_universe),
        fetcher=fetcher,
        pipeline=pipeline,
        alert_manager=alert_manager,
        alert_repo=alert_repo,
    )

    # --- Publicar en app.state ---
    app.state.settings = settings
    app.state.db = db
    app.state.history_repo = history_repo
    app.state.alert_repo = alert_repo
    app.state.catalyst_repo = catalyst_repo
    app.state.calendar = calendar
    app.state.scanner = scanner
    app.state.shutdown_event = asyncio.Event()

    # --- Lanzar loop periódico en background ---
    scanner_task = asyncio.create_task(
        scanner.scan_forever(
            interval_seconds=settings.scan_interval_seconds,
            shutdown_event=app.state.shutdown_event,
        ),
        name="scanner_periodic",
    )
    app.state.scanner_task = scanner_task
    log.info(
        "Periodic scanner task started",
        extra={"interval_s": settings.scan_interval_seconds},
    )

    try:
        yield
    finally:
        log.info("Web mode shutdown starting")
        app.state.shutdown_event.set()
        try:
            await asyncio.wait_for(scanner_task, timeout=30)
        except asyncio.TimeoutError:
            log.warning("Scanner task did not stop in time; cancelling")
            scanner_task.cancel()
        except Exception as exc:
            log.warning("Scanner task ended with error", extra={"error": str(exc)})


# =============================================================================
# App factory
# =============================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Arquitecto Financiero Supremo",
        description="API + Dashboard del bot de detección de oportunidades.",
        version="1.0.0",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).parent / "static"

    # ------------------------------------------------------------------
    # Root: sirve el dashboard
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Dashboard not found at {index_path}",
            )
        return FileResponse(index_path)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status(request: Request) -> dict:
        scanner: Scanner = request.app.state.scanner
        settings = request.app.state.settings
        return {
            "scanner": scanner.status(),
            "config": {
                "scan_interval_seconds": settings.scan_interval_seconds,
                "margen_bruto_minimo": settings.margen_bruto_minimo,
                "per_discount_min": settings.per_discount_min,
                "rsi_oversold_threshold": settings.rsi_oversold_threshold,
                "rsi_extreme_oversold": settings.rsi_extreme_oversold,
                "ma_period": settings.ma_period,
                "rsi_period": settings.rsi_period,
            },
        }

    @app.get("/api/opportunities")
    async def get_opportunities(request: Request, limit: int = 50) -> list[dict]:
        limit = max(1, min(limit, 500))
        repo: AlertRepository = request.app.state.alert_repo
        return await repo.list_recent(limit=limit)

    @app.get("/api/tickers")
    async def get_tickers(request: Request) -> list[str]:
        return list(request.app.state.settings.ticker_universe)

    @app.get("/api/tickers/{ticker}/history")
    async def get_ticker_history(
        request: Request,
        ticker: str,
        limit: int = 10,
    ) -> dict:
        ticker = ticker.upper().strip()
        if not ticker:
            raise HTTPException(status_code=400, detail="Invalid ticker")
        limit = max(1, min(limit, 50))
        repo: HistoryRepository = request.app.state.history_repo
        return {
            "ticker": ticker,
            "fundamental": await repo.fundamental_for(ticker, limit),
            "valuation": await repo.valuation_for(ticker, limit),
            "technical": await repo.technical_for(ticker, limit),
        }

    @app.get("/api/catalysts", response_model=List[CatalystResponse])
    async def list_catalysts(request: Request) -> list[dict]:
        repo: CatalystRepository = request.app.state.catalyst_repo
        return await repo.list_all_with_ids()

    @app.post(
        "/api/catalysts",
        status_code=status.HTTP_201_CREATED,
    )
    async def add_catalyst(
        request: Request,
        body: CatalystCreateRequest,
    ) -> dict:
        repo: CatalystRepository = request.app.state.catalyst_repo
        calendar: EventCalendar = request.app.state.calendar
        catalyst = Catalyst(
            ticker=body.ticker,
            event_date=body.event_date,
            description=body.description,
            sensitivity_window_days=body.sensitivity_window_days,
        )
        await repo.upsert(catalyst)
        # Invalidamos la caché en memoria del calendario
        calendar._cache.clear()
        calendar._cache_loaded = False
        return {"created": True, "catalyst": {
            "ticker": catalyst.ticker,
            "event_date": catalyst.event_date.isoformat(),
            "description": catalyst.description,
            "sensitivity_window_days": catalyst.sensitivity_window_days,
        }}

    @app.delete("/api/catalysts/{catalyst_id}")
    async def delete_catalyst(request: Request, catalyst_id: int) -> dict:
        repo: CatalystRepository = request.app.state.catalyst_repo
        calendar: EventCalendar = request.app.state.calendar
        deleted = await repo.delete_by_id(catalyst_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Catalyst not found")
        calendar._cache.clear()
        calendar._cache_loaded = False
        return {"deleted": True, "id": catalyst_id}

    @app.post("/api/scan/trigger", response_model=ScanTriggerResponse)
    async def trigger_scan(request: Request) -> ScanTriggerResponse:
        scanner: Scanner = request.app.state.scanner
        if scanner.status()["is_running"]:
            raise HTTPException(
                status_code=409,
                detail="A scan is already in progress",
            )
        # Lanzamos como tarea para no bloquear la respuesta HTTP.
        # El cliente puede hacer polling a /api/status para ver el progreso.
        asyncio.create_task(_safe_scan(scanner), name="manual_scan")
        return ScanTriggerResponse(
            accepted=True,
            message="Scan started in background; poll /api/status for progress",
        )

    @app.get("/api/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    # ------------------------------------------------------------------
    # Static
    # ------------------------------------------------------------------
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


async def _safe_scan(scanner: Scanner) -> None:
    """Wrapper que NUNCA propaga excepciones desde la tarea de fondo."""
    try:
        await scanner.scan_once()
    except ScanInProgressError:
        log.info("Manual scan ignored (already running)")
    except Exception as exc:
        log.exception("Manual scan failed", extra={"error": str(exc)})


# Instancia global para uvicorn (uvicorn web_app:app)
app = create_app()
