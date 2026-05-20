"""
main.py
=======
Punto de entrada del sistema. Orquesta el pipeline completo:

    [Config] -> [Logging] -> [DB schema] -> [Catalysts bootstrap]
        -> Loop infinito:
            * Fetch async batch (todos los tickers)
            * Fase 1: Filtro Fundamental
            * Fase 2a: Filtro de Valoración
            * Fase 2b: Filtro Técnico (con modulación por catalizadores)
            * Persistencia de snapshots
            * Si las TRES fases pasan: emitir alerta (con dedup diaria)
        -> Sleep `scan_interval_seconds`

Manejo de señales: SIGINT/SIGTERM aborta limpiamente el loop.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import date, datetime, timezone
from typing import List

from alerts import (
    AlertManager,
    ConsoleNotifier,
    EmailNotifier,
    Notifier,
    TelegramNotifier,
)
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
from models import GoldenOpportunity, RawMarketData
from technical_filter import TechnicalFilter
from valuation_filter import ValuationFilter


# =============================================================================
# Construcción de notifiers según configuración
# =============================================================================

def build_notifiers(settings) -> List[Notifier]:
    """
    Construye la lista de notifiers a partir de las env vars.
    Console siempre activo (fallback garantizado).
    """
    notifiers: List[Notifier] = [ConsoleNotifier()]

    if settings.telegram_enabled:
        notifiers.append(TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        ))

    if settings.smtp_enabled:
        notifiers.append(EmailNotifier(
            host=settings.smtp_host,
            port=settings.smtp_port,
            user=settings.smtp_user,
            password=settings.smtp_password,
            sender=settings.smtp_from,
            recipient=settings.smtp_to,
        ))

    return notifiers


# =============================================================================
# Pipeline por ticker
# =============================================================================

class Pipeline:
    """
    Aplica las tres fases en orden estricto sobre un `RawMarketData`.
    Persiste cada snapshot. Si las tres fases pasan, devuelve
    `GoldenOpportunity`; en caso contrario devuelve None.
    """

    def __init__(
        self,
        fundamental: FundamentalFilter,
        valuation: ValuationFilter,
        technical: TechnicalFilter,
        history_repo: HistoryRepository,
        calendar: EventCalendar,
    ) -> None:
        self._fundamental = fundamental
        self._valuation = valuation
        self._technical = technical
        self._history = history_repo
        self._calendar = calendar
        self._log = logging.getLogger(self.__class__.__name__)

    async def process(self, raw: RawMarketData) -> GoldenOpportunity | None:
        try:
            # --- FASE 1 ---
            fund_result, fund_snap = self._fundamental.evaluate(raw)
            await self._history.save_fundamental(fund_snap)
            if not fund_result.passed:
                self._log.debug(
                    "Discarded at fundamental phase",
                    extra={"ticker": raw.ticker, "reasons": fund_result.reasons},
                )
                return None

            # --- FASE 2a: Valoración ---
            val_result, val_snap = self._valuation.evaluate(
                raw, fund_snap.eps_dates, fund_snap.eps_series
            )
            await self._history.save_valuation(val_snap)
            if not val_result.passed:
                self._log.debug(
                    "Discarded at valuation phase",
                    extra={"ticker": raw.ticker, "reasons": val_result.reasons},
                )
                return None

            # --- FASE 2b: Técnico ---
            today = date.today()
            active_catalysts = await self._calendar.active_catalysts(raw.ticker, today)
            all_catalysts = await self._calendar.catalysts_for(raw.ticker)
            tech_result, tech_snap = self._technical.evaluate(raw, all_catalysts, today)
            await self._history.save_technical(tech_snap)
            if not tech_result.passed:
                self._log.debug(
                    "Discarded at technical phase",
                    extra={"ticker": raw.ticker, "reasons": tech_result.reasons},
                )
                return None

            # --- LLEGÓ AL FINAL: oportunidad real ---
            return GoldenOpportunity(
                ticker=raw.ticker,
                detected_at=datetime.now(tz=timezone.utc),
                gross_margin=fund_snap.gross_margin or 0.0,
                pe_discount=val_snap.pe_discount or 0.0,
                rsi=tech_snap.rsi or 0.0,
                last_close=tech_snap.last_close,
                ma200=tech_snap.ma200 or 0.0,
                below_ma200=tech_snap.below_ma200,
                active_catalysts=active_catalysts,
            )

        except Exception as exc:
            # Cualquier excepción no controlada: log y continuar.
            # JAMÁS abortamos el batch por un ticker problemático.
            self._log.exception(
                "Pipeline failed for ticker",
                extra={"ticker": raw.ticker, "error": str(exc)},
            )
            return None


# =============================================================================
# Aplicación principal
# =============================================================================

class Application:
    """Orquestador top-level. Maneja ciclo de vida y señales."""

    def __init__(self) -> None:
        self._settings = get_settings()
        setup_logging(self._settings.log_level, self._settings.log_path)
        self._log = logging.getLogger(self.__class__.__name__)
        self._shutdown = asyncio.Event()

    def install_signal_handlers(self) -> None:
        """SIGINT/SIGTERM -> shutdown limpio."""
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self._request_shutdown, sig_name)
            except NotImplementedError:
                # Windows no soporta add_signal_handler para SIGTERM
                pass

    def _request_shutdown(self, sig_name: str) -> None:
        self._log.warning("Shutdown requested", extra={"signal": sig_name})
        self._shutdown.set()

    async def run(self) -> None:
        self._log.info(
            "Arquitecto Financiero Supremo starting",
            extra={
                "universe_size": len(self._settings.ticker_universe),
                "concurrency": self._settings.concurrency_limit,
                "scan_interval_s": self._settings.scan_interval_seconds,
            },
        )

        # --- DB ---
        db = Database(self._settings.database_path)
        await db.initialize()
        history_repo = HistoryRepository(db)
        alert_repo = AlertRepository(db)
        catalyst_repo = CatalystRepository(db)

        # --- Catalizadores ---
        calendar = EventCalendar(catalyst_repo)
        await calendar.bootstrap_defaults(DEFAULT_CATALYSTS)

        # --- Filtros ---
        fundamental = FundamentalFilter(
            gross_margin_min=self._settings.margen_bruto_minimo,
            eps_years_lookback=self._settings.eps_years_lookback,
            shares_years_lookback=self._settings.shares_years_lookback,
        )
        valuation = ValuationFilter(
            discount_min=self._settings.per_discount_min,
            history_years=self._settings.per_history_years,
        )
        technical = TechnicalFilter(
            ma_period=self._settings.ma_period,
            rsi_period=self._settings.rsi_period,
            rsi_oversold_threshold=self._settings.rsi_oversold_threshold,
            rsi_extreme_oversold=self._settings.rsi_extreme_oversold,
        )

        # --- Fetcher ---
        fetcher = AsyncDataFetcher(
            concurrency_limit=self._settings.concurrency_limit,
            retry_max_attempts=self._settings.retry_max_attempts,
            retry_base_delay=self._settings.retry_base_delay,
            history_years=self._settings.per_history_years,
        )

        # --- Notifiers ---
        manager = AlertManager(build_notifiers(self._settings))

        # --- Pipeline ---
        pipeline = Pipeline(fundamental, valuation, technical, history_repo, calendar)

        # --- Loop principal ---
        while not self._shutdown.is_set():
            await self._scan_cycle(fetcher, pipeline, manager, alert_repo)
            if self._shutdown.is_set():
                break
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._settings.scan_interval_seconds,
                )
            except asyncio.TimeoutError:
                # Timeout esperado: significa que el intervalo expiró sin shutdown
                pass

        self._log.info("Arquitecto shutting down cleanly")

    # -------------------------------------------------------------------------
    # Un ciclo completo de escaneo
    # -------------------------------------------------------------------------

    async def _scan_cycle(
        self,
        fetcher: AsyncDataFetcher,
        pipeline: Pipeline,
        manager: AlertManager,
        alert_repo: AlertRepository,
    ) -> None:
        cycle_start = datetime.now(tz=timezone.utc)
        self._log.info("Scan cycle starting", extra={"start": cycle_start.isoformat()})

        try:
            raw_batch = await fetcher.fetch_batch(self._settings.ticker_universe)
        except Exception as exc:
            self._log.exception("Batch fetch failed entirely", extra={"error": str(exc)})
            return

        self._log.info(
            "Batch fetched",
            extra={
                "requested": len(self._settings.ticker_universe),
                "received": len(raw_batch),
            },
        )

        # Procesamos el pipeline en paralelo controlado.
        # Cada `pipeline.process` ya es internamente sync/async según corresponda
        # y nunca debería propagar excepciones; pero protegemos con gather de todos modos.
        results = await asyncio.gather(
            *(pipeline.process(r) for r in raw_batch),
            return_exceptions=True,
        )

        opportunities: List[GoldenOpportunity] = []
        for raw, outcome in zip(raw_batch, results):
            if isinstance(outcome, Exception):
                self._log.exception(
                    "Pipeline raised unexpectedly",
                    extra={"ticker": raw.ticker, "error": str(outcome)},
                )
            elif outcome is not None:
                opportunities.append(outcome)

        # Despachar alertas con dedup diaria
        for opp in opportunities:
            try:
                if await alert_repo.already_alerted_today(opp.ticker):
                    self._log.info(
                        "Opportunity skipped (already alerted today)",
                        extra={"ticker": opp.ticker},
                    )
                    continue
                await manager.dispatch(opp)
                await alert_repo.record(opp)
            except Exception as exc:
                self._log.exception(
                    "Alert dispatch failed",
                    extra={"ticker": opp.ticker, "error": str(exc)},
                )

        elapsed = (datetime.now(tz=timezone.utc) - cycle_start).total_seconds()
        self._log.info(
            "Scan cycle completed",
            extra={
                "elapsed_s": elapsed,
                "opportunities_detected": len(opportunities),
            },
        )


# =============================================================================
# Entrada de proceso
# =============================================================================

async def amain() -> None:
    app = Application()
    app.install_signal_handlers()
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        # Mensaje final en caso de Ctrl+C antes de instalar handlers
        print("\nInterrupted by user.")
