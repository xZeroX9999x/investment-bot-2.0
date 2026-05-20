"""
scanner.py
==========
Encapsula la lógica de un ciclo de escaneo en una clase reutilizable.

Por qué existe (y no se reusa `Application._scan_cycle`):
    El modo web necesita exponer estado en vivo (¿está corriendo un escaneo?
    ¿cuándo terminó el último? ¿cuántas oportunidades encontró?) y debe poder
    aceptar disparos manuales sin colisionar con la ejecución periódica.
    Una clase con lock + atributos de estado es la forma canónica.

El modo headless (`main.py`) mantiene su propio loop por simplicidad; este
Scanner se usa desde `web_app.py` y `web_runner.py`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from alerts import AlertManager
from data_fetcher import AsyncDataFetcher
from database import AlertRepository
from main import Pipeline  # reutilizamos la Pipeline ya verificada
from models import GoldenOpportunity

log = logging.getLogger(__name__)


class ScanInProgressError(RuntimeError):
    """Se intentó disparar un escaneo mientras otro estaba en curso."""


class Scanner:
    """
    Maneja la ejecución de un ciclo de escaneo: fetch -> pipeline -> dispatch.
    Es seguro contra disparos concurrentes (lock asíncrono interno).
    Mantiene metadatos del último escaneo accesibles vía `status()`.
    """

    def __init__(
        self,
        universe: List[str],
        fetcher: AsyncDataFetcher,
        pipeline: Pipeline,
        alert_manager: AlertManager,
        alert_repo: AlertRepository,
    ) -> None:
        self._universe = universe
        self._fetcher = fetcher
        self._pipeline = pipeline
        self._alert_manager = alert_manager
        self._alert_repo = alert_repo
        self._lock = asyncio.Lock()

        # Estado expuesto a la API
        self._is_running: bool = False
        self._last_started_at: Optional[datetime] = None
        self._last_completed_at: Optional[datetime] = None
        self._last_duration_s: Optional[float] = None
        self._last_opportunities_count: int = 0
        self._last_processed_count: int = 0
        self._last_error: Optional[str] = None
        self._scan_count_total: int = 0

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------

    def status(self) -> dict:
        """Snapshot del estado actual (apto para JSON)."""
        return {
            "is_running": self._is_running,
            "universe_size": len(self._universe),
            "universe": list(self._universe),
            "last_started_at": (
                self._last_started_at.isoformat() if self._last_started_at else None
            ),
            "last_completed_at": (
                self._last_completed_at.isoformat() if self._last_completed_at else None
            ),
            "last_duration_s": self._last_duration_s,
            "last_opportunities_count": self._last_opportunities_count,
            "last_processed_count": self._last_processed_count,
            "last_error": self._last_error,
            "scan_count_total": self._scan_count_total,
        }

    async def scan_once(self) -> int:
        """
        Ejecuta un ciclo. Si ya hay uno en marcha, lanza `ScanInProgressError`.
        Devuelve el número de oportunidades efectivamente alertadas
        (después de la deduplicación diaria).
        """
        if self._lock.locked():
            raise ScanInProgressError("A scan is already in progress")

        async with self._lock:
            self._is_running = True
            self._last_started_at = datetime.now(tz=timezone.utc)
            self._last_error = None
            opportunities_alerted = 0

            try:
                log.info(
                    "Scan started",
                    extra={"universe_size": len(self._universe)},
                )

                # 1. Adquisición de datos
                try:
                    raw_batch = await self._fetcher.fetch_batch(self._universe)
                except Exception as exc:
                    log.exception("Batch fetch failed entirely", extra={"error": str(exc)})
                    self._last_error = f"fetch_failed: {exc}"
                    return 0

                self._last_processed_count = len(raw_batch)

                # 2. Pipeline en paralelo
                results = await asyncio.gather(
                    *(self._pipeline.process(r) for r in raw_batch),
                    return_exceptions=True,
                )
                opportunities: List[GoldenOpportunity] = []
                for raw, outcome in zip(raw_batch, results):
                    if isinstance(outcome, Exception):
                        log.exception(
                            "Pipeline raised unexpectedly",
                            extra={"ticker": raw.ticker, "error": str(outcome)},
                        )
                    elif outcome is not None:
                        opportunities.append(outcome)

                # 3. Dispatch con dedup diaria
                for opp in opportunities:
                    try:
                        if await self._alert_repo.already_alerted_today(opp.ticker):
                            log.info(
                                "Opportunity skipped (already alerted today)",
                                extra={"ticker": opp.ticker},
                            )
                            continue
                        await self._alert_manager.dispatch(opp)
                        await self._alert_repo.record(opp)
                        opportunities_alerted += 1
                    except Exception as exc:
                        log.exception(
                            "Alert dispatch failed",
                            extra={"ticker": opp.ticker, "error": str(exc)},
                        )

                self._last_opportunities_count = opportunities_alerted
                self._scan_count_total += 1
                return opportunities_alerted

            finally:
                self._last_completed_at = datetime.now(tz=timezone.utc)
                self._last_duration_s = (
                    self._last_completed_at - self._last_started_at
                ).total_seconds()
                self._is_running = False
                log.info(
                    "Scan completed",
                    extra={
                        "duration_s": self._last_duration_s,
                        "opportunities_alerted": opportunities_alerted,
                        "processed": self._last_processed_count,
                    },
                )

    async def scan_forever(
        self,
        interval_seconds: int,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Loop de escaneos periódicos. Sale limpiamente cuando `shutdown_event`
        se activa. Los errores en un ciclo NUNCA detienen el loop.
        """
        while not shutdown_event.is_set():
            try:
                await self.scan_once()
            except ScanInProgressError:
                # Otro disparo (probablemente manual desde la web) ya está
                # ejecutándose. Aceptamos y dormimos hasta el siguiente tick.
                log.info("Periodic scan skipped (manual scan in progress)")
            except Exception as exc:
                log.exception("Scan cycle failed", extra={"error": str(exc)})

            if shutdown_event.is_set():
                break

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=interval_seconds,
                )
            except asyncio.TimeoutError:
                # Timeout esperado: significa que el intervalo expiró sin shutdown
                pass
