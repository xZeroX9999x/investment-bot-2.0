"""
database.py
===========
Capa de persistencia local sobre SQLite (vía aiosqlite). El objetivo es la
**soberanía de datos**: cada ejecución acumula métricas históricas en disco
para reducir la dependencia de la API externa y permitir auditoría.

Tablas:
    raw_snapshots        -> snapshot crudo de la última descarga por ticker
    fundamental_history  -> métricas fundamentales evaluadas (auditoría)
    valuation_history    -> métricas de valoración evaluadas (auditoría)
    technical_history    -> métricas técnicas evaluadas (auditoría)
    alerts_log           -> alertas emitidas (dedup + histórico)
    catalysts            -> catalizadores temporales registrados

Diseño:
    * Conexión por operación + WAL mode para concurrencia ligera.
    * Schema autoaplicado en `initialize()` — idempotente.
    * Repositorios separados por dominio (SRP).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from models import (
    Catalyst,
    FundamentalSnapshot,
    GoldenOpportunity,
    TechnicalSnapshot,
    ValuationSnapshot,
)

log = logging.getLogger(__name__)


# =============================================================================
# DDL
# =============================================================================

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS fundamental_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    captured_at     TEXT    NOT NULL,
    eps_trend_pos   INTEGER NOT NULL,
    gross_margin    REAL,
    margin_passes   INTEGER NOT NULL,
    shares_decrease INTEGER NOT NULL,
    dividend_paid   INTEGER NOT NULL,
    dividend_no_cut INTEGER NOT NULL,
    fcf_pos         INTEGER NOT NULL,
    fcf_covers_debt INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamental_history(ticker, captured_at);

CREATE TABLE IF NOT EXISTS valuation_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    captured_at     TEXT    NOT NULL,
    current_pe      REAL,
    historical_pe   REAL,
    pe_discount     REAL,
    discount_passes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_val_ticker ON valuation_history(ticker, captured_at);

CREATE TABLE IF NOT EXISTS technical_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    captured_at     TEXT    NOT NULL,
    last_close      REAL,
    ma200           REAL,
    below_ma200     INTEGER NOT NULL,
    rsi             REAL,
    rsi_oversold    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tech_ticker ON technical_history(ticker, captured_at);

CREATE TABLE IF NOT EXISTS alerts_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    detected_at     TEXT    NOT NULL,
    gross_margin    REAL    NOT NULL,
    pe_discount     REAL    NOT NULL,
    rsi             REAL    NOT NULL,
    last_close      REAL    NOT NULL,
    ma200           REAL    NOT NULL,
    below_ma200     INTEGER NOT NULL,
    payload         TEXT    NOT NULL    -- mensaje completo enviado (auditoría)
);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker_date ON alerts_log(ticker, detected_at);

CREATE TABLE IF NOT EXISTS catalysts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    event_date      TEXT    NOT NULL,    -- ISO 8601
    description     TEXT    NOT NULL,
    sensitivity_window_days INTEGER NOT NULL DEFAULT 30,
    UNIQUE(ticker, event_date, description)
);
CREATE INDEX IF NOT EXISTS idx_catalysts_ticker ON catalysts(ticker);
"""


# =============================================================================
# Manager de conexión y schema
# =============================================================================

class Database:
    """
    Wrapper ligero sobre aiosqlite. No mantiene la conexión abierta entre
    operaciones — cada repositorio abre una conexión corta. Esto sacrifica
    micro-latencia a cambio de robustez ante crashes (el archivo siempre
    queda consistente gracias a WAL).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Crea el schema si no existe. Idempotente."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.executescript(_SCHEMA)
                await conn.commit()
            log.info("Database schema initialized", extra={"db_path": str(self.db_path)})
        except aiosqlite.Error as exc:
            log.exception("Failed to initialize database", extra={"error": str(exc)})
            raise


# =============================================================================
# Repositorios
# =============================================================================

class HistoryRepository:
    """Persistencia append-only de snapshots de cada filtro (auditoría)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _list_by_ticker(
        self,
        table: str,
        ticker: str,
        limit: int,
    ) -> list[dict]:
        """Genérico: lista los últimos N snapshots de una tabla para un ticker."""
        # Whitelist de tablas para prevenir cualquier inyección
        if table not in {"fundamental_history", "valuation_history", "technical_history"}:
            raise ValueError(f"Invalid history table: {table}")
        sql = f"""
            SELECT * FROM {table}
            WHERE ticker = ?
            ORDER BY captured_at DESC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, (ticker, limit)) as cur:
                    async for row in cur:
                        out.append({k: row[k] for k in row.keys()})
        except aiosqlite.Error as exc:
            log.warning(
                f"Could not list {table}",
                extra={"ticker": ticker, "error": str(exc)},
            )
        return out

    async def fundamental_for(self, ticker: str, limit: int = 10) -> list[dict]:
        return await self._list_by_ticker("fundamental_history", ticker, limit)

    async def valuation_for(self, ticker: str, limit: int = 10) -> list[dict]:
        return await self._list_by_ticker("valuation_history", ticker, limit)

    async def technical_for(self, ticker: str, limit: int = 10) -> list[dict]:
        return await self._list_by_ticker("technical_history", ticker, limit)

    async def save_fundamental(self, snap: FundamentalSnapshot) -> None:
        sql = """
            INSERT INTO fundamental_history
                (ticker, captured_at, eps_trend_pos, gross_margin, margin_passes,
                 shares_decrease, dividend_paid, dividend_no_cut, fcf_pos, fcf_covers_debt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await conn.execute(sql, (
                    snap.ticker,
                    datetime.now(tz=timezone.utc).isoformat(),
                    int(snap.eps_trend_positive),
                    snap.gross_margin,
                    int(snap.gross_margin_passes),
                    int(snap.shares_decreasing),
                    int(snap.dividend_paid),
                    int(snap.dividend_no_recent_cuts),
                    int(snap.fcf_consistently_positive),
                    int(snap.fcf_covers_short_term_debt),
                ))
                await conn.commit()
        except aiosqlite.Error as exc:
            log.warning(
                "Could not persist fundamental snapshot",
                extra={"ticker": snap.ticker, "error": str(exc)},
            )

    async def save_valuation(self, snap: ValuationSnapshot) -> None:
        sql = """
            INSERT INTO valuation_history
                (ticker, captured_at, current_pe, historical_pe, pe_discount, discount_passes)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await conn.execute(sql, (
                    snap.ticker,
                    datetime.now(tz=timezone.utc).isoformat(),
                    snap.current_pe,
                    snap.historical_pe_mean,
                    snap.pe_discount,
                    int(snap.pe_discount_passes),
                ))
                await conn.commit()
        except aiosqlite.Error as exc:
            log.warning(
                "Could not persist valuation snapshot",
                extra={"ticker": snap.ticker, "error": str(exc)},
            )

    async def save_technical(self, snap: TechnicalSnapshot) -> None:
        sql = """
            INSERT INTO technical_history
                (ticker, captured_at, last_close, ma200, below_ma200, rsi, rsi_oversold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await conn.execute(sql, (
                    snap.ticker,
                    datetime.now(tz=timezone.utc).isoformat(),
                    snap.last_close,
                    snap.ma200,
                    int(snap.below_ma200),
                    snap.rsi,
                    int(snap.rsi_oversold),
                ))
                await conn.commit()
        except aiosqlite.Error as exc:
            log.warning(
                "Could not persist technical snapshot",
                extra={"ticker": snap.ticker, "error": str(exc)},
            )


class AlertRepository:
    """Persistencia de alertas emitidas + utilidades de deduplicación."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def already_alerted_today(self, ticker: str) -> bool:
        """
        Evita disparar la misma alerta múltiples veces en el mismo día.
        Pánico real dura más de un escaneo, pero el destinatario humano no
        necesita ver la misma señal cada hora.
        """
        today_iso = datetime.now(tz=timezone.utc).date().isoformat()
        sql = """
            SELECT 1 FROM alerts_log
            WHERE ticker = ? AND substr(detected_at, 1, 10) = ?
            LIMIT 1
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                async with conn.execute(sql, (ticker, today_iso)) as cur:
                    row = await cur.fetchone()
                    return row is not None
        except aiosqlite.Error as exc:
            # En caso de error de DB, preferimos NO emitir alerta duplicada:
            # devolver True (asumir que ya alertamos) es el sesgo conservador.
            log.warning(
                "Dedup check failed; assuming already alerted",
                extra={"ticker": ticker, "error": str(exc)},
            )
            return True

    async def list_recent(self, limit: int = 50) -> list[dict]:
        """Últimas alertas registradas (para la API web)."""
        sql = """
            SELECT id, ticker, detected_at, gross_margin, pe_discount, rsi,
                   last_close, ma200, below_ma200, payload
            FROM alerts_log
            ORDER BY detected_at DESC
            LIMIT ?
        """
        out: list[dict] = []
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql, (limit,)) as cur:
                    async for row in cur:
                        out.append({
                            "id": row["id"],
                            "ticker": row["ticker"],
                            "detected_at": row["detected_at"],
                            "gross_margin": row["gross_margin"],
                            "pe_discount": row["pe_discount"],
                            "rsi": row["rsi"],
                            "last_close": row["last_close"],
                            "ma200": row["ma200"],
                            "below_ma200": bool(row["below_ma200"]),
                            "payload": row["payload"],
                        })
        except aiosqlite.Error as exc:
            log.warning("Could not list recent alerts", extra={"error": str(exc)})
        return out

    async def record(self, opp: GoldenOpportunity) -> None:
        sql = """
            INSERT INTO alerts_log
                (ticker, detected_at, gross_margin, pe_discount, rsi,
                 last_close, ma200, below_ma200, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await conn.execute(sql, (
                    opp.ticker,
                    opp.detected_at.isoformat(),
                    opp.gross_margin,
                    opp.pe_discount,
                    opp.rsi,
                    opp.last_close,
                    opp.ma200,
                    int(opp.below_ma200),
                    opp.to_alert_text(),
                ))
                await conn.commit()
        except aiosqlite.Error as exc:
            log.error(
                "Could not record alert in DB",
                extra={"ticker": opp.ticker, "error": str(exc)},
            )


class CatalystRepository:
    """CRUD de catalizadores temporales."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(self, catalyst: Catalyst) -> None:
        sql = """
            INSERT OR IGNORE INTO catalysts
                (ticker, event_date, description, sensitivity_window_days)
            VALUES (?, ?, ?, ?)
        """
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                await conn.execute(sql, (
                    catalyst.ticker,
                    catalyst.event_date.isoformat(),
                    catalyst.description,
                    catalyst.sensitivity_window_days,
                ))
                await conn.commit()
        except aiosqlite.Error as exc:
            log.warning(
                "Could not persist catalyst",
                extra={"ticker": catalyst.ticker, "error": str(exc)},
            )

    async def list_for(self, ticker: str) -> List[Catalyst]:
        sql = """
            SELECT ticker, event_date, description, sensitivity_window_days
            FROM catalysts
            WHERE ticker = ?
            ORDER BY event_date ASC
        """
        results: List[Catalyst] = []
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                async with conn.execute(sql, (ticker,)) as cur:
                    rows = await cur.fetchall()
                    for row in rows:
                        results.append(Catalyst(
                            ticker=row[0],
                            event_date=date.fromisoformat(row[1]),
                            description=row[2],
                            sensitivity_window_days=row[3],
                        ))
        except aiosqlite.Error as exc:
            log.warning(
                "Could not list catalysts",
                extra={"ticker": ticker, "error": str(exc)},
            )
        return results

    async def list_all(self) -> List[Catalyst]:
        sql = """
            SELECT ticker, event_date, description, sensitivity_window_days
            FROM catalysts ORDER BY event_date ASC
        """
        results: List[Catalyst] = []
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                async with conn.execute(sql) as cur:
                    rows = await cur.fetchall()
                    for row in rows:
                        results.append(Catalyst(
                            ticker=row[0],
                            event_date=date.fromisoformat(row[1]),
                            description=row[2],
                            sensitivity_window_days=row[3],
                        ))
        except aiosqlite.Error as exc:
            log.warning("Could not list all catalysts", extra={"error": str(exc)})
        return results

    async def list_all_with_ids(self) -> list[dict]:
        """Para la API: devuelve catalizadores con su `id` para permitir DELETE."""
        sql = """
            SELECT id, ticker, event_date, description, sensitivity_window_days
            FROM catalysts ORDER BY event_date ASC
        """
        out: list[dict] = []
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(sql) as cur:
                    async for row in cur:
                        out.append({
                            "id": row["id"],
                            "ticker": row["ticker"],
                            "event_date": row["event_date"],
                            "description": row["description"],
                            "sensitivity_window_days": row["sensitivity_window_days"],
                        })
        except aiosqlite.Error as exc:
            log.warning("Could not list catalysts with ids", extra={"error": str(exc)})
        return out

    async def delete_by_id(self, catalyst_id: int) -> bool:
        """Devuelve True si efectivamente se eliminó una fila."""
        sql = "DELETE FROM catalysts WHERE id = ?"
        try:
            async with aiosqlite.connect(self._db.db_path) as conn:
                cur = await conn.execute(sql, (catalyst_id,))
                await conn.commit()
                return cur.rowcount > 0
        except aiosqlite.Error as exc:
            log.warning(
                "Could not delete catalyst",
                extra={"catalyst_id": catalyst_id, "error": str(exc)},
            )
            return False
