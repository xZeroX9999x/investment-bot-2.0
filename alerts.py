"""
alerts.py
=========
Sistema de alertas asíncrono y modular. Implementa el patrón Strategy:

    Notifier (interfaz)
        |
        +-- TelegramNotifier   (aiohttp + Bot API)
        +-- EmailNotifier      (smtplib en thread, suficiente para baja frecuencia)
        +-- ConsoleNotifier    (fallback siempre disponible)

`AlertManager` enruta cada `GoldenOpportunity` a TODOS los notifiers activos
en paralelo. Un fallo en uno no degrada los demás.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from abc import ABC, abstractmethod
from email.message import EmailMessage
from typing import List, Optional

import aiohttp

from models import GoldenOpportunity

log = logging.getLogger(__name__)


# =============================================================================
# Interfaz
# =============================================================================

class Notifier(ABC):
    """Contrato de cualquier canal de notificación."""

    name: str = "abstract"

    @abstractmethod
    async def send(self, opportunity: GoldenOpportunity) -> bool:
        """Devuelve True si la entrega fue exitosa."""
        raise NotImplementedError


# =============================================================================
# Implementaciones
# =============================================================================

class ConsoleNotifier(Notifier):
    """Fallback que siempre funciona: imprime la alerta a stdout/log."""

    name = "console"

    async def send(self, opportunity: GoldenOpportunity) -> bool:
        log.warning(
            "GOLDEN OPPORTUNITY DETECTED",
            extra={
                "ticker": opportunity.ticker,
                "pe_discount": opportunity.pe_discount,
                "rsi": opportunity.rsi,
                "alert_channel": "console",
            },
        )
        # Doble emisión: log estructurado + bloque legible
        print("\n" + "=" * 60)
        print(opportunity.to_alert_text())
        print("=" * 60 + "\n", flush=True)
        return True


class TelegramNotifier(Notifier):
    """
    Envía un mensaje al chat configurado vía la Bot API de Telegram.
    Tolerante a fallos: cualquier excepción se loguea y se devuelve False.
    """

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout_sec: float = 15.0) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def send(self, opportunity: GoldenOpportunity) -> bool:
        payload = {
            "chat_id": self._chat_id,
            "text": opportunity.to_alert_text(),
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(self._url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.error(
                            "Telegram delivery failed",
                            extra={
                                "ticker": opportunity.ticker,
                                "status": resp.status,
                                "body": body[:500],
                            },
                        )
                        return False
            log.info(
                "Telegram alert delivered",
                extra={"ticker": opportunity.ticker},
            )
            return True
        except asyncio.TimeoutError:
            log.error(
                "Telegram delivery timed out",
                extra={"ticker": opportunity.ticker},
            )
        except aiohttp.ClientError as exc:
            log.error(
                "Telegram client error",
                extra={"ticker": opportunity.ticker, "error": str(exc)},
            )
        except Exception as exc:
            log.exception(
                "Unexpected error in Telegram delivery",
                extra={"ticker": opportunity.ticker, "error": str(exc)},
            )
        return False


class EmailNotifier(Notifier):
    """
    Envío por SMTP. Usamos `smtplib` (stdlib) ejecutado en un thread para
    no añadir dependencias adicionales y mantener la asincronía aparente.
    """

    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        sender: str,
        recipient: str,
        timeout_sec: float = 20.0,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._sender = sender
        self._recipient = recipient
        self._timeout = timeout_sec

    async def send(self, opportunity: GoldenOpportunity) -> bool:
        try:
            return await asyncio.to_thread(self._sync_send, opportunity)
        except Exception as exc:
            log.exception(
                "Unexpected error in Email delivery",
                extra={"ticker": opportunity.ticker, "error": str(exc)},
            )
            return False

    def _sync_send(self, opportunity: GoldenOpportunity) -> bool:
        msg = EmailMessage()
        msg["Subject"] = f"[ARQUITECTO] Oportunidad: {opportunity.ticker}"
        msg["From"] = self._sender
        msg["To"] = self._recipient
        msg.set_content(opportunity.to_alert_text())

        try:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(self._user, self._password)
                smtp.send_message(msg)
            log.info(
                "Email alert delivered",
                extra={"ticker": opportunity.ticker},
            )
            return True
        except (smtplib.SMTPException, OSError) as exc:
            log.error(
                "Email delivery failed",
                extra={"ticker": opportunity.ticker, "error": str(exc)},
            )
            return False


# =============================================================================
# Manager
# =============================================================================

class AlertManager:
    """
    Enruta cada oportunidad a TODOS los notifiers configurados en paralelo.
    Garantía: un fallo en un canal no afecta a los demás.
    """

    def __init__(self, notifiers: List[Notifier]) -> None:
        if not notifiers:
            raise ValueError("AlertManager requires at least one notifier")
        self._notifiers = notifiers
        log.info(
            "AlertManager initialized",
            extra={"channels": [n.name for n in notifiers]},
        )

    async def dispatch(self, opportunity: GoldenOpportunity) -> None:
        tasks = [n.send(opportunity) for n in self._notifiers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for notifier, outcome in zip(self._notifiers, results):
            if isinstance(outcome, Exception):
                log.error(
                    "Notifier raised unexpectedly",
                    extra={
                        "channel": notifier.name,
                        "ticker": opportunity.ticker,
                        "error": str(outcome),
                    },
                )
            elif outcome is False:
                log.warning(
                    "Notifier returned failure",
                    extra={"channel": notifier.name, "ticker": opportunity.ticker},
                )
