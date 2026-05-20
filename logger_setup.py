"""
logger_setup.py
===============
Logging estructurado en formato JSON Lines (`.jsonl`). Cada registro es una
línea JSON autodescriptiva, apta para ingesta en sistemas SIEM, ELK, Loki o
simple `jq` desde la terminal. No se usan librerías pesadas: solo stdlib.

Justificación: en sistemas que manejan capital, el log debe ser auditable
de manera no ambigua. JSON estructurado elimina la dependencia de parsers
frágiles basados en regex.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Convierte cada `LogRecord` en una línea JSON con campos canónicos."""

    # Campos estándar de LogRecord que ya extraemos explícitamente.
    # Cualquier otro atributo arbitrario (los pasados como `extra=`) se serializa.
    _RESERVED_ATTRS = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Serializa campos `extra={...}` que no choquen con los reservados.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # Verifica serializabilidad
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)

        # Excepciones: incluir traceback completo para forensia.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_level: str, log_path: Path) -> logging.Logger:
    """
    Configura el logger raíz. Idempotente: múltiples llamadas no duplican
    handlers (defensa contra `setup_logging` invocado desde tests).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Limpieza para idempotencia
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = JsonFormatter()

    # Handler de fichero rotativo (10 MB x 5 archivos = 50 MB máximo)
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Handler de stdout para observabilidad en vivo
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Silenciar logs verbosos de librerías externas
    for noisy in ("urllib3", "yfinance", "peewee", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
