"""
web_runner.py
=============
Entrada del modo web. Levanta uvicorn sirviendo:
    * El dashboard estático (/)
    * La API REST (/api/*)
    * El loop periódico del scanner en background (mismo intervalo que el modo headless)

Ejecutar:
    python web_runner.py
    # o equivalentemente, para autoreload en desarrollo:
    uvicorn web_app:app --host 127.0.0.1 --port 8765 --reload

Por defecto se enlaza a 127.0.0.1 (solo accesible localmente). Para exponer
en una red de confianza, sobreescribir con HOST=0.0.0.0 en el entorno (y
montar autenticación delante, p.ej. nginx + basic auth).
"""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8765"))
    log_level = os.environ.get("WEB_LOG_LEVEL", "info")

    print(f"  Arquitecto Financiero Supremo · modo web")
    print(f"  Dashboard:   http://{host}:{port}/")
    print(f"  API docs:    http://{host}:{port}/docs")
    print(f"  Health:      http://{host}:{port}/api/healthz")
    print(f"  Ctrl+C para detener.\n", flush=True)

    try:
        uvicorn.run(
            "web_app:app",
            host=host,
            port=port,
            log_level=log_level,
            access_log=False,    # nuestros logs JSON ya cubren los accesos relevantes
            reload=False,
        )
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)


if __name__ == "__main__":
    main()
