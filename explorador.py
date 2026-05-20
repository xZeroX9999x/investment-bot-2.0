import asyncio
import aiohttp
import logging
from config import get_settings
from database import Database
from models import RawMarketData
from fundamental_filter import FundamentalFilter
from datetime import datetime, timezone
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("ExploradorSEC")

async def minero_autonomo():
    settings = get_settings()
    db = Database(settings.database_path)
    await db.initialize()
    
    headers = {"User-Agent": settings.sec_user_agent}
    filtro_fundamental = FundamentalFilter(
        settings.margen_bruto_minimo, 
        settings.eps_years_lookback, 
        settings.shares_years_lookback
    )

    log.info("Iniciando Motor de Descubrimiento SEC 24/7...")

    while True:
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get("https://www.sec.gov/files/company_tickers.json") as resp:
                    if resp.status != 200:
                        log.error("No se pudo descargar la lista maestra. Esperando 10 min...")
                        await asyncio.sleep(600)
                        continue
                    mercado_total = await resp.json()

                for key, data in mercado_total.items():
                    ticker = data["ticker"]
                    cik = str(data["cik_str"]).zfill(10)

                    # Aquí tu lógica de base de datos determinaría si ya lo analizaste hoy.
                    # Por simplicidad en este script base, lo procesaremos directamente.
                    log.info(f"Analizando CIK {cik} ({ticker})...")

                    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                    async with session.get(facts_url) as facts_resp:
                        if facts_resp.status == 429:
                            log.warning("Límite de la SEC alcanzado (429). Durmiendo 10 minutos.")
                            await asyncio.sleep(600)
                            continue
                        elif facts_resp.status != 200:
                            await asyncio.sleep(1.5)
                            continue
                        
                        company_facts = await facts_resp.json()
                        
                        # Creamos un RawMarketData falso (sin precios) solo para pasar el filtro fase 1
                        raw_mock = RawMarketData(
                            ticker=ticker, 
                            sec_facts=company_facts, 
                            history=pd.DataFrame(), 
                            fetched_at=datetime.now(tz=timezone.utc)
                        )
                        
                        resultado, snap = filtro_fundamental.evaluate(raw_mock)
                        
                        if resultado.passed:
                            log.info(f"🚀 ¡DESCUBRIMIENTO! {ticker} cumple la regla de >50% Margen y FCF+. Añadir a Universo.")
                            # Aquí llamarías a tu DB para añadirlo: await db.agregar_a_universo(ticker)
                        else:
                            log.debug(f"Descartado {ticker}: {resultado.reasons[0] if resultado.reasons else 'No apto'}")

                    # REGLA DE ORO: 1.5 segundos entre empresas para no ser baneado por la SEC
                    await asyncio.sleep(1.5)

        except Exception as e:
            log.error(f"Error crítico en el minero: {e}. Reiniciando en 60s...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(minero_autonomo())
    except KeyboardInterrupt:
        log.info("Motor de exploración detenido por el usuario.")