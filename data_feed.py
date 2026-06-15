"""
data_feed.py  — ATRAIN FOREX Binary Signal Robot v2.1
══════════════════════════════════════════════════════════════════════
WHAT CHANGED FROM v2.0
  ✦ fetch_multi_tf()  — fetches M1/M5/M15/H1 bars concurrently and
    returns a dict ready for MultiTFBiasEngine.evaluate()
  ✦ OHLCVBar conversion helper _to_ohlcv()
  ✦ All original methods (fetch_candles, fetch_live_quote,
    get_current_seconds_into_minute, close) are unchanged.

Twelve Data intervals used:
  M1  → "1min"    M5  → "5min"
  M15 → "15min"   H1  → "1h"

Bar counts:  M1 = 60, M5 = 60, M15 = 60, H1 = 210
(H1 needs 210 bars to warm up EMA-200 in the bias engine.)
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from multi_tf_bias import OHLCVBar

logger = logging.getLogger("DataFeed")

# ── Asset catalogue — unchanged from v2.0 ────────────────────────────────────

ASSET_CATALOGUE = {
    "Forex": {
        "EUR/USD": "EUR/USD",
        "GBP/USD": "GBP/USD",
        "USD/JPY": "USD/JPY",
        "AUD/USD": "AUD/USD",
        "USD/CAD": "USD/CAD",
        "USD/CHF": "USD/CHF",
        "EUR/JPY": "EUR/JPY",
        "EUR/GBP": "EUR/GBP",
        "GBP/JPY": "GBP/JPY",
        "NZD/USD": "NZD/USD",
        "EUR/AUD": "EUR/AUD",
        "AUD/JPY": "AUD/JPY",
    },
    "Crypto": {
        "BTC/USD": "BTC/USD",
        "ETH/USD": "ETH/USD",
        "XRP/USD": "XRP/USD",
        "LTC/USD": "LTC/USD",
        "BNB/USD": "BNB/USD",
    },
    "Commodities": {
        "XAU/USD": "XAU/USD",
        "XAG/USD": "XAG/USD",
        "WTI/USD": "USOIL",
        "BRENT":   "UKOIL",
    },
    "Stocks": {
        "AAPL":  "AAPL",
        "TSLA":  "TSLA",
        "AMZN":  "AMZN",
        "GOOGL": "GOOGL",
        "MSFT":  "MSFT",
        "META":  "META",
        "NVDA":  "NVDA",
    },
    "Indices": {
        "US500":  "SPX",
        "US30":   "DJI",
        "NAS100": "NDX",
        "GER40":  "DAX",
        "UK100":  "FTSE",
        "JPN225": "NI225",
    },
}


def find_asset(symbol: str) -> Optional[str]:
    for cat in ASSET_CATALOGUE.values():
        if symbol in cat:
            return cat[symbol]
    return None


# ── TF configuration ──────────────────────────────────────────────────────────

_TF_CONFIG = {
    "M1":  {"interval": "1min",  "count": 60},
    "M5":  {"interval": "5min",  "count": 60},
    "M15": {"interval": "15min", "count": 60},
    "H1":  {"interval": "1h",    "count": 210},   # 210 for EMA-200 warmup
}


# ─────────────────────────────────────────────────────────────────────────────

class TwelveDataFeed:

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str):
        self._key     = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core candle fetch (unchanged from v2.0) ───────────────────────────────

    async def fetch_candles(
        self, symbol: str, interval: str = "1min", count: int = 60
    ) -> list[dict]:
        """
        Fetch OHLCV candles from Twelve Data.
        Returns a chronological list (oldest → newest) of dicts with
        keys: open, high, low, close, volume.
        """
        session = await self._get_session()
        params = {
            "symbol":   symbol,
            "interval": interval,
            "outputsize": count,
            "apikey":   self._key,
        }
        try:
            async with session.get(f"{self.BASE_URL}/time_series", params=params) as resp:
                data = await resp.json()
                values = data.get("values", [])
                if not values:
                    logger.warning(f"No candles returned for {symbol} {interval}")
                    return []
                # Twelve Data returns newest-first → reverse to chronological
                candles = []
                for v in reversed(values):
                    candles.append({
                        "open":   float(v["open"]),
                        "high":   float(v["high"]),
                        "low":    float(v["low"]),
                        "close":  float(v["close"]),
                        "volume": float(v.get("volume", 0)),
                    })
                return candles
        except Exception as e:
            logger.error(f"fetch_candles error [{symbol} {interval}]: {e}")
            return []

    async def fetch_live_quote(self, symbol: str) -> Optional[dict]:
        """Fetch the currently forming (live) candle quote."""
        session = await self._get_session()
        params = {"symbol": symbol, "apikey": self._key}
        try:
            async with session.get(f"{self.BASE_URL}/quote", params=params) as resp:
                data = await resp.json()
                if "open" not in data:
                    return None
                return {
                    "open":   float(data["open"]),
                    "high":   float(data["high"]),
                    "low":    float(data["low"]),
                    "close":  float(data["close"]),
                    "volume": float(data.get("volume", 0)),
                }
        except Exception as e:
            logger.error(f"fetch_live_quote error [{symbol}]: {e}")
            return None

    async def get_current_seconds_into_minute(self) -> int:
        """Returns 0–59: how far into the current minute we are."""
        return datetime.now(timezone.utc).second

    # ── NEW: Multi-TF batch fetch ─────────────────────────────────────────────

    async def fetch_multi_tf(
        self, symbol: str
    ) -> dict[str, list[OHLCVBar]]:
        """
        Concurrently fetch M1, M5, M15, H1 bars for `symbol`.

        Returns:
            {
                "M1":  [OHLCVBar, ...],   # 60 bars, chronological
                "M5":  [OHLCVBar, ...],   # 60 bars
                "M15": [OHLCVBar, ...],   # 60 bars
                "H1":  [OHLCVBar, ...],   # 210 bars (EMA-200 warmup)
            }
        Missing / failed TFs return an empty list; the bias engine
        will simply assign 0.0 for that timeframe.
        """
        tasks = {
            tf: self.fetch_candles(symbol, cfg["interval"], cfg["count"])
            for tf, cfg in _TF_CONFIG.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        tf_bars: dict[str, list[OHLCVBar]] = {}
        for tf, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"fetch_multi_tf [{symbol}] {tf} failed: {result}")
                tf_bars[tf] = []
            else:
                tf_bars[tf] = [_to_ohlcv(c) for c in result]
        return tf_bars


# ── Conversion helper ─────────────────────────────────────────────────────────

def _to_ohlcv(raw: dict) -> OHLCVBar:
    return OHLCVBar(
        open=raw["open"],
        high=raw["high"],
        low=raw["low"],
        close=raw["close"],
        volume=raw.get("volume", 0.0),
    )