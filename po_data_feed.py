"""
po_data_feed.py — Pocket Option Direct Data Feed (v8.3-stable)

FIXES v8.3
────────────────────────────────────────
- Fixed non-candle payload parsing crash
- Added strict candle validation
- Skips asset metadata packets safely
- Improved websocket stability
- Better PocketOption compatibility
- Graceful reconnect handling
- Full RAW logging
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("websockets is required: pip install websockets")

logger = logging.getLogger("PocketOptionFeed")


def _decode(msg) -> str:
    if isinstance(msg, (bytes, bytearray)):
        return msg.decode("utf-8", errors="replace")
    return msg


def _normalize(symbol: str) -> str:
    return (
        symbol.lstrip("#")
        .replace("_otc", "")
        .replace("-OTC", "")
        .replace("-otc", "")
        .replace("/", "")
        .replace(" ", "")
        .upper()
    )


class PocketOptionFeed:

    OTC_ASSETS = {
        "EUR/USD-OTC": 66,
        "GBP/USD-OTC": 86,
        "USD/JPY-OTC": 93,
        "AUD/USD-OTC": 71,
        "USD/CAD-OTC": 91,
        "USD/CHF-OTC": 92,
        "EUR/JPY-OTC": 79,
        "EUR/GBP-OTC": 78,
        "GBP/JPY-OTC": 84,
        "NZD/USD-OTC": 90,
        "EUR/AUD-OTC": 44,
        "AUD/JPY-OTC": 69,
    }

    _NORM_TO_SYMBOL = {}

    _WS_DEMO = (
        "wss://demo-api-eu.po.market/socket.io/?EIO=4&transport=websocket"
    )

    _WS_LIVE = (
        "wss://api-eu.po.market/socket.io/?EIO=4&transport=websocket"
    )

    _HISTORY_EVENTS = {
        "loadHistoryPeriod",
        "history",
        "candles",
        "chartData",
        "updateChartData",
        "successHistory",
    }

    def __init__(self, ssid: str = "", demo: bool = True):

        self.demo = demo
        self._raw_ssid = ssid.strip()

        self.session_token = ""
        self.uid = ""

        self.keepalive_task = None

        if ssid:

            try:

                start = ssid.find("{")
                end = ssid.rfind("}") + 1

                if start >= 0 and end > start:

                    data = json.loads(ssid[start:end])

                    self.session_token = (
                        data.get("sessionToken")
                        or data.get("session", "")
                    )

                    self.uid = data.get("uid", "")

                logger.info(f"SSID parsed — uid={self.uid}")

            except Exception as e:
                logger.error(f"SSID parse failed: {e}")

        PocketOptionFeed._NORM_TO_SYMBOL = {
            _normalize(sym): sym
            for sym in self.OTC_ASSETS
        }

        self.ws_url = (
            self._WS_DEMO
            if demo
            else self._WS_LIVE
        )

    # ─────────────────────────────────────────────

    async def connect(self):
        return True

    async def close(self):

        if self.keepalive_task:
            self.keepalive_task.cancel()

    async def get_current_seconds_into_minute(self) -> int:
        return datetime.now(timezone.utc).second

    # ─────────────────────────────────────────────

    async def fetch_live_quote(
        self,
        symbol: str
    ) -> Optional[dict]:

        candles = await self.fetch_candles(
            symbol=symbol,
            interval="1min",
            count=2
        )

        if candles:
            return candles[-1]

        return None

    # ─────────────────────────────────────────────

    async def fetch_candles(
        self,
        symbol: str,
        interval: str = "1min",
        count: int = 60,
    ) -> list:

        asset_id = self.OTC_ASSETS.get(symbol)

        if asset_id is None:
            logger.error(f"Unknown asset: {symbol}")
            return []

        period = {
            "1min": 60,
            "1m": 60,
            "5min": 300,
            "15min": 900,
        }.get(interval, 60)

        for attempt in range(2):

            try:

                async with await self._open_authenticated_ws() as ws:

                    logger.info(
                        f"📡 Requesting candles for "
                        f"{symbol} "
                        f"(asset={asset_id}, period={period})"
                    )

                    candles = await self._request_candles(
                        ws,
                        asset_id,
                        period,
                        count
                    )

                    if candles:
                        return candles

            except Exception as e:

                logger.error(
                    f"fetch_candles failed: {e}",
                    exc_info=True
                )

            await asyncio.sleep(3)

        return []

    # ─────────────────────────────────────────────
    # WEBSOCKET
    # ─────────────────────────────────────────────

    async def _open_authenticated_ws(self):

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),
            "Origin": "https://pocketoption.com",
            "Referer": "https://pocketoption.com/",
        }

        ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=None,
            open_timeout=15,
            close_timeout=5,
            max_size=None,
        )

        msg = _decode(
            await asyncio.wait_for(
                ws.recv(),
                timeout=10
            )
        )

        logger.info(f"[OPEN] {repr(msg)}")

        if not msg.startswith("0"):
            raise Exception(f"Bad EIO open packet: {msg}")

        await ws.send("40")

        namespace_ok = False

        for _ in range(10):

            msg = _decode(
                await asyncio.wait_for(
                    ws.recv(),
                    timeout=10
                )
            )

            logger.info(
                f"[NAMESPACE] {repr(msg)}"
            )

            if msg == "2":
                await ws.send("3")
                continue

            if msg.startswith("40"):
                namespace_ok = True
                break

        if not namespace_ok:
            raise Exception("Namespace connect failed")

        await ws.send(self._raw_ssid)

        logger.info(f"Auth sent (uid={self.uid})")

        self.keepalive_task = asyncio.create_task(
            self._keepalive(ws)
        )

        await self._capture_asset_map(ws)

        await asyncio.sleep(2)

        return ws

    # ─────────────────────────────────────────────

    async def _keepalive(self, ws):

        try:

            while True:

                await asyncio.sleep(20)

                try:
                    await ws.send("2")
                    logger.debug("Ping sent")

                except:
                    break

        except asyncio.CancelledError:
            pass

    # ─────────────────────────────────────────────

    async def _capture_asset_map(self, ws):

        expecting_assets = False

        for _ in range(50):

            try:

                msg = _decode(
                    await asyncio.wait_for(
                        ws.recv(),
                        timeout=5
                    )
                )

            except asyncio.TimeoutError:
                continue

            except ConnectionClosed:
                logger.warning(
                    "WS closed during auth"
                )
                return

            logger.info(
                f"[AUTH PHASE] {repr(msg[:200])}"
            )

            if msg == "2":
                await ws.send("3")
                continue

            if (
                msg.startswith("451-")
                and "updateAssets" in msg
            ):
                expecting_assets = True
                continue

            if expecting_assets and msg.startswith("[["):

                self._build_asset_map(msg)

                logger.info(
                    "✅ Asset map captured"
                )

                return

    # ─────────────────────────────────────────────

    def _build_asset_map(self, payload_str: str):

        try:
            assets = json.loads(payload_str)

        except Exception as e:
            logger.error(
                f"Asset parse failed: {e}"
            )
            return

        new_map = {}

        for entry in assets:

            if not isinstance(entry, list):
                continue

            if len(entry) < 2:
                continue

            asset_id = entry[0]
            po_symbol = str(entry[1])

            norm = _normalize(po_symbol)

            if norm in self._NORM_TO_SYMBOL:

                canonical = (
                    self._NORM_TO_SYMBOL[norm]
                )

                new_map[canonical] = asset_id

        if new_map:

            self.OTC_ASSETS.update(new_map)

            logger.info(
                f"✅ Asset map updated "
                f"({len(new_map)} assets)"
            )

    # ─────────────────────────────────────────────
    # CANDLES
    # ─────────────────────────────────────────────

    async def _request_candles(
        self,
        ws,
        asset_id: int,
        period: int,
        count: int,
    ) -> list:

        change_symbol = (
            '42["changeSymbol",{'
            f'"asset":{asset_id},'
            f'"period":{period}'
            '}]'
        )

        history_request = (
            '42["loadHistoryPeriod",{'
            f'"asset":{asset_id},'
            f'"period":{period},'
            f'"time":{int(time.time())},'
            '"index":0,'
            '"offset":0'
            '}]'
        )

        logger.info(f"→ {change_symbol}")
        await ws.send(change_symbol)

        await asyncio.sleep(2)

        logger.info(f"→ {history_request}")
        await ws.send(history_request)

        deadline = time.time() + 30

        try:

            while time.time() < deadline:

                try:

                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=5
                    )

                    msg = _decode(raw)

                    logger.info(
                        f"[FULL RAW] {repr(msg)}"
                    )

                except asyncio.TimeoutError:
                    continue

                except ConnectionClosed:

                    logger.warning(
                        "WS closed by PocketOption"
                    )

                    break

                if msg == "2":
                    await ws.send("3")
                    continue

                # ─────────────────────
                # 451 events
                # ─────────────────────

                if msg.startswith("451-"):

                    try:

                        json_part = (
                            msg[msg.index("["):]
                        )

                        payload = json.loads(
                            json_part
                        )

                        event = payload[0]

                        logger.info(
                            f"[451 EVENT] {event}"
                        )

                    except:
                        continue

                    continue

                # ─────────────────────
                # Binary attachment
                # ─────────────────────

                if msg.startswith("[["):

                    candles = self._parse_candles(
                        msg,
                        count
                    )

                    if candles:

                        logger.info(
                            f"✅ Parsed "
                            f"{len(candles)} candles "
                            f"from binary attachment"
                        )

                        if self.keepalive_task:
                            self.keepalive_task.cancel()

                        return candles

                # ─────────────────────
                # 42 events
                # ─────────────────────

                if msg.startswith("42"):

                    try:

                        payload = json.loads(
                            msg[2:]
                        )

                    except:
                        continue

                    if not isinstance(payload, list):
                        continue

                    if len(payload) < 2:
                        continue

                    event = payload[0]
                    data = payload[1]

                    logger.info(
                        f"← EVENT: {event}"
                    )

                    if event in self._HISTORY_EVENTS:

                        candles = self._parse_candles(
                            data,
                            count
                        )

                        if candles:

                            logger.info(
                                f"✅ Parsed "
                                f"{len(candles)} candles"
                            )

                            if self.keepalive_task:
                                self.keepalive_task.cancel()

                            return candles

        finally:

            if self.keepalive_task:
                self.keepalive_task.cancel()

        logger.warning(
            "❌ Candle request timed out"
        )

        return []

    # ─────────────────────────────────────────────
    # PARSER
    # ─────────────────────────────────────────────

    def _parse_candles(
        self,
        data,
        count: int
    ) -> list:

        candles = []

        try:

            if isinstance(data, str):

                try:
                    data = json.loads(data)
                except:
                    pass

            raw = None

            if isinstance(data, list):
                raw = data

            elif isinstance(data, dict):

                raw = (
                    data.get("candles")
                    or data.get("history")
                    or data.get("data")
                    or data.get("result")
                )

            if not raw:
                return []

            for c in raw[-count:]:

                # ─────────────────────
                # Dict candle
                # ─────────────────────

                if isinstance(c, dict):

                    try:

                        candles.append({
                            "open": float(
                                c.get("open", c.get("o", 0))
                            ),
                            "high": float(
                                c.get("high", c.get("h", 0))
                            ),
                            "low": float(
                                c.get("low", c.get("l", 0))
                            ),
                            "close": float(
                                c.get("close", c.get("c", 0))
                            ),
                            "volume": float(
                                c.get("volume", c.get("v", 0))
                            ),
                        })

                    except:
                        continue

                # ─────────────────────
                # List candle
                # ─────────────────────

                elif (
                    isinstance(c, (list, tuple))
                    and len(c) >= 5
                ):

                    # FIX v8.3
                    # Validate candle structure

                    try:

                        open_price = float(c[1])
                        high_price = float(c[2])
                        low_price = float(c[3])
                        close_price = float(c[4])

                    except (ValueError, TypeError):

                        logger.warning(
                            f"Skipping non-candle payload: "
                            f"{repr(c[:10])}"
                        )

                        continue

                    candles.append({
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": (
                            float(c[5])
                            if (
                                len(c) > 5
                                and isinstance(
                                    c[5],
                                    (int, float)
                                )
                            )
                            else 0.0
                        ),
                    })

        except Exception as e:

            logger.error(
                f"Candle parse failed: {e}",
                exc_info=True
            )

        return candles