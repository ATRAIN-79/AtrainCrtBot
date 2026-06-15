"""
╔══════════════════════════════════════════════════════════════╗
║          BINARY SIGNAL ROBOT — TELEGRAM BOT  v2.2            ║
║   Expert 1-Minute Binary Options Signal Bot                  ║
║   Powered by Candle Psychology + CRT + RSI + ATR Filters     ║
║   + Multi-TF Bias Engine (Binary_Forex_Forecast_Pro v4.00)   ║
║                                                              ║
║   NEW IN v2.2                                                ║
║   ✦ fetch_multi_tf() now called on every scan — Layer 6 live ║
║   ✦ Correlation trigger: extra scan fires when MTF alignment  ║
║     score clears 0.30 threshold (every 30s check)            ║
║   ✦ SCAN_INTERVAL kept at 60s; correlation job runs at 30s   ║
║   ✦ Duplicate-signal guard: same direction not re-sent        ║
║     within 90s on same pair                                  ║
║   ✦ Confidence band widened in signal_engine (55–72, 85–97)  ║
║                                                              ║
║   Commands:                                                  ║
║   /start   — Welcome + category selector                     ║
║   /pairs   — Back to category/pair selector any time         ║
║   /stats   — Performance stats for this session              ║
║   /stop    — Pause all signals                               ║
║   /resume  — Resume + re-open pair selector                  ║
║   /help    — Command reference                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, JobQueue
)
from telegram.constants import ParseMode

from data_feed import TwelveDataFeed, ASSET_CATALOGUE, find_asset
from signal_engine import ConfluenceEngine, FinalSignal
from candle_psychology import Candle

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BinaryRobot")

OTC_PAIRS = {
    "EUR/USD-OTC": "EUR/USD",
    "GBP/USD-OTC": "GBP/USD",
    "USD/JPY-OTC": "USD/JPY",
    "AUD/USD-OTC": "AUD/USD",
    "USD/CAD-OTC": "USD/CAD",
    "USD/CHF-OTC": "USD/CHF",
    "EUR/JPY-OTC": "EUR/JPY",
    "EUR/GBP-OTC": "EUR/GBP",
    "GBP/JPY-OTC": "GBP/JPY",
    "NZD/USD-OTC": "NZD/USD",
    "EUR/AUD-OTC": "EUR/AUD",
    "AUD/JPY-OTC": "AUD/JPY",
}

CATEGORY_DISPLAY = [
    ("forex",       "💱", "Forex Majors",
     list(ASSET_CATALOGUE["Forex"].keys()), 3),
    ("otc",         "🔵", "OTC Pairs",
     list(OTC_PAIRS.keys()), 2),
    ("crypto",      "₿",  "Crypto",
     list(ASSET_CATALOGUE["Crypto"].keys()), 3),
    ("commodities", "🥇", "Commodities",
     list(ASSET_CATALOGUE["Commodities"].keys()), 2),
    ("stocks",      "📈", "Stocks",
     list(ASSET_CATALOGUE["Stocks"].keys()), 3),
    ("indices",     "📊", "Indices",
     list(ASSET_CATALOGUE["Indices"].keys()), 3),
]

_CAT_BY_ID  = {cid: (icon, label, syms, cols) for cid, icon, label, syms, cols in CATEGORY_DISPLAY}
_SYM_TO_CAT = {}
for cid, icon, label, syms, cols in CATEGORY_DISPLAY:
    for s in syms:
        _SYM_TO_CAT[s] = cid


def resolve_symbol(symbol: str) -> str:
    return OTC_PAIRS.get(symbol, symbol)


def build_category_keyboard() -> InlineKeyboardMarkup:
    keyboard = []
    row = []
    for cid, icon, label, _, _ in CATEGORY_DISPLAY:
        row.append(InlineKeyboardButton(f"{icon} {label}", callback_data=f"cat_{cid}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)


def build_pairs_keyboard(cat_id: str) -> InlineKeyboardMarkup:
    icon, label, syms, cols = _CAT_BY_ID[cat_id]
    keyboard = []
    row = []
    for sym in syms:
        display = sym.replace("-OTC", " OTC")
        row.append(InlineKeyboardButton(display, callback_data=f"pair_{sym}"))
        if len(row) == cols:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("⬅️ Categories", callback_data="cat_menu")])
    return InlineKeyboardMarkup(keyboard)


def build_post_signal_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📂 Change Pair",  callback_data="change_pair"),
        InlineKeyboardButton("🔄 Same Pair",    callback_data="same_pair"),
    ]])


def build_paused_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("▶️ Resume & Pick Pair", callback_data="resume_pick"),
    ]])


class MessageFormatter:

    _SEPARATORS = {
        "━━━ CRT + VOLUME LAYER ━━━",
        "━━━ ACCURACY FILTERS ━━━",
        "━━━ ICT FVG STRUCTURE ━━━",
    }

    @staticmethod
    def _top_confluences(factors: list, n: int = 2) -> list:
        cleaned  = [f for f in factors if f not in MessageFormatter._SEPARATORS]
        priority = [f for f in cleaned if f.startswith("✅")]
        rest     = [f for f in cleaned if not f.startswith("✅")]
        return (priority + rest)[:n]

    @staticmethod
    def signal_message(sig: FinalSignal, elapsed: int = 0,
                       triggered_by: str = "scan") -> str:
        direction_emoji = "📈" if sig.direction == "CALL" else "📉"
        direction_color = "🟢" if sig.direction == "CALL" else "🔴"

        filled   = int(sig.confidence / 10)
        bar      = "█" * filled + "░" * (10 - filled)
        strength = (
            "🔥 EXTREME"   if sig.confidence >= 85 else
            "💪 STRONG"    if sig.confidence >= 73 else
            "✅ MODERATE"  if sig.confidence >= 62 else
            "⚠️ WATCH"
        )

        if sig.pre_close:
            secs_left   = max(0, 60 - elapsed)
            entry_block = (
                f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                f"┃  ⚡⚡  E N T E R   N O W  ⚡⚡  ┃\n"
                f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
                f"_Candle closing in ~{secs_left}s — enter on next open_"
            )
        else:
            entry_block = (
                f"┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                f"┃  ⏱  N E X T  C A N D L E  ⏱  ┃\n"
                f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
                f"_Enter at the OPEN of the next candle_"
            )

        trigger_tag = ""
        if triggered_by == "correlation":
            trigger_tag = "\n⚡ _Triggered by MTF correlation alignment_"

        top2           = MessageFormatter._top_confluences(sig.confluence_factors, n=2)
        confluence_text = "\n".join(f"  {f}" for f in top2) if top2 else "  —"

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{direction_emoji} *{sig.direction}*  {direction_color}  `{sig.asset}`\n"
            f"⏱ *Expiry:* `{sig.expiry}`  |  🕐 `{sig.timestamp}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{entry_block}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 {sig.candle_emoji} _{sig.candle_pattern}_\n"
            f"📊 *Confidence:* `{sig.confidence}%` {strength}  `[{bar}]`\n"
            f"{trigger_tag}\n\n"
            f"🔍 *Why:*\n"
            f"{confluence_text}\n\n"
            f"_Risk responsibly. 1–3% per trade._"
        )

    @staticmethod
    def no_signal_message(symbol: str) -> str:
        return (
            f"🔍 *No Signal — `{symbol}`*\n\n"
            f"No high-confluence setup detected.\n"
            f"_Threshold: 55% — market may be choppy_\n\n"
            f"Tap *Same Pair* to re-scan or *Change Pair* to pick another."
        )

    @staticmethod
    def welcome_message(name: str) -> str:
        return (
            f"👋 Welcome, *{name}*!\n\n"
            f"🤖 *Binary Signal Robot v2.2*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"I scan live M1 candles using:\n"
            f"  🕯 *Candle Psychology* (12+ patterns)\n"
            f"  📐 *CRT + Volume* sweep detection\n"
            f"  📊 *RSI + ATR* accuracy filters\n"
            f"  🕰 *EMA 9/21/50* trend alignment\n"
            f"  🏔 *Support & Resistance* levels\n"
            f"  📡 *Multi-TF Bias* (M1/M5/M15/H1)\n\n"
            f"*How it works:*\n"
            f"  1️⃣  Pick a *category* below\n"
            f"  2️⃣  Tap a *pair* — signal fires instantly\n"
            f"  3️⃣  Bot scans every 60s + fires on MTF correlation\n"
            f"  4️⃣  After each signal — tap *Change* or *Same*\n\n"
            f"📡 *Assets:* Forex · OTC · Crypto · Commodities · Stocks · Indices\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Select a category to begin:_"
        )

    @staticmethod
    def category_menu_text() -> str:
        return (
            "📂 *Select a Category*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "_Tap a category to see available pairs:_"
        )

    @staticmethod
    def pairs_menu_text(icon: str, label: str, count: int) -> str:
        return (
            f"{icon} *{label}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Tap a pair to start receiving signals:_\n"
            f"_{count} pairs available_"
        )


class BotState:
    def __init__(self):
        self.active_pair:    dict[int, str]           = {}
        self.paused:         set[int]                 = set()
        self.stats:          dict[int, dict]          = {}
        # duplicate-signal guard: {chat_id: (direction, timestamp_float)}
        self.last_signal:    dict[int, tuple]         = {}

    def is_duplicate(self, chat_id: int, direction: str, window: int = 60) -> bool:
        """True if same direction was sent within `window` seconds."""
        import time
        entry = self.last_signal.get(chat_id)
        if not entry:
            return False
        last_dir, last_ts = entry
        return last_dir == direction and (time.time() - last_ts) < window

    def record_signal(self, chat_id: int, direction: str):
        import time
        self.last_signal[chat_id] = (direction, time.time())


class BinarySignalBot:

    SCAN_INTERVAL        = 60    # regular candle-close scan (seconds)
    CORRELATION_INTERVAL = 30    # MTF correlation check (seconds)
    PRE_CLOSE_TRIGGER    = 45

    def __init__(self, telegram_token: str, twelve_api_key: str):
        self.token     = telegram_token
        self.feed      = TwelveDataFeed(twelve_api_key)
        self.engine    = ConfluenceEngine()
        self.formatter = MessageFormatter()
        self.state     = BotState()
        self.app       = Application.builder().token(telegram_token).job_queue(JobQueue()).build()
        self._register_handlers()

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start",  self.cmd_start))
        self.app.add_handler(CommandHandler("pairs",  self.cmd_pairs))
        self.app.add_handler(CommandHandler("help",   self.cmd_help))
        self.app.add_handler(CommandHandler("stats",  self.cmd_stats))
        self.app.add_handler(CommandHandler("stop",   self.cmd_stop))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

    def _ensure_state(self, chat_id: int):
        if chat_id not in self.state.stats:
            self.state.stats[chat_id] = {"signals_sent": 0, "calls": 0, "puts": 0}

    def _start_scan_jobs(self, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
        """Start both the regular 60s scan and the 30s correlation check."""
        # Regular scan
        job_name = f"scan_{chat_id}"
        for job in ctx.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        ctx.job_queue.run_repeating(
            self._auto_scan_job,
            interval=self.SCAN_INTERVAL,
            first=self.SCAN_INTERVAL,
            name=job_name,
            data={"chat_id": chat_id}
        )
        # Correlation trigger
        corr_name = f"corr_{chat_id}"
        for job in ctx.job_queue.get_jobs_by_name(corr_name):
            job.schedule_removal()
        ctx.job_queue.run_repeating(
            self._correlation_check_job,
            interval=self.CORRELATION_INTERVAL,
            first=self.CORRELATION_INTERVAL,
            name=corr_name,
            data={"chat_id": chat_id}
        )

    def _record_stat(self, chat_id: int, sig: FinalSignal):
        self._ensure_state(chat_id)
        s = self.state.stats[chat_id]
        s["signals_sent"] += 1
        s["calls"] += (1 if sig.direction == "CALL" else 0)
        s["puts"]  += (1 if sig.direction == "PUT"  else 0)

    # ── Commands ──────────────────────────────────────────────────────────────

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self._ensure_state(chat_id)
        name = update.effective_user.first_name or "Trader"
        await update.message.reply_text(
            self.formatter.welcome_message(name),
            reply_markup=build_category_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_pairs(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            self.formatter.category_menu_text(),
            reply_markup=build_category_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        active  = self.state.active_pair.get(chat_id)
        active_text = f"`{active}`" if active else "_none selected_"
        msg = (
            "🤖 *Binary Signal Robot v2.2 — Help*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "*/start*  — Welcome screen + pair selector\n"
            "*/pairs*  — Open category/pair selector any time\n"
            "*/stats*  — Session performance stats\n"
            "*/stop*   — Pause auto-signals\n"
            "*/resume* — Resume + re-open pair selector\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "*How signals fire:*\n"
            "  🕐 Every *60s* — regular candle-close scan\n"
            "  ⚡ Every *30s* — MTF correlation check\n"
            "     Fires extra signal when M1/M5/M15/H1 align\n\n"
            f"📡 *Currently watching:* {active_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "*Entry Types:*\n"
            "  ⚡ *ENTER NOW* — Pre-close alert (45s+ mark)\n"
            "  ⏱ *NEXT CANDLE* — Enter at open of next candle\n\n"
            "_Minimum confidence: 55%_"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def cmd_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self._ensure_state(chat_id)
        s      = self.state.stats[chat_id]
        active = self.state.active_pair.get(chat_id, "None")
        total  = s["signals_sent"]
        await update.message.reply_text(
            f"📊 *Session Statistics*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📡 Active Pair:    `{active}`\n"
            f"🔔 Signals Sent:   `{total}`\n"
            f"📈 CALL Signals:   `{s['calls']}`\n"
            f"📉 PUT Signals:    `{s['puts']}`\n\n"
            f"_Stats reset each session._",
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self.state.paused.add(chat_id)
        # Remove jobs
        for prefix in ("scan_", "corr_"):
            for job in ctx.job_queue.get_jobs_by_name(f"{prefix}{chat_id}"):
                job.schedule_removal()
        await update.message.reply_text(
            "⏸ *Signals paused.*\n\n_Tap Resume to restart and pick a pair._",
            reply_markup=build_paused_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self.state.paused.discard(chat_id)
        await update.message.reply_text(
            "▶️ *Signals resumed!*\n\n_Pick a pair to start scanning:_",
            reply_markup=build_category_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def handle_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query   = update.callback_query
        await query.answer()
        data    = query.data
        chat_id = update.effective_chat.id
        self._ensure_state(chat_id)
        try:
            await self._dispatch_callback(query, data, chat_id, ctx)
        except Exception as e:
            logger.error(f"Callback error [{data}] chat={chat_id}: {e}")

    async def _dispatch_callback(self, query, data: str, chat_id: int, ctx):
        if data == "cat_menu":
            await query.edit_message_text(
                self.formatter.category_menu_text(),
                reply_markup=build_category_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        elif data.startswith("cat_"):
            cat_id = data[4:]
            if cat_id not in _CAT_BY_ID:
                return
            icon, label, syms, cols = _CAT_BY_ID[cat_id]
            await query.edit_message_text(
                self.formatter.pairs_menu_text(icon, label, len(syms)),
                reply_markup=build_pairs_keyboard(cat_id),
                parse_mode=ParseMode.MARKDOWN
            )
        elif data.startswith("pair_"):
            symbol = data[5:]
            await self._select_pair_and_analyse(query, chat_id, symbol, ctx)
        elif data == "change_pair":
            await query.edit_message_text(
                self.formatter.category_menu_text(),
                reply_markup=build_category_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )
        elif data == "same_pair":
            symbol = self.state.active_pair.get(chat_id)
            if not symbol:
                await query.edit_message_text(
                    "⚠️ No active pair found.\n_Pick a pair to continue:_",
                    reply_markup=build_category_keyboard(),
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            await query.edit_message_text(
                f"⏳ *Re-scanning `{symbol}`...*",
                parse_mode=ParseMode.MARKDOWN
            )
            await self._run_analysis_and_send(chat_id, symbol, ctx.bot)
        elif data == "resume_pick":
            self.state.paused.discard(chat_id)
            await query.edit_message_text(
                "▶️ *Resumed!*\n\n_Pick a pair to start scanning:_",
                reply_markup=build_category_keyboard(),
                parse_mode=ParseMode.MARKDOWN
            )

    async def _select_pair_and_analyse(self, query, chat_id, symbol, ctx):
        self.state.active_pair[chat_id] = symbol
        self.state.paused.discard(chat_id)
        self.state.last_signal.pop(chat_id, None)   # clear duplicate guard on pair change
        await query.edit_message_text(
            f"⏳ *Analysing `{symbol}`...*\n_Scanning M1 candles — please wait_",
            parse_mode=ParseMode.MARKDOWN
        )
        await self._run_analysis_and_send(chat_id, symbol, ctx.bot)
        self._start_scan_jobs(ctx, chat_id)

    # ── Core analysis ─────────────────────────────────────────────────────────

    async def _run_analysis_and_send(
        self,
        chat_id: int,
        symbol: str,
        bot,
        triggered_by: str = "scan"
    ):
        signal  = await self._analyse_asset(symbol)
        elapsed = await self.feed.get_current_seconds_into_minute()

        if signal:
            # Duplicate-signal guard
            if self.state.is_duplicate(chat_id, signal.direction):
                logger.info(
                    f"Duplicate {signal.direction} suppressed for {chat_id} on {symbol}"
                )
                return
            self.state.record_signal(chat_id, signal.direction)
            self._record_stat(chat_id, signal)
            try:
                await bot.send_message(
                    chat_id      = chat_id,
                    text         = self.formatter.signal_message(
                                       signal, elapsed, triggered_by),
                    parse_mode   = ParseMode.MARKDOWN,
                    reply_markup = build_post_signal_keyboard()
                )
            except Exception as e:
                logger.error(f"Send signal error to {chat_id}: {e}")
        else:
            if triggered_by == "scan":   # only show "no signal" on regular scans
                try:
                    await bot.send_message(
                        chat_id      = chat_id,
                        text         = self.formatter.no_signal_message(symbol),
                        parse_mode   = ParseMode.MARKDOWN,
                        reply_markup = build_post_signal_keyboard()
                    )
                except Exception as e:
                    logger.error(f"Send no-signal error to {chat_id}: {e}")

    async def _auto_scan_job(self, ctx: ContextTypes.DEFAULT_TYPE):
        """Regular 60s candle-close scan."""
        chat_id = ctx.job.data["chat_id"]
        if chat_id in self.state.paused:
            return
        symbol = self.state.active_pair.get(chat_id)
        if not symbol:
            return
        await self._run_analysis_and_send(chat_id, symbol, ctx.bot, triggered_by="scan")

    async def _correlation_check_job(self, ctx: ContextTypes.DEFAULT_TYPE):
        """
        Every 30s: fetch multi-TF bars and check MTF alignment.
        If alignment score clears threshold, fire an immediate analysis.
        Silent if no strong alignment — no "no signal" message sent.
        """
        chat_id = ctx.job.data["chat_id"]
        if chat_id in self.state.paused:
            return
        symbol = self.state.active_pair.get(chat_id)
        if not symbol:
            return
        try:
            fetch_symbol = resolve_symbol(symbol)
            tf_bars      = await self.feed.fetch_multi_tf(fetch_symbol)
            ready, hint  = self.engine.check_correlation_ready(tf_bars)
            if ready:
                logger.info(
                    f"Correlation alignment detected for {symbol} "
                    f"— direction hint: {hint} — triggering scan"
                )
                await self._run_analysis_and_send(
                    chat_id, symbol, ctx.bot, triggered_by="correlation"
                )
        except Exception as e:
            logger.error(f"Correlation check error [{symbol}]: {e}")

    async def _analyse_asset(self, symbol: str) -> Optional[FinalSignal]:
        """
        Full analysis: fetches M1 candles + multi-TF bars concurrently,
        then passes both to the engine so Layer 6 (MTF Bias) is active.
        """
        try:
            fetch_symbol = resolve_symbol(symbol)

            # Fetch M1 candles and multi-TF bars concurrently
            m1_task  = self.feed.fetch_candles(fetch_symbol, interval="1min", count=60)
            mtf_task = self.feed.fetch_multi_tf(fetch_symbol)
            quote_task = self.feed.fetch_live_quote(fetch_symbol)

            raw_candles, tf_bars, raw_forming = await asyncio.gather(
                m1_task, mtf_task, quote_task, return_exceptions=True
            )

            # Handle partial failures gracefully
            if isinstance(raw_candles, Exception):
                logger.error(f"M1 fetch failed [{symbol}]: {raw_candles}")
                return None
            if isinstance(tf_bars, Exception):
                logger.warning(f"Multi-TF fetch failed [{symbol}]: {tf_bars}")
                tf_bars = None
            if isinstance(raw_forming, Exception):
                raw_forming = None

            if len(raw_candles) < 10:
                return None

            candles = [
                Candle(open=c["open"], high=c["high"], low=c["low"],
                       close=c["close"], volume=c.get("volume", 0.0))
                for c in raw_candles
            ]
            forming = (
                Candle(
                    open=raw_forming["open"], high=raw_forming["high"],
                    low=raw_forming["low"],   close=raw_forming["close"],
                    volume=raw_forming.get("volume", 0.0)
                ) if raw_forming and not isinstance(raw_forming, Exception) else None
            )
            elapsed = await self.feed.get_current_seconds_into_minute()
            ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            return self.engine.evaluate(
                asset=symbol,
                candles=candles,
                forming_candle=forming,
                elapsed_seconds=elapsed,
                timestamp=ts,
                tf_bars=tf_bars,       # ← Layer 6 now receives real data
            )
        except Exception as e:
            logger.error(f"Analysis error [{symbol}]: {e}")
            return None

    async def run(self):
        logger.info("🤖 Binary Signal Robot v2.2 starting...")
        try:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)
            logger.info("🚀 Bot is live — waiting for commands...")
            await asyncio.Event().wait()
        finally:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
            except Exception:
                pass
            await self.feed.close()


if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TWELVE_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
    if not TWELVE_API_KEY:
        raise ValueError("TWELVE_DATA_API_KEY not set in .env")
    bot = BinarySignalBot(TELEGRAM_TOKEN, TWELVE_API_KEY)
    asyncio.run(bot.run())