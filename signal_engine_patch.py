"""
╔══════════════════════════════════════════════════════════════╗
║      signal_engine_patch.py — CRT INTEGRATION GUIDE         ║
║                                                              ║
║  This file shows exactly how to wire CRTVolumeEngine into   ║
║  your existing ConfluenceEngine and BinarySignalBot.        ║
║                                                              ║
║  Step 1: Drop crt_volume_engine.py into your project folder ║
║  Step 2: Apply the changes shown below to signal_engine.py  ║
║  Step 3: Apply the bot.py patch for CRT signal formatting   ║
╚══════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════
# PATCH A — signal_engine.py
# Add these imports at the TOP of your signal_engine.py
# ══════════════════════════════════════════════════════════════

# from crt_volume_engine import CRTVolumeEngine, CRTSignal


# ══════════════════════════════════════════════════════════════
# PATCH B — ConfluenceEngine class
# Inside your ConfluenceEngine.__init__(), add:
# ══════════════════════════════════════════════════════════════

class ConfluenceEngineExtension:
    """
    Mixin / reference showing how to extend your ConfluenceEngine.
    Do NOT use this class directly — copy the relevant parts
    into your existing ConfluenceEngine.
    """

    def __init__(self):
        # ── EXISTING CODE (keep as-is) ──────────────────────
        # ... your existing __init__ code ...

        # ── ADD THIS LINE ────────────────────────────────────
        self.crt_engine = CRTVolumeEngine(min_confidence=62)

    def evaluate(
        self,
        asset:           str,
        candles:         list,
        forming_candle,
        elapsed_seconds: int,
        timestamp:       str,
    ):
        """
        Extended evaluate() — runs your existing analysis
        PLUS CRT + Volume analysis, then merges confidence scores.

        Replace your existing evaluate() with this version,
        keeping all your original logic intact.
        """

        # ── 1. RUN YOUR EXISTING ANALYSIS (keep all original code) ──
        existing_signal = self._your_original_evaluate(
            asset, candles, forming_candle, elapsed_seconds, timestamp
        )

        # ── 2. RUN CRT + VOLUME ANALYSIS ────────────────────────────
        crt_signal = self.crt_engine.analyse(
            asset           = asset,
            candles         = candles,
            forming_candle  = forming_candle,
            elapsed_seconds = elapsed_seconds,
        )

        # ── 3. MERGE RESULTS ─────────────────────────────────────────
        return self._merge_signals(existing_signal, crt_signal, timestamp, asset)

    def _merge_signals(self, base_signal, crt_signal, timestamp, asset):
        """
        Merging strategy — three scenarios:

        Scenario A: BOTH signals agree on direction
          → HIGH CONFIDENCE: average confidence + 10% bonus
          → Merge confluence factors from both

        Scenario B: CRT signal only (no base signal)
          → Use CRT signal confidence as-is
          → Label as CRT-exclusive signal

        Scenario C: Signals DISAGREE on direction
          → SUPPRESS — opposing signals = choppy market
          → Return None (no trade)

        Scenario D: Base signal only (no CRT signal)
          → Use existing signal unchanged
        """

        # Scenario D: no CRT, return existing unchanged
        if crt_signal is None:
            return base_signal

        # Scenario B: CRT only, no base signal
        if base_signal is None:
            return self._crt_to_final_signal(crt_signal, timestamp, asset)

        # Scenario C: Direction conflict — suppress
        if base_signal.direction != crt_signal.direction:
            return None  # Market is ambiguous, skip this candle

        # Scenario A: Both agree — boost confidence
        merged_confidence = min(
            100,
            int((base_signal.confidence + crt_signal.confidence) / 2) + 10
        )

        # Merge confluence factors
        merged_factors = base_signal.confluence_factors.copy()
        merged_factors.append("━━━ CRT + VOLUME LAYER ━━━")
        merged_factors.extend(crt_signal.confluence_factors)

        # Update the existing FinalSignal with merged data
        base_signal.confidence         = merged_confidence
        base_signal.confluence_factors = merged_factors

        return base_signal

    def _crt_to_final_signal(self, crt: "CRTSignal", timestamp: str, asset: str):
        """
        Convert a standalone CRTSignal into a FinalSignal
        when there is no base candle psychology signal.

        Adapt the field names to match YOUR FinalSignal dataclass.
        """
        from signal_engine import FinalSignal  # your existing class

        sweep_emoji = "📉⚡" if crt.sweep_side == "HIGH_SWEPT" else "📈⚡"
        sweep_label = "CRT High Sweep → PUT" if crt.sweep_side == "HIGH_SWEPT" else "CRT Low Sweep → CALL"

        # Expiry: next candle (standard M1 binary)
        expiry = "1 minute (next candle close)"

        entry_note = (
            "Enter at open of next candle after sweep rejection. "
            + ("OTC: allow 1-tick spread buffer." if asset.endswith("-OTC") else "")
        )

        return FinalSignal(
            asset               = asset,
            direction           = crt.direction,
            confidence          = crt.confidence,
            candle_emoji        = sweep_emoji,
            candle_pattern      = sweep_label,
            candle_description  = (
                f"CRT {crt.sweep_side.replace('_', ' ').title()} — "
                f"Range: {crt.zone.range_low:.5f}–{crt.zone.range_high:.5f} | "
                f"Vol spike: {crt.volume.volume_ratio:.1f}×"
            ),
            confluence_factors  = crt.confluence_factors,
            expiry              = expiry,
            entry_note          = entry_note,
            timestamp           = timestamp,
            pre_close           = crt.is_pre_close,
        )


# ══════════════════════════════════════════════════════════════
# PATCH C — bot.py MessageFormatter
# Add the CRT-enhanced signal message block
# ══════════════════════════════════════════════════════════════

def signal_message_with_crt(sig, crt_signal=None) -> str:
    """
    Enhanced signal message formatter that adds a CRT Zone block
    when CRT data is available. Drop this into your MessageFormatter
    class or call it from signal_message().
    """

    direction_emoji = "📈" if sig.direction == "CALL" else "📉"
    direction_color = "🟢" if sig.direction == "CALL" else "🔴"
    pre_alert       = "⚡ *PRE-CLOSE ALERT*\n" if sig.pre_close else ""

    filled  = int(sig.confidence / 10)
    bar     = "█" * filled + "░" * (10 - filled)

    if sig.confidence >= 85:
        strength = "🔥 EXTREME"
    elif sig.confidence >= 73:
        strength = "💪 STRONG"
    elif sig.confidence >= 62:
        strength = "✅ MODERATE"
    else:
        strength = "⚠️ WEAK"

    confluence_text = "\n".join(f"  {f}" for f in sig.confluence_factors)

    # CRT Zone block (appended when CRT data available)
    crt_block = ""
    if crt_signal:
        crt_block = (
            f"\n📐 *CRT Zone:*\n"
            f"  🔴 Range High: `{crt_signal.zone.range_high:.5f}`\n"
            f"  🟢 Range Low:  `{crt_signal.zone.range_low:.5f}`\n"
            f"  💥 Sweep Side: `{crt_signal.sweep_side.replace('_', ' ')}`\n"
            f"  📊 Vol Ratio:  `{crt_signal.volume.volume_ratio:.2f}×` avg\n"
            f"  {'✅ Rejection Confirmed' if crt_signal.rejection_confirmed else '⏳ Awaiting Rejection'}\n"
        )

    msg = (
        f"{pre_alert}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{direction_emoji} *BINARY SIGNAL* {direction_color}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *Asset:*   `{sig.asset}`\n"
        f"🎯 *Action:*  *{sig.direction}*  ({'UP ↑' if sig.direction == 'CALL' else 'DOWN ↓'})\n"
        f"⏱ *Expiry:* `{sig.expiry}`\n\n"
        f"📐 *Pattern:* {sig.candle_emoji} _{sig.candle_pattern}_\n"
        f"💬 _{sig.candle_description}_\n"
        f"{crt_block}\n"
        f"📊 *Confidence:* `{sig.confidence}%` {strength}\n"
        f"`[{bar}]`\n\n"
        f"🔍 *Confluence Analysis:*\n"
        f"{confluence_text}\n\n"
        f"🔔 *Entry:* _{sig.entry_note}_\n"
        f"🕐 *Time:*  `{sig.timestamp}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Risk responsibly. 1–3% per trade._"
    )
    return msg


# ══════════════════════════════════════════════════════════════
# PATCH D — data_feed.py
# Your candle dicts MUST include "volume".
# If Twelve Data returns volume, ensure your TwelveDataFeed
# maps it correctly. Expected format per candle dict:
#   {
#     "open":   float,
#     "high":   float,
#     "low":    float,
#     "close":  float,
#     "volume": float,   ← REQUIRED for CRT volume analysis
#   }
# If volume is missing, the engine falls back gracefully
# (volume analysis is skipped, score reduced accordingly).
# ══════════════════════════════════════════════════════════════

def ensure_volume_in_candles(candles: list) -> list:
    """
    Safety wrapper: ensures every candle dict has a 'volume' key.
    Add this call in your TwelveDataFeed.fetch_candles() before returning.
    """
    for c in candles:
        if "volume" not in c or c["volume"] is None:
            c["volume"] = 0.0   # engine handles zero-volume gracefully
    return candles
