"""
╔══════════════════════════════════════════════════════════════╗
║              ACCURACY FILTER SUITE  v1.0                     ║
║   RSI · ATR · Momentum · Session · Cooldown                  ║
║                                                              ║
║   Plug-in accuracy layer for ConfluenceEngine.               ║
║   All filters compute from the existing candle array —       ║
║   zero extra API calls, zero added latency.                  ║
║                                                              ║
║   OTC-AWARE: Every filter has OTC-specific logic since       ║
║   broker-side price offsets and 24/7 availability change     ║
║   how thresholds should be applied.                          ║
║                                                              ║
║   Implementation priority (highest accuracy ROI first):      ║
║     1. RSI Filter       — kills exhausted / chop entries     ║
║     2. ATR Filter       — kills dead markets & news spikes   ║
║     3. Momentum Score   — confirms directional pressure      ║
║     4. Session Quality  — weights by liquidity window        ║
║     5. Cooldown Tracker — prevents chasing same direction    ║
╚══════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import statistics
import time


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATED FILTER RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Single object that ConfluenceEngine reads after running all filters.

    suppress         → True = kill the signal entirely, do not send
    confidence_delta → Float added to raw confidence (positive = boost,
                       negative = penalty). Applied BEFORE threshold gate.
    factors          → Human-readable strings that are appended to the
                       signal's confluence_factors list in the Telegram msg.
    rsi_value        → Last computed RSI (for logging / debugging)
    atr_ratio        → Current ATR vs baseline ratio (for logging)
    session_label    → Active session name
    momentum_score   → Directional agreement ratio from recent candles
    """
    suppress:          bool  = False
    confidence_delta:  float = 0.0
    factors:           list  = field(default_factory=list)
    rsi_value:         float = 50.0
    atr_ratio:         float = 1.0
    session_label:     str   = "UNKNOWN"
    momentum_score:    float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# FILTER 1 — RSI  (Highest ROI — eliminates the most false signals)
# ═════════════════════════════════════════════════════════════════════════════

def compute_rsi(closes: list, period: int = 14) -> float:
    """
    Wilder Smoothed RSI using a list of closing prices.
    Returns 50.0 (neutral) when there is insufficient data — no penalty,
    no boost, just transparent passthrough.

    Why period=14: Standard across all timeframes. On M1 = last 14 minutes,
    which is sufficient to detect overbought/oversold for 1-min binary entries.
    """
    if len(closes) < period + 1:
        return 50.0  # Neutral fallback — not enough history yet

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    # Seed with simple average for the first period
    avg_gain = statistics.mean(gains[:period])
    avg_loss = statistics.mean(losses[:period])

    # Wilder smoothing for remaining values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def apply_rsi_filter(
    rsi:       float,
    direction: str,
    is_otc:    bool = False,
) -> tuple:
    """
    Score RSI against the signal direction.

    OTC pairs get ±4 wider zones to absorb broker-side price offsets
    that can shift RSI readings by a few points vs real market.

    Returns: (confidence_delta: float, factors: list[str], suppress: bool)

    Suppression rules (hard kills):
      CALL with RSI > 82 → market is severely overbought, reversal imminent
      PUT  with RSI < 18 → market is severely oversold, bounce imminent
    """
    delta    = 0.0
    factors  = []
    suppress = False

    # OTC broker price offsets shift RSI by ~3-4 points — widen zones
    w = 4 if is_otc else 0

    if direction == "CALL":
        if rsi <= 30 + w:
            delta = +0.13
            factors.append(f"🟢 RSI {rsi:.1f} — Oversold reversal zone → Strong CALL confirmation")
        elif rsi <= 44 + w:
            delta = +0.07
            factors.append(f"🟢 RSI {rsi:.1f} — Recovering from oversold → CALL momentum building")
        elif rsi <= 56 + w:
            delta = +0.02
            factors.append(f"✅ RSI {rsi:.1f} — Neutral zone (mild CALL support)")
        elif rsi <= 69 + w:
            delta = -0.05
            factors.append(f"⚠️ RSI {rsi:.1f} — Approaching overbought → CALL conviction reduced")
        else:
            # Chasing a rally that is already extended
            delta    = -0.12
            suppress = rsi > 82 + w
            tag      = "🔴 SUPPRESSED — " if suppress else "❌ "
            factors.append(
                f"{tag}RSI {rsi:.1f} — Overbought on CALL entry "
                f"({'KILLED: extreme exhaustion' if suppress else 'very high reversal risk'})"
            )

    else:  # PUT
        if rsi >= 70 - w:
            delta = +0.13
            factors.append(f"🔴 RSI {rsi:.1f} — Overbought reversal zone → Strong PUT confirmation")
        elif rsi >= 56 - w:
            delta = +0.07
            factors.append(f"🔴 RSI {rsi:.1f} — Recovering from overbought → PUT momentum building")
        elif rsi >= 44 - w:
            delta = +0.02
            factors.append(f"✅ RSI {rsi:.1f} — Neutral zone (mild PUT support)")
        elif rsi >= 31 - w:
            delta = -0.05
            factors.append(f"⚠️ RSI {rsi:.1f} — Approaching oversold → PUT conviction reduced")
        else:
            # Chasing a drop that is already extended
            delta    = -0.12
            suppress = rsi < 18 - w
            tag      = "🔴 SUPPRESSED — " if suppress else "❌ "
            factors.append(
                f"{tag}RSI {rsi:.1f} — Oversold on PUT entry "
                f"({'KILLED: extreme exhaustion' if suppress else 'very high bounce risk'})"
            )

    return delta, factors, suppress


# ═════════════════════════════════════════════════════════════════════════════
# FILTER 2 — ATR VOLATILITY  (Second highest ROI)
# ═════════════════════════════════════════════════════════════════════════════

def compute_atr(candles: list, period: int = 14) -> float:
    """
    Wilder Average True Range from a list of candle dicts.
    candles must be: [{"open":..., "high":..., "low":..., "close":...}, ...]
    Returns 0.0 when insufficient data.
    """
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]
        tr   = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"]  - prev["close"]),
        )
        true_ranges.append(tr)

    # Seed with simple mean, then Wilder-smooth
    atr = statistics.mean(true_ranges[:period])
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    return atr


def apply_atr_filter(
    candles: list,
    atr:     float,
    is_otc:  bool = False,
) -> tuple:
    """
    Compare current ATR to a rolling baseline of all candle True Ranges.

    Dead market  → ATR well below baseline → 1-min candle won't move enough to profit
    News spike   → ATR far above baseline  → chaotic, unpredictable movement
    Sweet spot   → ATR near baseline       → clean, readable price action

    OTC uses looser thresholds (±15%) because OTC broker pricing can
    widen spreads and inflate TR without real underlying volatility.

    Returns: (confidence_delta, factors, suppress)
    """
    delta    = 0.0
    factors  = []
    suppress = False

    if atr == 0.0 or len(candles) < 10:
        factors.append("⚠️ ATR unavailable — volatility filter skipped")
        return 0.0, factors, False

    # Build baseline: mean of ALL true ranges in the dataset
    all_trs = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i]["high"] - candles[i]["low"],
            abs(candles[i]["high"] - candles[i - 1]["close"]),
            abs(candles[i]["low"]  - candles[i - 1]["close"]),
        )
        all_trs.append(tr)

    if not all_trs:
        return 0.0, ["⚠️ ATR baseline empty — skipping"], False

    baseline = statistics.mean(all_trs)
    if baseline == 0:
        return 0.0, ["⚠️ Zero ATR baseline — skipping"], False

    ratio = round(atr / baseline, 3)

    # OTC gets ±15% tolerance on both thresholds
    dead_floor   = 0.45 if is_otc else 0.55   # Below this = dead market
    dead_kill    = 0.30 if is_otc else 0.35   # Below this = hard suppress
    spike_ceil   = 3.40 if is_otc else 2.90   # Above this = spike
    spike_kill   = 5.00 if is_otc else 4.00   # Above this = hard suppress

    if ratio < dead_kill:
        suppress = True
        delta    = -0.15
        factors.append(
            f"🔴 SUPPRESSED — ATR {ratio:.2f}× baseline: Market DEAD "
            f"(flat/no movement — 1-min entry has no profit potential)"
        )
    elif ratio < dead_floor:
        delta = -0.09
        factors.append(
            f"❌ ATR {ratio:.2f}× baseline — Very low volatility "
            f"(candle likely won't move enough for clean profit)"
        )
    elif ratio > spike_kill:
        suppress = True
        delta    = -0.15
        factors.append(
            f"🔴 SUPPRESSED — ATR {ratio:.2f}× baseline: NEWS SPIKE "
            f"(erratic movement — pattern analysis unreliable)"
        )
    elif ratio > spike_ceil:
        delta = -0.09
        factors.append(
            f"❌ ATR {ratio:.2f}× baseline — High volatility spike "
            f"(possible news event — use caution)"
        )
    elif 0.75 <= ratio <= 1.90:
        # Sweet spot — trending but controlled
        delta = +0.05
        factors.append(
            f"✅ ATR {ratio:.2f}× baseline — Volatility in sweet spot "
            f"(clean movement expected)"
        )
    else:
        # Acceptable zone — slightly below/above sweet spot but not dangerous
        delta = +0.01
        factors.append(f"✅ ATR {ratio:.2f}× baseline — Acceptable volatility")

    return delta, factors, suppress


# ═════════════════════════════════════════════════════════════════════════════
# FILTER 3 — MOMENTUM CONSISTENCY SCORE
# ═════════════════════════════════════════════════════════════════════════════

def apply_momentum_filter(
    candles:   list,
    direction: str,
    lookback:  int = 5,
) -> tuple:
    """
    Measures directional pressure by counting bullish vs bearish candles
    over the last N closed candles.

    This is NOT the same as Three White Soldiers (which requires very specific
    body sizes). This is a general pressure gauge — it rewards entries where
    recent candles clearly lean in the signal direction.

    Works identically for OTC — pure price action, broker-agnostic.

    Returns: (confidence_delta, factors)
    """
    if len(candles) < lookback:
        return 0.0, ["⚠️ Not enough candles for momentum filter"]

    recent = candles[-lookback:]

    bull = sum(1 for c in recent if c["close"] > c["open"])
    bear = sum(1 for c in recent if c["close"] < c["open"])
    doji = lookback - bull - bear  # Neutral candles

    # Agreement = how many of the last N candles support signal direction
    agreement = bull if direction == "CALL" else bear
    agreement_pct = agreement / lookback

    delta   = 0.0
    factors = []

    if agreement_pct >= 0.80:
        delta = +0.08
        factors.append(
            f"✅ Momentum: {agreement}/{lookback} recent candles align with {direction} "
            f"— strong directional pressure"
        )
    elif agreement_pct >= 0.60:
        delta = +0.04
        factors.append(
            f"✅ Momentum: {agreement}/{lookback} candles lean {direction} "
            f"— moderate directional pressure"
        )
    elif agreement_pct == 0.50:
        delta = -0.02
        factors.append(
            f"⚠️ Momentum: 50/50 split ({bull} bull / {bear} bear) — choppy market"
        )
    else:
        # Majority of recent candles oppose the signal direction
        opposing = bear if direction == "CALL" else bull
        delta    = -0.07
        factors.append(
            f"❌ Counter-momentum: {opposing}/{lookback} candles oppose {direction} "
            f"— high-risk counter-trend entry"
        )

    return delta, factors


# ═════════════════════════════════════════════════════════════════════════════
# FILTER 4 — SESSION QUALITY
# ═════════════════════════════════════════════════════════════════════════════

def get_session_quality(asset_type: str, is_otc: bool = False) -> tuple:
    """
    Returns session quality based on UTC time and asset type.
    (session_label, quality, confidence_delta, factors)

    OTC forex still follows the underlying pair's session structure —
    the broker runs 24/7 but liquidity quality mirrors real-market hours.
    OTC gets a lighter penalty in dead zones since spreads are pre-fixed by
    the broker, reducing (but not eliminating) the liquidity risk.

    Quality levels:
      PRIME    → Best signal quality, full boost
      ACTIVE   → Good, small boost
      MODERATE → Acceptable, small penalty
      DEAD     → Very low quality, large penalty + hard suppress for non-OTC
    """
    utc_hour = datetime.now(timezone.utc).hour
    factors  = []

    # ── CRYPTO: Truly 24/7 — session doesn't affect signal quality ───────────
    if asset_type == "crypto":
        return (
            "24/7 Crypto",
            "ACTIVE",
            +0.02,
            ["✅ Crypto asset — 24/7 liquidity, session not a constraint"]
        )

    # ── STOCKS / INDICES: Highly sensitive to exchange hours ─────────────────
    if asset_type in ("stock", "index"):
        if 13 <= utc_hour < 21:
            return ("US Market Hours", "PRIME",    +0.07,
                    ["✅ US market hours (13:00–21:00 UTC) — full institutional liquidity"])
        elif 7 <= utc_hour < 15:
            return ("EU Market Hours", "ACTIVE",   +0.04,
                    ["✅ EU market hours (07:00–15:00 UTC) — good liquidity"])
        else:
            suppress_note = "" if is_otc else " → Signal suppressed for real assets"
            return ("After Hours", "DEAD", -0.14,
                    [f"🔴 After-hours/pre-market{suppress_note} — very thin stock/index liquidity"])

    # ── COMMODITIES (XAU, XAG, Oil): Follow forex sessions loosely ───────────
    if asset_type == "commodity":
        if 12 <= utc_hour < 16:
            return ("London–NY Overlap", "PRIME",    +0.07,
                    ["✅ Commodity prime hours — London + NY overlap (peak Gold/Oil volume)"])
        elif 7 <= utc_hour < 21:
            return ("Active Session",    "ACTIVE",   +0.02,
                    ["✅ Commodity active hours — adequate liquidity"])
        else:
            return ("Off-Hours",         "MODERATE", -0.04,
                    ["⚠️ Commodity off-hours — thinner liquidity, wider spreads"])

    # ── FOREX (and OTC Forex): Full session matrix ────────────────────────────
    # London–NY overlap is the undisputed best window for forex
    if 12 <= utc_hour < 16:
        return ("London–NY Overlap",   "PRIME",    +0.09,
                ["✅ SESSION: London–NY Overlap (12:00–16:00 UTC) — PEAK LIQUIDITY 🔥"])
    elif 7 <= utc_hour < 12:
        return ("London Open",         "ACTIVE",   +0.05,
                ["✅ SESSION: London Open (07:00–12:00 UTC) — High liquidity"])
    elif 16 <= utc_hour < 21:
        return ("NY Session",          "ACTIVE",   +0.04,
                ["✅ SESSION: NY Session (16:00–21:00 UTC) — Good liquidity"])
    elif 0 <= utc_hour < 7:
        return ("Asian Session",       "MODERATE", -0.03,
                ["⚠️ SESSION: Asian Session (00:00–07:00 UTC) — Lower forex liquidity"])
    else:
        # 21:00–24:00 UTC — pre-Asian dead zone
        # OTC gets lighter penalty (broker absorbs spread risk)
        delta = -0.06 if is_otc else -0.10
        return ("Pre-Asian Dead Zone", "DEAD", delta,
                [f"{'⚠️' if is_otc else '🔴'} SESSION: Dead Zone (21:00–00:00 UTC) — "
                 f"{'Reduced liquidity (OTC mitigated)' if is_otc else 'Very low liquidity — skip if possible'}"])


# ═════════════════════════════════════════════════════════════════════════════
# FILTER 5 — SIGNAL COOLDOWN TRACKER
# ═════════════════════════════════════════════════════════════════════════════

class SignalCooldownTracker:
    """
    Stateful tracker that prevents the engine from chasing consecutive signals
    on the same asset in the same direction.

    On M1 binary options, firing the same direction twice in a row on the same
    pair almost always means you are chasing a move that has already happened.
    The second signal is almost always lower quality than the first.

    OTC pairs are tracked separately from their real equivalents:
    "EUR/USD-OTC" and "EUR/USD" maintain independent cooldown states
    because they can diverge in price and timing.

    Usage:
        # Add to ConfluenceEngine.__init__:
        self.cooldown = SignalCooldownTracker()

        # Inside evaluate(), BEFORE returning FinalSignal:
        cd_delta, cd_factors = self.cooldown.check(asset, direction)

        # Inside evaluate(), AFTER deciding to return a signal:
        self.cooldown.record(asset, direction)
    """

    HARD_COOLDOWN_CANDLES    = 2    # 0–2 candles: heavy penalty
    PARTIAL_COOLDOWN_CANDLES = 5    # 2–5 candles: mild penalty
    CANDLE_SECONDS           = 60   # M1 = 60 seconds per candle

    def __init__(self):
        # {asset_key: {"direction": str, "ts": float}}
        self._history: dict = {}

    def check(
        self,
        asset:     str,
        direction: str,
        now_ts:    Optional[float] = None,
    ) -> tuple:
        """
        Check if this asset/direction combination is in cooldown.
        Returns (confidence_delta, factors).

        Does NOT record — call .record() separately after the signal fires.
        This separation is important: evaluate() checks cooldown, then decides
        whether to return a signal, then records only on actual send.
        """
        if now_ts is None:
            now_ts = time.time()

        record = self._history.get(asset)
        if not record:
            return 0.0, []

        same_direction  = record["direction"] == direction
        elapsed_seconds = now_ts - record["ts"]
        elapsed_candles = elapsed_seconds / self.CANDLE_SECONDS

        if not same_direction:
            # Opposite direction flip — this is a valid reversal, no penalty
            # (CRT reversal after a sweep, or a new pattern in opposite direction)
            return 0.0, []

        if elapsed_candles <= self.HARD_COOLDOWN_CANDLES:
            return -0.18, [
                f"⚠️ COOLDOWN: Same direction ({direction}) fired "
                f"{elapsed_candles:.1f} candles ago — chasing penalty −18%"
            ]
        elif elapsed_candles <= self.PARTIAL_COOLDOWN_CANDLES:
            return -0.07, [
                f"⚠️ Recent signal: {direction} was {elapsed_candles:.1f} candles ago "
                f"— mild penalty −7%"
            ]

        # Beyond cooldown window — fresh entry
        return 0.0, []

    def record(self, asset: str, direction: str, now_ts: Optional[float] = None):
        """
        Record that a signal fired for this asset.
        Call this ONLY when a signal is actually going to be sent to the user.
        """
        self._history[asset] = {
            "direction": direction,
            "ts":        now_ts or time.time(),
        }

    def reset(self, asset: str):
        """Clear cooldown for one asset (useful on /stop → /resume)."""
        self._history.pop(asset, None)

    def reset_all(self):
        """Clear all cooldowns (useful on full bot restart)."""
        self._history.clear()


# ═════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER — call this once inside ConfluenceEngine.evaluate()
# ═════════════════════════════════════════════════════════════════════════════

def run_all_filters(
    asset:      str,
    asset_type: str,                      # "forex"|"crypto"|"commodity"|"stock"|"index"
    direction:  str,                      # "CALL" | "PUT"
    candle_dicts: list,                   # list of plain dicts (open/high/low/close/volume)
    closes:     list,                     # list of float close prices (for RSI)
    cooldown:   SignalCooldownTracker,
    is_otc:     bool = False,
    now_ts:     Optional[float] = None,
) -> FilterResult:
    """
    Run all five filters in priority order and return an aggregated FilterResult.

    The ConfluenceEngine passes candle_dicts (already converted via _candles_to_dicts)
    and the closes list (extracted from Candle objects).

    Priority order matches ROI:
      1. RSI          — highest false-positive reduction
      2. ATR          — second highest (dead market / news spike guard)
      3. Momentum     — directional pressure confirmation
      4. Session      — market quality weighting
      5. Cooldown     — chasing prevention
    """
    result = FilterResult()

    # ── 1. RSI ───────────────────────────────────────────────────────────────
    rsi = compute_rsi(closes)
    result.rsi_value = rsi
    rsi_delta, rsi_factors, rsi_suppress = apply_rsi_filter(rsi, direction, is_otc)
    result.confidence_delta += rsi_delta
    result.factors.extend(rsi_factors)
    if rsi_suppress:
        result.suppress = True

    # ── 2. ATR ───────────────────────────────────────────────────────────────
    atr = compute_atr(candle_dicts)
    result.atr_ratio = atr
    atr_delta, atr_factors, atr_suppress = apply_atr_filter(candle_dicts, atr, is_otc)
    result.confidence_delta += atr_delta
    result.factors.extend(atr_factors)
    if atr_suppress:
        result.suppress = True

    # ── 3. Momentum ──────────────────────────────────────────────────────────
    mom_delta, mom_factors = apply_momentum_filter(candle_dicts, direction)
    result.momentum_score   = mom_delta
    result.confidence_delta += mom_delta
    result.factors.extend(mom_factors)

    # ── 4. Session ───────────────────────────────────────────────────────────
    s_label, s_quality, s_delta, s_factors = get_session_quality(asset_type, is_otc)
    result.session_label     = s_label
    result.confidence_delta += s_delta
    result.factors.extend(s_factors)

    # Hard suppress dead session for non-OTC non-crypto
    if s_quality == "DEAD" and not is_otc and asset_type not in ("crypto",):
        result.suppress = True
        result.factors.append(
            f"🔴 SUPPRESSED: Dead session for {asset_type} — signal killed "
            f"(insufficient liquidity to trade reliably)"
        )

    # ── 5. Cooldown ──────────────────────────────────────────────────────────
    cd_delta, cd_factors = cooldown.check(asset, direction, now_ts)
    result.confidence_delta += cd_delta
    result.factors.extend(cd_factors)

    return result
