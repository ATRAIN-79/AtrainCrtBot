"""
╔══════════════════════════════════════════════════════════════╗
║         CRT + VOLUME ANALYSIS ENGINE                         ║
║   Candle Range Theory × Volume Confluence                    ║
║   Designed for M1 Binary Options — OTC Compatible            ║
╚══════════════════════════════════════════════════════════════╝

Strategy Logic:
  Phase 1 — RANGE DEFINITION
    • Use the previous CLOSED candle's High/Low as the CRT range
    • Optionally extend to the last 3-candle swing high/low for robustness

  Phase 2 — MANIPULATION DETECTION
    • A sweep is confirmed when price wicks beyond the range boundary
    • Volume on the sweep candle must spike above the rolling average
    • For OTC: tolerance buffer applied (price offset is broker-side)

  Phase 3 — REVERSAL CONFIRMATION
    • Rejection: candle closes BACK inside the range after sweep
    • Volume must CONTRACT after the spike (exhaustion)
    • Forming candle (live) should show opposite momentum

  Phase 4 — CONFLUENCE SCORING
    • Each confirmed element adds points to a 0-100 confidence score
    • EMA trend alignment is used as a bonus multiplier
    • Pre-close timing (40-55s into candle) earns extra weight
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CRTZone:
    """Defines a Candle Range Theory zone."""
    range_high:     float          # Top of the CRT range
    range_low:      float          # Bottom of the CRT range
    range_size:     float          # range_high - range_low (in pips)
    source_candles: int            # How many candles were used to define the range


@dataclass
class VolumeProfile:
    """Volume statistics derived from recent candle history."""
    average_volume:     float      # Rolling average (N candles)
    current_volume:     float      # Volume on the most recent closed candle
    sweep_volume:       float      # Volume on the candle that swept the range
    volume_ratio:       float      # sweep_volume / average_volume
    is_spike:           bool       # True if volume_ratio > SPIKE_THRESHOLD
    is_contracting:     bool       # True if volume is declining after spike
    divergence:         bool       # Price extreme + falling volume = divergence


@dataclass
class CRTSignal:
    """Output of a full CRT + Volume analysis pass."""
    direction:          str        # "CALL" or "PUT"
    confidence:         int        # 0–100
    sweep_side:         str        # "HIGH_SWEPT" or "LOW_SWEPT"
    zone:               CRTZone
    volume:             VolumeProfile
    confluence_factors: list[str]  # Human-readable reasons
    is_pre_close:       bool       # True if fired in the 40–55s window
    rejection_confirmed: bool      # Price closed back inside range
    ema_aligned:        bool       # Trend supports the direction
    raw_score:          int        # Pre-normalised score (debugging)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Tuned for M1 Binary / OTC
# ─────────────────────────────────────────────────────────────────────────────

VOLUME_SPIKE_THRESHOLD  = 1.5    # sweep candle volume must be 1.5× the average
VOLUME_LOOKBACK         = 14     # candles used for average volume calculation
VOLUME_CONTRACTION_MIN  = 0.85   # post-spike volume must be ≤ 85% of spike vol

SWEEP_MIN_PIPS          = 0.5    # minimum wick beyond range (pips)
OTC_SWEEP_TOLERANCE     = 0.3    # extra pip tolerance for OTC broker offsets

RANGE_LOOKBACK_CANDLES  = 3      # candles used to build the CRT range
RANGE_OFFSET            = 3      # how many candles back the range window starts
                                 # Zone = candles[-(RANGE_OFFSET+LOOKBACK):-RANGE_OFFSET]
                                 # Sweep = candles[-RANGE_OFFSET:]  (most recent)
EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 50

PRE_CLOSE_WINDOW_START  = 40     # seconds into candle: pre-close phase begins
PRE_CLOSE_WINDOW_END    = 55     # seconds into candle: pre-close phase ends

MIN_CONFIDENCE          = 62     # gate — signals below this are suppressed


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> list[float]:
    """Compute EMA for a list of closing prices."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema_vals = [sum(values[:period]) / period]
    for v in values[period:]:
        ema_vals.append(v * k + ema_vals[-1] * (1 - k))
    return ema_vals


def _pip_size(symbol: str) -> float:
    """Returns pip size for a given symbol (JPY pairs differ)."""
    symbol_upper = symbol.upper()
    if "JPY" in symbol_upper:
        return 0.01
    if any(c in symbol_upper for c in ["XAU", "GOLD"]):
        return 0.1
    if "BTC" in symbol_upper or "ETH" in symbol_upper:
        return 1.0
    return 0.0001


def _to_pips(price_diff: float, symbol: str) -> float:
    """Convert a price difference to pips."""
    pip = _pip_size(symbol)
    if pip == 0:
        return 0.0
    return abs(price_diff) / pip


# ─────────────────────────────────────────────────────────────────────────────
# CRT RANGE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_crt_zone(
    candles:  list,
    lookback: int = RANGE_LOOKBACK_CANDLES,
    offset:   int = RANGE_OFFSET,
) -> CRTZone:
    """
    Build the CRT range from an OLDER candle window, separated from
    the sweep detection window.

    Window selection:
      Zone  = candles[-(offset + lookback) : -offset]   ← older structure
      Sweep = candles[-offset:]                          ← recent candles

    This separation is critical: the sweep candle must NOT be inside
    the zone window, otherwise the zone high/low absorbs the wick
    and no sweep can ever be detected.

    candles: list of dicts with keys: open, high, low, close, volume
             ordered oldest → newest, last item = most recent CLOSED candle
    """
    end_idx   = -offset if offset > 0 else len(candles)
    start_idx = end_idx - lookback

    window = candles[start_idx:end_idx] if start_idx < end_idx else candles[:end_idx]
    if not window:
        window = candles[:max(1, len(candles) // 2)]

    range_high = max(c["high"] for c in window)
    range_low  = min(c["low"]  for c in window)
    range_size = range_high - range_low

    return CRTZone(
        range_high=range_high,
        range_low=range_low,
        range_size=range_size,
        source_candles=len(window)
    )


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

def analyse_volume(
    candles:       list,
    sweep_index:   int,            # index of the candle that swept the range
    lookback:      int = VOLUME_LOOKBACK,
    spike_thresh:  float = VOLUME_SPIKE_THRESHOLD,
) -> VolumeProfile:
    """
    Compute the volume profile around a CRT sweep event.

    sweep_index: position in the candles list of the sweep candle
                 (use -2 for the most recent closed, -1 for forming).
    """
    # Guard: need at least lookback + 1 candles
    if len(candles) < lookback + 1:
        lookback = max(2, len(candles) - 1)

    # Baseline average (candles BEFORE the sweep)
    baseline_candles = candles[max(0, sweep_index - lookback): sweep_index]
    baseline_vols    = [c["volume"] for c in baseline_candles if c["volume"] > 0]
    avg_vol          = statistics.mean(baseline_vols) if baseline_vols else 1.0

    sweep_candle  = candles[sweep_index]
    sweep_vol     = sweep_candle["volume"]
    vol_ratio     = sweep_vol / avg_vol if avg_vol > 0 else 1.0
    is_spike      = vol_ratio >= spike_thresh

    # Post-sweep contraction check (candle immediately after sweep)
    post_index = sweep_index + 1
    is_contracting = False
    if post_index < len(candles):
        post_vol       = candles[post_index]["volume"]
        is_contracting = (post_vol <= sweep_vol * VOLUME_CONTRACTION_MIN)

    # Divergence: last 3 candle closes trending up but volume trending down
    recent = candles[-4:] if len(candles) >= 4 else candles
    prices = [c["close"] for c in recent]
    vols   = [c["volume"] for c in recent]
    price_rising = prices[-1] > prices[0]
    vol_falling  = vols[-1] < vols[0]
    price_falling = prices[-1] < prices[0]
    vol_rising    = vols[-1] > vols[0]
    divergence    = (price_rising and vol_falling) or (price_falling and vol_rising)

    return VolumeProfile(
        average_volume=avg_vol,
        current_volume=candles[-1]["volume"],
        sweep_volume=sweep_vol,
        volume_ratio=vol_ratio,
        is_spike=is_spike,
        is_contracting=is_contracting,
        divergence=divergence
    )


# ─────────────────────────────────────────────────────────────────────────────
# CRT SWEEP DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_sweep(
    candles:        list,
    zone:           CRTZone,
    symbol:         str,
    is_otc:         bool = False,
    sweep_window:   int  = RANGE_OFFSET,
) -> tuple[Optional[str], int]:
    """
    Scan the most recent `sweep_window` closed candles for a CRT manipulation sweep.
    These candles are SEPARATE from the zone-defining candles.

    Returns:
        (sweep_side, sweep_index) where sweep_side is:
        "HIGH_SWEPT"  → price broke above range_high then rejected
        "LOW_SWEPT"   → price broke below range_low then rejected
        None          → no sweep found
    """
    tolerance_pips = OTC_SWEEP_TOLERANCE if is_otc else 0.0
    tolerance      = tolerance_pips * _pip_size(symbol)
    min_wick_price = SWEEP_MIN_PIPS * _pip_size(symbol)

    # Check only the most recent `sweep_window` closed candles
    check_candles = candles[-sweep_window:]

    for offset, candle in enumerate(reversed(check_candles)):
        idx = len(candles) - 1 - offset  # actual index in candles list

        high_wick = candle["high"] - zone.range_high
        low_wick  = zone.range_low - candle["low"]

        # HIGH SWEEP: wick above range_high, candle closes back INSIDE range
        if (high_wick >= (min_wick_price - tolerance) and
                candle["close"] <= zone.range_high + tolerance):
            return "HIGH_SWEPT", idx

        # LOW SWEEP: wick below range_low, candle closes back INSIDE range
        if (low_wick >= (min_wick_price - tolerance) and
                candle["close"] >= zone.range_low - tolerance):
            return "LOW_SWEPT", idx

    return None, -1


# ─────────────────────────────────────────────────────────────────────────────
# FORMING CANDLE MOMENTUM CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_forming_momentum(
    forming:    Optional[dict],
    direction:  str,
    elapsed:    int,
) -> tuple[bool, str]:
    """
    Validate that the live forming candle supports the signal direction.

    Returns (confirmed: bool, note: str)
    """
    if not forming:
        return False, "No forming candle data"

    body = forming["close"] - forming["open"]

    if direction == "CALL":
        if body > 0:
            strength = "strong" if abs(body) > abs(forming["open"] * 0.0002) else "mild"
            return True, f"Forming candle bullish ({strength} body, {elapsed}s elapsed)"
        else:
            return False, "Forming candle bearish — momentum mismatch"

    else:  # PUT
        if body < 0:
            strength = "strong" if abs(body) > abs(forming["open"] * 0.0002) else "mild"
            return True, f"Forming candle bearish ({strength} body, {elapsed}s elapsed)"
        else:
            return False, "Forming candle bullish — momentum mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# EMA TREND FILTER
# ─────────────────────────────────────────────────────────────────────────────

def check_ema_alignment(candles: list, direction: str) -> tuple[bool, str]:
    """
    Check if EMA 9/21/50 trend supports the CRT signal direction.
    Returns (aligned: bool, description: str)
    """
    closes = [c["close"] for c in candles]
    if len(closes) < EMA_SLOW + 5:
        return False, "Insufficient data for EMA"

    ema9  = _ema(closes, EMA_FAST)
    ema21 = _ema(closes, EMA_MID)
    ema50 = _ema(closes, EMA_SLOW)

    if not (ema9 and ema21 and ema50):
        return False, "EMA calculation failed"

    e9, e21, e50 = ema9[-1], ema21[-1], ema50[-1]

    if direction == "CALL":
        # Bullish stack: EMA9 > EMA21 > EMA50
        if e9 > e21 > e50:
            return True, f"EMA9 > EMA21 > EMA50 (bullish stack ✅)"
        elif e9 > e21:
            return True, f"EMA9 > EMA21 (partial bullish alignment)"
        else:
            return False, f"EMA bearish — counter-trend CALL"

    else:  # PUT
        # Bearish stack: EMA9 < EMA21 < EMA50
        if e9 < e21 < e50:
            return True, f"EMA9 < EMA21 < EMA50 (bearish stack ✅)"
        elif e9 < e21:
            return True, f"EMA9 < EMA21 (partial bearish alignment)"
        else:
            return False, f"EMA bullish — counter-trend PUT"


# ─────────────────────────────────────────────────────────────────────────────
# REJECTION CONFIRMATION
# ─────────────────────────────────────────────────────────────────────────────

def confirm_rejection(
    candles:    list,
    sweep_idx:  int,
    sweep_side: str,
    zone:       CRTZone,
    symbol:     str,
    is_otc:     bool = False,
) -> tuple[bool, str]:
    """
    After a sweep, check that the NEXT candle(s) confirm rejection.
    The close should be meaningfully inside the range.
    """
    tolerance = (OTC_SWEEP_TOLERANCE if is_otc else 0.0) * _pip_size(symbol)

    post_candles = candles[sweep_idx + 1:]
    if not post_candles:
        return False, "No post-sweep candles to confirm rejection"

    confirm_candle = post_candles[0]

    if sweep_side == "HIGH_SWEPT":
        # Price should close below range_high (back inside or moving down)
        if confirm_candle["close"] < zone.range_high - tolerance:
            pip_dist = _to_pips(zone.range_high - confirm_candle["close"], symbol)
            return True, f"Confirmed rejection below range high ({pip_dist:.1f} pips inside)"
        else:
            return False, "Price still above range high — no rejection yet"

    else:  # LOW_SWEPT
        # Price should close above range_low (back inside or moving up)
        if confirm_candle["close"] > zone.range_low + tolerance:
            pip_dist = _to_pips(confirm_candle["close"] - zone.range_low, symbol)
            return True, f"Confirmed rejection above range low ({pip_dist:.1f} pips inside)"
        else:
            return False, "Price still below range low — no rejection yet"


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    sweep_detected:         bool,
    volume_spike:           bool,
    volume_ratio:           float,
    volume_contracting:     bool,
    volume_divergence:      bool,
    rejection_confirmed:    bool,
    ema_aligned:            bool,
    forming_confirmed:      bool,
    is_pre_close:           bool,
) -> tuple[int, list[str]]:
    """
    Score the CRT+Volume setup from 0–100.

    Point breakdown:
      Core CRT
        Sweep detected             +25 (mandatory)
        Rejection confirmed        +20
      Volume
        Volume spike on sweep      +20
        Volume ratio boost          +5 (if ratio > 2.0×)
        Volume contracting         +10
        Volume divergence           +8
      Filters
        EMA aligned                +10
        Forming candle confirms     +8
      Timing
        Pre-close window            +5
    ─────────────────────────────────
    Max theoretical              ~111 → clamped to 100
    """
    score   = 0
    factors = []

    if not sweep_detected:
        return 0, ["❌ No CRT sweep detected"]

    score += 25
    factors.append("✅ CRT sweep detected (range manipulation confirmed)")

    if rejection_confirmed:
        score += 20
        factors.append("✅ Price rejected back inside range (reversal confirmed)")
    else:
        factors.append("⚠️ Rejection not yet confirmed — early entry")

    if volume_spike:
        score += 20
        factors.append(f"✅ Volume spike on sweep ({volume_ratio:.1f}× average — liquidity absorbed)")
        if volume_ratio >= 2.0:
            score += 5
            factors.append(f"🔥 Extreme volume spike ({volume_ratio:.1f}×) — high conviction sweep")
    else:
        factors.append(f"⚠️ Volume below spike threshold ({volume_ratio:.1f}× — weak sweep signal)")

    if volume_contracting:
        score += 10
        factors.append("✅ Volume contracting post-sweep (exhaustion confirmed)")
    else:
        factors.append("⚠️ Volume not contracting yet")

    if volume_divergence:
        score += 8
        factors.append("✅ Volume divergence detected (price/volume disagreement)")

    if ema_aligned:
        score += 10
        factors.append("✅ EMA trend aligned with signal direction")
    else:
        factors.append("⚠️ EMA counter-trend — higher risk entry")

    if forming_confirmed:
        score += 8
        factors.append("✅ Forming candle momentum confirms direction")
    else:
        factors.append("⚠️ Forming candle not yet confirming")

    if is_pre_close:
        score += 5
        factors.append("⚡ Pre-close alert window (40–55s) — next candle entry")

    # Clamp to 100
    score = min(score, 100)
    return score, factors


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CRT + VOLUME ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CRTVolumeEngine:
    """
    Main analysis class. Accepts the same candle/forming data
    format used by your existing ConfluenceEngine.

    Usage:
        engine = CRTVolumeEngine()
        result = engine.analyse(
            asset          = "EUR/USD-OTC",
            candles        = candles,        # list of dicts: open/high/low/close/volume
            forming_candle = forming,        # live candle dict (or None)
            elapsed_seconds= elapsed,        # int: seconds into current minute
        )
        if result and result.confidence >= 62:
            # Fire signal
    """

    def __init__(self, min_confidence: int = MIN_CONFIDENCE):
        self.min_confidence = min_confidence

    def analyse(
        self,
        asset:           str,
        candles:         list,
        forming_candle:  Optional[dict],
        elapsed_seconds: int,
    ) -> Optional[CRTSignal]:
        """
        Run the full CRT + Volume pipeline.

        Returns a CRTSignal if a valid setup is found above min_confidence,
        otherwise returns None.
        """
        if len(candles) < VOLUME_LOOKBACK + RANGE_LOOKBACK_CANDLES + 2:
            return None

        is_otc      = asset.endswith("-OTC")
        is_pre_close= PRE_CLOSE_WINDOW_START <= elapsed_seconds <= PRE_CLOSE_WINDOW_END

        # ── Phase 1: Build CRT Range (from OLDER candles) ────────────────
        # Zone window  = candles[-(RANGE_OFFSET+RANGE_LOOKBACK) : -RANGE_OFFSET]
        # Sweep window = candles[-RANGE_OFFSET:]
        # Keeping these separate is critical: the zone must form BEFORE
        # the sweep candle, so the sweep wick actually exceeds the zone boundary.
        zone = build_crt_zone(candles, lookback=RANGE_LOOKBACK_CANDLES, offset=RANGE_OFFSET)

        # ── Phase 2: Detect Sweep (in most recent RANGE_OFFSET candles) ───
        sweep_side, sweep_idx = detect_sweep(candles, zone, asset, is_otc)
        if sweep_side is None:
            return None  # No sweep → no CRT setup

        # Signal direction is OPPOSITE to the sweep (reversal trade)
        direction = "CALL" if sweep_side == "LOW_SWEPT" else "PUT"

        # ── Phase 3: Volume Analysis ──────────────────────────────────────
        vol = analyse_volume(candles, sweep_idx)

        # ── Phase 4: Rejection Confirmation ──────────────────────────────
        rejection_confirmed, rejection_note = confirm_rejection(
            candles, sweep_idx, sweep_side, zone, asset, is_otc
        )

        # ── Phase 5: EMA Filter ───────────────────────────────────────────
        ema_aligned, ema_note = check_ema_alignment(candles, direction)

        # ── Phase 6: Forming Candle Momentum ─────────────────────────────
        forming_ok, forming_note = check_forming_momentum(
            forming_candle, direction, elapsed_seconds
        )

        # ── Phase 7: Scoring ──────────────────────────────────────────────
        confidence, base_factors = _compute_confidence(
            sweep_detected      = True,
            volume_spike        = vol.is_spike,
            volume_ratio        = vol.volume_ratio,
            volume_contracting  = vol.is_contracting,
            volume_divergence   = vol.divergence,
            rejection_confirmed = rejection_confirmed,
            ema_aligned         = ema_aligned,
            forming_confirmed   = forming_ok,
            is_pre_close        = is_pre_close,
        )

        # Add context notes to confluence factors
        confluence_factors = base_factors.copy()
        confluence_factors.append(f"📐 {ema_note}")
        confluence_factors.append(f"🕯 {rejection_note}")
        if forming_candle:
            confluence_factors.append(f"🔄 {forming_note}")

        # Append OTC note
        if is_otc:
            confluence_factors.append(
                "🔵 OTC pair — real market data proxy applied (±0.3 pip tolerance)"
            )

        # ── Gate: suppress weak signals ───────────────────────────────────
        if confidence < self.min_confidence:
            return None

        return CRTSignal(
            direction           = direction,
            confidence          = confidence,
            sweep_side          = sweep_side,
            zone                = zone,
            volume              = vol,
            confluence_factors  = confluence_factors,
            is_pre_close        = is_pre_close,
            rejection_confirmed = rejection_confirmed,
            ema_aligned         = ema_aligned,
            raw_score           = confidence,
        )