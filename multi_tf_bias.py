"""
multi_tf_bias.py
════════════════════════════════════════════════════════════════════
ATRAIN FOREX — Multi-Timeframe Bias Engine
Ported from Binary_Forex_Forecast_Pro v4.00 (MQL4 → Python)

Scores any timeframe bar using 8 independent components:
  1. EMA Stack  20/50/200              max ±0.20
  2. Price Structure HH/HL · LH/LL     max ±0.25
  3. RSI(14) Momentum                  max ±0.15
  4. MACD(12,26,9) Histogram           max ±0.12
  5. Stochastic(14,3,3)                max ±0.08
  6. Candle Body Strength              max ±0.10
  7. Bollinger Bands(20,2)             max ±0.08
  8. Tick-Volume Ratio                 max ±0.12 / ×0.85 dampen

Theoretical max ±1.10 → clamped to ±1.0

Multi-TF weights for BINARY M1 mode (matches MQL4 original):
  M1 × 0.45 · M5 × 0.25 · M15 × 0.15 · H1 × 0.15
════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OHLCVBar:
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float = 0.0


# ── Low-level indicator maths ────────────────────────────────────────────────

def _ema(values: List[float], period: int) -> List[float]:
    """Standard EMA using Wilder multiplier 2/(period+1)."""
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2.0 / (period + 1)
    out = [float("nan")] * len(values)
    # seed with SMA
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(values: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1 : i + 1]) / period
    return out


def _rsi(closes: List[float], period: int = 14) -> List[float]:
    """Wilder-smoothed RSI — matches MT4 iRSI."""
    out = [float("nan")] * len(closes)
    if len(closes) < period + 1:
        return out
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    if al == 0:
        out[period] = 100.0
    else:
        rs = ag / al
        out[period] = 100.0 - 100.0 / (1 + rs)
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            out[i + 1] = 100.0
        else:
            rs = ag / al
            out[i + 1] = 100.0 - 100.0 / (1 + rs)
    return out


def _macd(closes: List[float], fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line) arrays."""
    e_fast = _ema(closes, fast)
    e_slow = _ema(closes, slow)
    macd_line = [
        (f - s) if not (np.isnan(f) or np.isnan(s)) else float("nan")
        for f, s in zip(e_fast, e_slow)
    ]
    # signal line is EMA of macd_line (skip nans)
    valid_start = next((i for i, v in enumerate(macd_line) if not np.isnan(v)), None)
    sig_line = [float("nan")] * len(macd_line)
    if valid_start is not None:
        segment = macd_line[valid_start:]
        seg_ema = _ema(segment, signal)
        for i, v in enumerate(seg_ema):
            sig_line[valid_start + i] = v
    return macd_line, sig_line


def _bollinger(closes: List[float], period=20, mult=2.0):
    """Returns (upper, mid, lower) arrays."""
    mid = _sma(closes, period)
    upper, lower = [float("nan")] * len(closes), [float("nan")] * len(closes)
    for i in range(period - 1, len(closes)):
        std = np.std(closes[i - period + 1 : i + 1], ddof=0)
        upper[i] = mid[i] + mult * std
        lower[i] = mid[i] - mult * std
    return upper, mid, lower


def _stochastic(highs, lows, closes, k_period=14, d_period=3, slowing=3):
    """
    Full Stochastic (%K smoothed, %D signal) — matches MT4 iStochastic MODE_SMA.
    Returns (k_values, d_values).
    """
    n = len(closes)
    raw_k = [float("nan")] * n
    for i in range(k_period - 1, n):
        hh = max(highs[i - k_period + 1 : i + 1])
        ll = min(lows[i - k_period + 1 : i + 1])
        if hh == ll:
            raw_k[i] = 50.0
        else:
            raw_k[i] = 100.0 * (closes[i] - ll) / (hh - ll)

    # Apply slowing (SMA of raw %K)
    k_vals = _sma(raw_k, slowing)
    # %D is SMA of smoothed %K
    d_vals = _sma(k_vals, d_period)
    return k_vals, d_vals


# ── Single-TF scorer ─────────────────────────────────────────────────────────

def score_single_tf(bars: List[OHLCVBar], shift: int = 1) -> float:
    """
    Score one timeframe at `shift` (0 = forming, 1 = last closed).
    Returns a value clamped to [-1.0, +1.0].
    Requires at least 210 bars for EMA-200 warmup.
    """
    MIN_BARS = 50
    if len(bars) < MIN_BARS + shift + 3:
        return 0.0

    closes  = [b.close  for b in bars]
    opens   = [b.open   for b in bars]
    highs   = [b.high   for b in bars]
    lows    = [b.low    for b in bars]
    volumes = [b.volume for b in bars]

    # Most-recent index = last in list (MT4 reverses; we keep chronological order)
    # shift=1 → second from last, shift=0 → last
    i    = len(bars) - 1 - shift   # current eval bar
    i1   = i - 1                   # one bar back
    i2   = i - 2                   # two bars back

    if i2 < 0:
        return 0.0

    score = 0.0

    # ── 1. EMA Stack ─────────────────────────────────────────────────────────
    ema20  = _ema(closes, 20)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200) if len(closes) >= 200 else [float("nan")] * len(closes)

    e20  = ema20[i]
    e50  = ema50[i]
    e200 = ema200[i]
    c    = closes[i]

    if not any(np.isnan(v) for v in [e20, e50, e200]):
        if e20 > e50 > e200 and c > e20:
            score += 0.20
        elif e20 < e50 < e200 and c < e20:
            score -= 0.20
        elif c > e20:
            score += 0.08
        else:
            score -= 0.08
    elif not np.isnan(e20):
        score += 0.08 if c > e20 else -0.08

    # ── 2. Price Structure HH/HL · LH/LL ─────────────────────────────────────
    h0, h1, h2 = highs[i], highs[i1], highs[i2]
    l0, l1, l2 = lows[i],  lows[i1],  lows[i2]

    hhhl = (h0 > h1 > h2) and (l0 > l1 > l2)
    lhll = (h0 < h1 < h2) and (l0 < l1 < l2)

    if hhhl:
        score += 0.25
    elif lhll:
        score -= 0.25
    elif h0 > h1:
        score += 0.10
    elif l0 < l1:
        score -= 0.10

    # ── 3. RSI(14) ────────────────────────────────────────────────────────────
    rsi_vals = _rsi(closes, 14)
    rsi0 = rsi_vals[i]
    rsi1 = rsi_vals[i1] if i1 >= 0 else float("nan")

    if not np.isnan(rsi0):
        if rsi0 > 60 and not np.isnan(rsi1) and rsi0 > rsi1:
            score += 0.15
        elif rsi0 > 55:
            score += 0.08
        elif rsi0 < 40 and not np.isnan(rsi1) and rsi0 < rsi1:
            score -= 0.15
        elif rsi0 < 45:
            score -= 0.08

    # ── 4. MACD(12,26,9) ─────────────────────────────────────────────────────
    macd_line, sig_line = _macd(closes, 12, 26, 9)
    m0  = macd_line[i]
    s0  = sig_line[i]
    m1_ = macd_line[i1]
    s1_ = sig_line[i1]

    if not any(np.isnan(v) for v in [m0, s0, m1_, s1_]):
        hist0 = m0 - s0
        hist1 = m1_ - s1_
        if m0 > 0 and hist0 > 0 and hist0 > hist1:
            score += 0.12
        elif m0 < 0 and hist0 < 0 and hist0 < hist1:
            score -= 0.12
        elif m0 > 0:
            score += 0.05
        else:
            score -= 0.05

    # ── 5. Stochastic(14,3,3) ─────────────────────────────────────────────────
    stk, std = _stochastic(highs, lows, closes, 14, 3, 3)
    sk0, sd0 = stk[i],  std[i]
    sk1, sd1 = stk[i1], std[i1]

    if not any(np.isnan(v) for v in [sk0, sd0, sk1, sd1]):
        if sk0 > 80 and sk0 < sk1:
            score -= 0.08
        elif sk0 < 20 and sk0 > sk1:
            score += 0.08
        elif sk0 > sd0 and sk1 < sd1 and sk0 < 50:
            score += 0.05
        elif sk0 < sd0 and sk1 > sd1 and sk0 > 50:
            score -= 0.05

    # ── 6. Candle Body Strength ───────────────────────────────────────────────
    body  = abs(closes[i] - opens[i])
    rng   = highs[i] - lows[i]
    if rng > 0 and (body / rng) > 0.60:
        score += 0.10 if closes[i] > opens[i] else -0.10

    # ── 7. Bollinger Bands(20,2) ──────────────────────────────────────────────
    bb_u, bb_m, bb_l = _bollinger(closes, 20, 2.0)
    u0, m0b, l0 = bb_u[i], bb_m[i], bb_l[i]

    if not any(np.isnan(v) for v in [u0, m0b, l0]):
        if closes[i] > u0:
            score -= 0.08
        elif closes[i] < l0:
            score += 0.08
        elif closes[i] > m0b:
            score += 0.04
        else:
            score -= 0.04

    # ── 8. Tick-Volume Ratio ──────────────────────────────────────────────────
    vol_start = i - 14
    if vol_start >= 0:
        avg_vol = sum(volumes[vol_start:i]) / 14.0
        if avg_vol > 0:
            vol_ratio = volumes[i] / avg_vol
            if vol_ratio >= 1.40:
                score += 0.12 if closes[i] > opens[i] else -0.12
            elif vol_ratio >= 1.15:
                score += 0.06 if closes[i] > opens[i] else -0.06
            elif vol_ratio < 0.70:
                score *= 0.85  # dampen on thin bar

    return max(-1.0, min(1.0, score))


# ── Multi-TF weighted bias ────────────────────────────────────────────────────

# Weights matching MQL4 Binary M1 mode
BINARY_M1_WEIGHTS = [
    ("M1",  0.45),
    ("M5",  0.25),
    ("M15", 0.15),
    ("H1",  0.15),
]


class MultiTFBiasEngine:
    """
    Computes a weighted multi-timeframe bias score for binary M1 signals.

    Usage:
        engine = MultiTFBiasEngine()
        result = engine.evaluate(tf_bars_dict, shift=1)

    tf_bars_dict: {"M1": [OHLCVBar, ...], "M5": [...], "M15": [...], "H1": [...]}
    shift: 1 → closed candle (for alerts), 0 → forming (for display)

    Returns BiasResult with score, stage, direction, and per-TF breakdown.
    """

    # Stage thresholds (matching MQL4 defaults)
    THRESH_WATCH     = 0.22
    THRESH_HIGH      = 0.35
    THRESH_VERY_HIGH = 0.45

    def evaluate(
        self,
        tf_bars: dict[str, List[OHLCVBar]],
        shift: int = 1,
    ) -> "BiasResult":
        total_score  = 0.0
        total_weight = 0.0
        breakdown: dict[str, float] = {}

        for tf_label, weight in BINARY_M1_WEIGHTS:
            bars = tf_bars.get(tf_label)
            if bars and len(bars) >= 55:
                s = score_single_tf(bars, shift=shift)
            else:
                s = 0.0
            breakdown[tf_label] = s
            total_score  += s * weight
            total_weight += weight

        if total_weight <= 0:
            norm = 0.0
        else:
            norm = max(-1.0, min(1.0, total_score / total_weight))

        abs_score = abs(norm)
        direction = "CALL" if norm > 0 else "PUT" if norm < 0 else "NEUTRAL"

        if abs_score >= self.THRESH_VERY_HIGH:
            stage = "ENTER"
        elif abs_score >= self.THRESH_HIGH:
            stage = "PREPARE"
        elif abs_score >= self.THRESH_WATCH:
            stage = "WATCH"
        else:
            stage = "NONE"

        return BiasResult(
            score=norm,
            abs_score=abs_score,
            direction=direction,
            stage=stage,
            breakdown=breakdown,
        )


@dataclass
class BiasResult:
    score:     float          # -1.0 … +1.0
    abs_score: float          # |score|
    direction: str            # "CALL" / "PUT" / "NEUTRAL"
    stage:     str            # "NONE" / "WATCH" / "PREPARE" / "ENTER"
    breakdown: dict           # per-TF raw scores

    def is_valid_signal(self) -> bool:
        """True when the closed-candle score clears the HIGH threshold."""
        return self.stage in ("PREPARE", "ENTER")

    def confluence_lines(self) -> list[str]:
        """Returns formatted strings for the bot signal message."""
        lines = []
        if self.stage == "ENTER":
            lines.append(f"✅ MTF Bias: VERY HIGH ({self.abs_score:.2f}) — confirmed")
        elif self.stage == "PREPARE":
            lines.append(f"✅ MTF Bias: HIGH ({self.abs_score:.2f}) — prepare entry")
        elif self.stage == "WATCH":
            lines.append(f"⚠️ MTF Bias: building ({self.abs_score:.2f})")
        bd = self.breakdown
        aligned = [tf for tf, s in bd.items()
                   if (s > 0.15 and self.direction == "CALL") or
                      (s < -0.15 and self.direction == "PUT")]
        if aligned:
            lines.append(f"✅ TF alignment: {', '.join(aligned)}")
        return lines