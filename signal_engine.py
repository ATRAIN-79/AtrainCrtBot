"""
signal_engine.py  — ATRAIN FOREX Binary Signal Robot v2.3
══════════════════════════════════════════════════════════════════════
CHANGES FROM v2.2
  ✦ Band filter REMOVED entirely — any score ≥ MIN_CONFIDENCE sends
  ✦ All score penalties REMOVED — layers are additive only
  ✦ Layer 1 fallback: if candle engine returns no direction,
    derive direction from close > open (bullish) or close < open
    so a signal is never killed by a missing candle pattern label
  ✦ Layer 3 ATR check removed — was killing OTC pairs constantly
  ✦ Layer 3 RSI: reward only, no penalty
  ✦ Layer 4 EMA: counter-trend no longer penalises, just no bonus
  ✦ Layer 6 MTF disagree: penalty removed — only add pts on agree
  ✦ Duplicate-signal window relaxed to 60s (was 90s) in bot.py
  ✦ MIN_CONFIDENCE = 55 — unchanged, as requested
══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from candle_psychology import Candle, CandlePsychologyEngine, CandleSignal
from multi_tf_bias import MultiTFBiasEngine, OHLCVBar

logger = logging.getLogger(__name__)


@dataclass
class FinalSignal:
    asset:              str
    direction:          str       # "CALL" or "PUT"
    confidence:         int       # 0–100
    expiry:             str
    timestamp:          str
    candle_pattern:     str
    candle_emoji:       str
    pre_close:          bool
    confluence_factors: list[str] = field(default_factory=list)


class ConfluenceEngine:
    """
    Additive-only confluence scorer.

    Layer 1 — Candle Psychology    base confidence from candle engine
    Layer 2 — CRT + Volume         +7 engulf, +8 volume spike
    Layer 3 — RSI momentum         +5 if aligned (no penalty)
    Layer 4 — EMA stack            +4 / +8 if aligned (no penalty)
    Layer 5 — S&R proximity        +6 if at swing zone
    Layer 6 — Multi-TF Bias        +8/+18/+25 if HTF agrees (no penalty)

    Signal emits if final score >= MIN_CONFIDENCE (55).
    No band filter. No penalties. Direction from candle or price action.
    """

    MIN_CONFIDENCE        = 55
    PRE_CLOSE_SECONDS     = 45
    CORRELATION_TRIGGER_SCORE = 0.25   # lowered from 0.30 → more correlation fires

    _mtf = MultiTFBiasEngine()
    _cp  = CandlePsychologyEngine()

    MTF_WEIGHT_ENTER   = 25
    MTF_WEIGHT_PREPARE = 18
    MTF_WEIGHT_WATCH   = 8

    def evaluate(
        self,
        asset:           str,
        candles:         List[Candle],
        forming_candle:  Optional[Candle],
        elapsed_seconds: int,
        timestamp:       str,
        tf_bars:         Optional[dict[str, List[OHLCVBar]]] = None,
    ) -> Optional[FinalSignal]:

        if len(candles) < 10:
            return None

        closed   = candles[-2]
        prev     = candles[-3]
        lookback = candles[:-1]

        direction, base_conf, pattern, emoji, factors = self._layer1_candle(
            closed, prev, lookback
        )
        if direction is None:
            return None

        conf = base_conf

        conf, factors = self._layer2_crt_volume(closed, prev, lookback, direction, conf, factors)
        conf, factors = self._layer3_rsi(lookback, direction, conf, factors)
        conf, factors = self._layer4_ema(lookback, direction, conf, factors)
        conf, factors = self._layer5_sr(closed, lookback, direction, conf, factors)

        if tf_bars:
            conf, factors = self._layer6_mtf_bias(tf_bars, direction, conf, factors)

        conf = max(0, min(100, conf))

        if conf < self.MIN_CONFIDENCE:
            return None

        expiry    = "1 Minute" if elapsed_seconds < self.PRE_CLOSE_SECONDS else "Next M1"
        pre_close = elapsed_seconds >= self.PRE_CLOSE_SECONDS

        return FinalSignal(
            asset=asset,
            direction=direction,
            confidence=conf,
            expiry=expiry,
            timestamp=timestamp,
            candle_pattern=pattern,
            candle_emoji=emoji,
            pre_close=pre_close,
            confluence_factors=factors,
        )

    def check_correlation_ready(
        self,
        tf_bars: dict[str, List[OHLCVBar]],
    ) -> tuple[bool, str]:
        try:
            result = self._mtf.evaluate(tf_bars, shift=0)
            ready  = result.abs_score >= self.CORRELATION_TRIGGER_SCORE
            return ready, result.direction
        except Exception as e:
            logger.debug(f"check_correlation_ready error: {e}")
            return False, "NEUTRAL"

    # ─────────────────────────────────────────────────────────────────────────
    # Layers
    # ─────────────────────────────────────────────────────────────────────────

    def _layer1_candle(self, closed: Candle, prev: Candle, lookback: List[Candle]):
        """
        Primary direction + base confidence.

        Strategy:
          1. Try CandlePsychologyEngine.analyse() — use its direction + confidence
          2. If engine returns None or direction not CALL/PUT, fall back to
             raw price action: close > open → CALL, close < open → PUT
             with base confidence of 55 (minimum passing score)
        This guarantees Layer 1 never silently kills a bar with a valid move.
        """
        direction = None
        pattern   = "Price Action"
        emoji     = "🕯"
        base_conf = 55
        factors   = []

        try:
            result: CandleSignal = self._cp.analyse(closed, prev, lookback)
            if result is not None and result.direction in ("CALL", "PUT"):
                direction = result.direction
                pattern   = result.pattern
                emoji     = result.emoji
                base_conf = max(55, int(result.confidence))
                strength  = result.strength
                factors.append(
                    f"✅ {pattern} {emoji} — {strength} ({base_conf}% candle confidence)"
                )
                if result.sub_patterns:
                    sub = ", ".join(result.sub_patterns[:2])
                    factors.append(f"✅ Sub-patterns: {sub}")
                if result.description:
                    factors.append(f"✅ {result.description}")
        except Exception as e:
            logger.debug(f"Candle engine error: {e}")

        # Fallback — derive direction from raw price action
        if direction is None:
            if closed.close > closed.open:
                direction = "CALL"
                factors.append("✅ Bullish price action (close > open)")
            elif closed.close < closed.open:
                direction = "PUT"
                factors.append("✅ Bearish price action (close < open)")
            else:
                return None, None, None, None, None   # doji with no pattern — skip

        return direction, base_conf, pattern, emoji, factors

    def _layer2_crt_volume(
        self,
        closed: Candle, prev: Candle, lookback: List[Candle],
        direction: str, conf: int, factors: list
    ):
        factors.append("━━━ CRT + VOLUME LAYER ━━━")
        try:
            vols    = [c.volume for c in lookback[-20:] if c.volume > 0]
            avg_vol = np.mean(vols) if vols else 0

            if avg_vol > 0 and closed.volume > avg_vol * 1.3:
                conf += 8
                factors.append("✅ Volume spike confirms move")

            body_closed = abs(closed.close - closed.open)
            body_prev   = abs(prev.close - prev.open)
            if body_closed > body_prev * 1.1:
                if direction == "CALL" and closed.close > prev.high:
                    conf += 7
                    factors.append("✅ Bullish engulf / CRT sweep")
                elif direction == "PUT" and closed.close < prev.low:
                    conf += 7
                    factors.append("✅ Bearish engulf / CRT sweep")
                else:
                    # Larger body regardless — still a confidence add
                    conf += 3
                    factors.append("✅ Strong body expansion")

        except Exception as e:
            logger.debug(f"Layer 2 error: {e}")
        return conf, factors

    def _layer3_rsi(
        self, lookback: List[Candle], direction: str, conf: int, factors: list
    ):
        """RSI momentum bonus only — no penalties."""
        factors.append("━━━ ACCURACY FILTERS ━━━")
        try:
            closes = [c.close for c in lookback]
            if len(closes) < 16:
                return conf, factors

            gains, losses = [], []
            for i in range(1, len(closes)):
                d = closes[i] - closes[i - 1]
                gains.append(max(d, 0))
                losses.append(max(-d, 0))

            ag = sum(gains[:14]) / 14
            al = sum(losses[:14]) / 14
            for i in range(14, len(gains)):
                ag = (ag * 13 + gains[i]) / 14
                al = (al * 13 + losses[i]) / 14
            rsi = 100 - 100 / (1 + ag / al) if al > 0 else 50.0

            if direction == "CALL" and rsi > 50:
                conf += 5
                factors.append(f"✅ RSI bullish ({rsi:.1f})")
            elif direction == "PUT" and rsi < 50:
                conf += 5
                factors.append(f"✅ RSI bearish ({rsi:.1f})")
            else:
                factors.append(f"  RSI neutral ({rsi:.1f})")

        except Exception as e:
            logger.debug(f"Layer 3 error: {e}")
        return conf, factors

    def _layer4_ema(
        self, lookback: List[Candle], direction: str, conf: int, factors: list
    ):
        """EMA alignment bonus only — no counter-trend penalty."""
        try:
            closes = [c.close for c in lookback]
            if len(closes) < 50:
                return conf, factors

            def ema_last(vals, p):
                k = 2 / (p + 1)
                v = sum(vals[:p]) / p
                for x in vals[p:]:
                    v = x * k + v * (1 - k)
                return v

            e9    = ema_last(closes, 9)
            e21   = ema_last(closes, 21)
            e50   = ema_last(closes, 50)
            price = closes[-1]

            if direction == "CALL":
                if price > e9 > e21 > e50:
                    conf += 8
                    factors.append("✅ EMA 9>21>50 full stack — uptrend")
                elif price > e21:
                    conf += 4
                    factors.append("✅ Price above EMA21")
                elif price > e50:
                    conf += 2
                    factors.append("✅ Price above EMA50")
                # No penalty for counter-trend
            else:
                if price < e9 < e21 < e50:
                    conf += 8
                    factors.append("✅ EMA 9<21<50 full stack — downtrend")
                elif price < e21:
                    conf += 4
                    factors.append("✅ Price below EMA21")
                elif price < e50:
                    conf += 2
                    factors.append("✅ Price below EMA50")

        except Exception as e:
            logger.debug(f"Layer 4 error: {e}")
        return conf, factors

    def _layer5_sr(
        self, closed: Candle, lookback: List[Candle],
        direction: str, conf: int, factors: list
    ):
        factors.append("━━━ ICT FVG STRUCTURE ━━━")
        try:
            highs  = [c.high for c in lookback[-30:]]
            lows   = [c.low  for c in lookback[-30:]]
            price  = closed.close
            # Widen proximity band to 8% of range (was 5%)
            spread = (max(highs) - min(lows)) * 0.08

            swing_high = max(highs[:-3])
            swing_low  = min(lows[:-3])

            if direction == "CALL" and abs(price - swing_low) < spread:
                conf += 6
                factors.append("✅ Near swing low support — bullish bounce zone")
            elif direction == "PUT" and abs(price - swing_high) < spread:
                conf += 6
                factors.append("✅ Near swing high resistance — bearish rejection zone")

        except Exception as e:
            logger.debug(f"Layer 5 error: {e}")
        return conf, factors

    def _layer6_mtf_bias(
        self,
        tf_bars: dict[str, list],
        direction: str,
        conf: int,
        factors: list,
    ):
        """MTF bonus on agreement only — zero penalty on disagreement."""
        try:
            result = self._mtf.evaluate(tf_bars, shift=1)

            agrees = (
                (direction == "CALL" and result.direction == "CALL") or
                (direction == "PUT"  and result.direction == "PUT")
            )

            if agrees:
                if result.stage == "ENTER":
                    pts = self.MTF_WEIGHT_ENTER
                elif result.stage == "PREPARE":
                    pts = self.MTF_WEIGHT_PREPARE
                elif result.stage == "WATCH":
                    pts = self.MTF_WEIGHT_WATCH
                else:
                    pts = 0

                if pts > 0:
                    conf += pts
                    for line in result.confluence_lines():
                        factors.append(line)
            else:
                factors.append(f"  MTF Bias: {result.direction} (monitoring)")

            bd      = result.breakdown
            best_tf = max(bd, key=lambda t: abs(bd[t]))
            factors.append(f"✅ Strongest HTF: {best_tf} ({bd[best_tf]:+.2f})")

        except Exception as e:
            logger.warning(f"Layer 6 MTF bias error: {e}")

        return conf, factors