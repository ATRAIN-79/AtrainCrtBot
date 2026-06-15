"""
╔══════════════════════════════════════════════════════════════╗
║           CANDLE PSYCHOLOGY ENGINE — CORE BRAIN              ║
║   Reads every candle as a battlefield of bulls vs bears      ║
╚══════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: str = ""

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low if self.high != self.low else 0.0001

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0

    @property
    def upper_wick_ratio(self) -> float:
        return self.upper_wick / self.range if self.range > 0 else 0

    @property
    def lower_wick_ratio(self) -> float:
        return self.lower_wick / self.range if self.range > 0 else 0


@dataclass
class CandleSignal:
    pattern: str
    direction: str          # "CALL" or "PUT"
    confidence: float       # 0.0 → 1.0
    description: str
    strength: str           # "WEAK" | "MODERATE" | "STRONG" | "EXTREME"
    emoji: str
    sub_patterns: list = field(default_factory=list)

    @property
    def strength_label(self) -> str:
        if self.confidence >= 0.85:
            return "EXTREME 🔥"
        elif self.confidence >= 0.72:
            return "STRONG 💪"
        elif self.confidence >= 0.58:
            return "MODERATE ✅"
        else:
            return "WEAK ⚠️"

    @property
    def confidence_pct(self) -> int:
        return int(self.confidence * 100)


class CandlePsychologyEngine:
    """
    The heart of the robot.
    Analyses every completed candle + the forming candle
    to extract psychological intent from the market.
    """

    MIN_WICK_RATIO = 0.55        # Minimum wick size for pin bar
    MIN_BODY_RATIO = 0.65        # Minimum body for Marubozu
    DOJI_BODY_LIMIT = 0.08       # Body must be < 8% of range for Doji
    ENGULF_THRESHOLD = 1.05      # Engulfing must be 5% bigger than prior body

    def analyse(self, candles: list[Candle]) -> Optional[CandleSignal]:
        """
        Main entry: takes a list of recent candles (most recent last)
        Returns the strongest signal found, or None.
        """
        if len(candles) < 2:
            return None

        current = candles[-1]
        prev = candles[-2]
        prior = candles[-3] if len(candles) >= 3 else None

        signals = []

        # ── Single-candle patterns ──────────────────────────────────────────
        pin = self._detect_pin_bar(current)
        if pin:
            signals.append(pin)

        marub = self._detect_marubozu(current)
        if marub:
            signals.append(marub)

        doji = self._detect_doji(current)
        if doji:
            signals.append(doji)

        # ── Two-candle patterns ─────────────────────────────────────────────
        engulf = self._detect_engulfing(current, prev)
        if engulf:
            signals.append(engulf)

        tweezer = self._detect_tweezer(current, prev)
        if tweezer:
            signals.append(tweezer)

        harami = self._detect_harami(current, prev)
        if harami:
            signals.append(harami)

        # ── Three-candle patterns ───────────────────────────────────────────
        if prior:
            star = self._detect_star(current, prev, prior)
            if star:
                signals.append(star)

            soldiers = self._detect_soldiers_crows(current, prev, prior)
            if soldiers:
                signals.append(soldiers)

        if not signals:
            return None

        # Return highest confidence signal, merge sub-patterns
        signals.sort(key=lambda s: s.confidence, reverse=True)
        best = signals[0]
        best.sub_patterns = [s.pattern for s in signals[1:] if s.direction == best.direction]
        return best

    # ─────────────────────────────────────────────────────────────────────────
    # PIN BAR  (Hammer / Shooting Star / Hanging Man / Inverted Hammer)
    # Psychology: The market probed a direction aggressively — then REJECTED it
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_pin_bar(self, c: Candle) -> Optional[CandleSignal]:
        if c.range == 0:
            return None

        # Bullish Pin Bar (Hammer): long lower wick, small body at top
        if (c.lower_wick_ratio >= self.MIN_WICK_RATIO and
                c.upper_wick_ratio <= 0.20 and
                c.body_ratio <= 0.35):
            conf = 0.55 + (c.lower_wick_ratio - self.MIN_WICK_RATIO) * 0.8
            conf = min(conf, 0.82)
            return CandleSignal(
                pattern="Hammer / Bullish Pin Bar",
                direction="CALL",
                confidence=conf,
                description=(
                    f"Bears drove price down hard (lower wick={c.lower_wick_ratio:.0%}) "
                    f"but bulls REJECTED it and closed near the top. "
                    f"Sellers exhausted — buyers taking control."
                ),
                strength="STRONG" if conf >= 0.72 else "MODERATE",
                emoji="🔨"
            )

        # Bearish Pin Bar (Shooting Star): long upper wick, small body at bottom
        if (c.upper_wick_ratio >= self.MIN_WICK_RATIO and
                c.lower_wick_ratio <= 0.20 and
                c.body_ratio <= 0.35):
            conf = 0.55 + (c.upper_wick_ratio - self.MIN_WICK_RATIO) * 0.8
            conf = min(conf, 0.82)
            return CandleSignal(
                pattern="Shooting Star / Bearish Pin Bar",
                direction="PUT",
                confidence=conf,
                description=(
                    f"Bulls pushed price up hard (upper wick={c.upper_wick_ratio:.0%}) "
                    f"but bears SLAMMED it back down. "
                    f"Buyers exhausted — sellers taking control."
                ),
                strength="STRONG" if conf >= 0.72 else "MODERATE",
                emoji="🌠"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # MARUBOZU  (No-wick candles)
    # Psychology: Complete dominance by one side — pure momentum
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_marubozu(self, c: Candle) -> Optional[CandleSignal]:
        if c.body_ratio < self.MIN_BODY_RATIO:
            return None

        no_upper = c.upper_wick_ratio < 0.06
        no_lower = c.lower_wick_ratio < 0.06
        conf = 0.60 + c.body_ratio * 0.30

        if c.is_bullish and no_upper and no_lower:
            return CandleSignal(
                pattern="Bullish Marubozu",
                direction="CALL",
                confidence=min(conf, 0.88),
                description=(
                    f"Bulls dominated ENTIRELY — open to close with zero opposition. "
                    f"Body={c.body_ratio:.0%} of candle range. Pure buying momentum."
                ),
                strength="EXTREME" if conf >= 0.82 else "STRONG",
                emoji="🟢"
            )

        if c.is_bearish and no_upper and no_lower:
            return CandleSignal(
                pattern="Bearish Marubozu",
                direction="PUT",
                confidence=min(conf, 0.88),
                description=(
                    f"Bears dominated ENTIRELY — open to close with zero opposition. "
                    f"Body={c.body_ratio:.0%} of candle range. Pure selling momentum."
                ),
                strength="EXTREME" if conf >= 0.82 else "STRONG",
                emoji="🔴"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # DOJI  (Open ≈ Close — battlefield standoff)
    # Psychology: Complete indecision. The NEXT candle reveals winner.
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_doji(self, c: Candle) -> Optional[CandleSignal]:
        if c.body_ratio > self.DOJI_BODY_LIMIT:
            return None

        # Dragonfly Doji: long lower wick only → bullish
        if c.lower_wick_ratio >= 0.60 and c.upper_wick_ratio <= 0.15:
            return CandleSignal(
                pattern="Dragonfly Doji",
                direction="CALL",
                confidence=0.70,
                description=(
                    "Bears pushed hard then surrendered completely. "
                    "Price returned to open — buyers absorbed ALL selling. Bullish reversal."
                ),
                strength="STRONG",
                emoji="🐉"
            )

        # Gravestone Doji: long upper wick only → bearish
        if c.upper_wick_ratio >= 0.60 and c.lower_wick_ratio <= 0.15:
            return CandleSignal(
                pattern="Gravestone Doji",
                direction="PUT",
                confidence=0.70,
                description=(
                    "Bulls pushed high then collapsed completely. "
                    "Price returned to open — sellers absorbed ALL buying. Bearish reversal."
                ),
                strength="STRONG",
                emoji="🪦"
            )

        # Standard Doji — indecision, slight bias from wick side
        direction = "CALL" if c.lower_wick > c.upper_wick else "PUT"
        return CandleSignal(
            pattern="Standard Doji",
            direction=direction,
            confidence=0.52,
            description=(
                "Market in complete equilibrium. "
                "Neither bulls nor bears won this candle. "
                "Wait for next candle to confirm direction."
            ),
            strength="WEAK",
            emoji="✝️"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ENGULFING  (Two-candle reversal — momentum shift)
    # Psychology: The new candle completely swallowed the previous — total takeover
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_engulfing(self, current: Candle, prev: Candle) -> Optional[CandleSignal]:
        if prev.body == 0:
            return None

        engulf_ratio = current.body / prev.body if prev.body > 0 else 0

        # Bullish Engulfing: current bullish, prev bearish, current body > prev body
        if (current.is_bullish and prev.is_bearish and
                current.open <= prev.close and
                current.close >= prev.open and
                engulf_ratio >= self.ENGULF_THRESHOLD):
            conf = 0.62 + min((engulf_ratio - 1.0) * 0.25, 0.20)
            return CandleSignal(
                pattern="Bullish Engulfing",
                direction="CALL",
                confidence=min(conf, 0.85),
                description=(
                    f"Bulls completely overpowered the prior bearish candle "
                    f"(engulf ratio: {engulf_ratio:.1f}x). "
                    f"A complete power transfer from sellers to buyers."
                ),
                strength="STRONG",
                emoji="🟢🔥"
            )

        # Bearish Engulfing: current bearish, prev bullish, current body > prev body
        if (current.is_bearish and prev.is_bullish and
                current.open >= prev.close and
                current.close <= prev.open and
                engulf_ratio >= self.ENGULF_THRESHOLD):
            conf = 0.62 + min((engulf_ratio - 1.0) * 0.25, 0.20)
            return CandleSignal(
                pattern="Bearish Engulfing",
                direction="PUT",
                confidence=min(conf, 0.85),
                description=(
                    f"Bears completely overpowered the prior bullish candle "
                    f"(engulf ratio: {engulf_ratio:.1f}x). "
                    f"A complete power transfer from buyers to sellers."
                ),
                strength="STRONG",
                emoji="🔴🔥"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # TWEEZER TOPS / BOTTOMS
    # Psychology: Two failed attempts at the same level — reversal imminent
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_tweezer(self, current: Candle, prev: Candle) -> Optional[CandleSignal]:
        tolerance = current.range * 0.03

        # Tweezer Top: both candles hit same high → PUT
        if (abs(current.high - prev.high) <= tolerance and
                current.is_bearish and prev.is_bullish):
            return CandleSignal(
                pattern="Tweezer Top",
                direction="PUT",
                confidence=0.68,
                description=(
                    f"Market tested the same high TWICE ({current.high:.5f}) "
                    f"and was rejected both times. Double resistance rejection."
                ),
                strength="MODERATE",
                emoji="📌🔴"
            )

        # Tweezer Bottom: both candles hit same low → CALL
        if (abs(current.low - prev.low) <= tolerance and
                current.is_bullish and prev.is_bearish):
            return CandleSignal(
                pattern="Tweezer Bottom",
                direction="CALL",
                confidence=0.68,
                description=(
                    f"Market tested the same low TWICE ({current.low:.5f}) "
                    f"and bounced both times. Double support confirmation."
                ),
                strength="MODERATE",
                emoji="📌🟢"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # HARAMI  (Inside bar — compression before explosion)
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_harami(self, current: Candle, prev: Candle) -> Optional[CandleSignal]:
        if prev.body == 0:
            return None

        # Current body must be inside previous body
        curr_top = max(current.open, current.close)
        curr_bot = min(current.open, current.close)
        prev_top = max(prev.open, prev.close)
        prev_bot = min(prev.open, prev.close)

        if curr_top <= prev_top and curr_bot >= prev_bot:
            # Bullish harami: prev bearish, current bullish small body
            if prev.is_bearish and current.is_bullish:
                return CandleSignal(
                    pattern="Bullish Harami",
                    direction="CALL",
                    confidence=0.60,
                    description=(
                        "Bearish momentum stalling — small bullish candle formed inside "
                        "prior bearish body. Sellers losing conviction, buyers stepping in."
                    ),
                    strength="MODERATE",
                    emoji="🤰🟢"
                )

            if prev.is_bullish and current.is_bearish:
                return CandleSignal(
                    pattern="Bearish Harami",
                    direction="PUT",
                    confidence=0.60,
                    description=(
                        "Bullish momentum stalling — small bearish candle formed inside "
                        "prior bullish body. Buyers losing conviction, sellers stepping in."
                    ),
                    strength="MODERATE",
                    emoji="🤰🔴"
                )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # MORNING / EVENING STAR  (3-candle reversal)
    # Psychology: Trend candle → indecision → explosive reversal
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_star(self, current: Candle, prev: Candle, prior: Candle) -> Optional[CandleSignal]:
        # Morning Star: prior bearish big → prev small body (gap/doji) → current bullish big
        if (prior.is_bearish and prior.body_ratio > 0.50 and
                prev.body_ratio < 0.35 and
                current.is_bullish and current.body_ratio > 0.50):
            return CandleSignal(
                pattern="Morning Star",
                direction="CALL",
                confidence=0.80,
                description=(
                    "Classic 3-candle reversal: strong bearish momentum → indecision → "
                    "powerful bullish recovery. Bears surrendered to bulls over 3 candles."
                ),
                strength="STRONG",
                emoji="🌅"
            )

        # Evening Star: prior bullish big → prev small body → current bearish big
        if (prior.is_bullish and prior.body_ratio > 0.50 and
                prev.body_ratio < 0.35 and
                current.is_bearish and current.body_ratio > 0.50):
            return CandleSignal(
                pattern="Evening Star",
                direction="PUT",
                confidence=0.80,
                description=(
                    "Classic 3-candle reversal: strong bullish momentum → indecision → "
                    "powerful bearish collapse. Bulls surrendered to bears over 3 candles."
                ),
                strength="STRONG",
                emoji="🌆"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # THREE WHITE SOLDIERS / THREE BLACK CROWS
    # Psychology: Pure trend confirmation over 3 consecutive candles
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_soldiers_crows(self, current: Candle, prev: Candle, prior: Candle) -> Optional[CandleSignal]:
        # Three White Soldiers: 3 consecutive bullish candles, each closing higher
        if (current.is_bullish and prev.is_bullish and prior.is_bullish and
                current.close > prev.close > prior.close and
                current.body_ratio > 0.45 and prev.body_ratio > 0.45):
            return CandleSignal(
                pattern="Three White Soldiers",
                direction="CALL",
                confidence=0.82,
                description=(
                    "Three consecutive strong bullish candles each closing higher. "
                    "Bulls are in full control — a powerful continuation signal."
                ),
                strength="STRONG",
                emoji="⚔️🟢"
            )

        # Three Black Crows: 3 consecutive bearish candles, each closing lower
        if (current.is_bearish and prev.is_bearish and prior.is_bearish and
                current.close < prev.close < prior.close and
                current.body_ratio > 0.45 and prev.body_ratio > 0.45):
            return CandleSignal(
                pattern="Three Black Crows",
                direction="PUT",
                confidence=0.82,
                description=(
                    "Three consecutive strong bearish candles each closing lower. "
                    "Bears are in full control — a powerful continuation signal."
                ),
                strength="STRONG",
                emoji="🐦‍⬛🔴"
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE CANDLE PRE-CLOSE PREDICTOR
    # Reads the forming candle at ~45s and projects closing direction
    # ─────────────────────────────────────────────────────────────────────────
    def predict_forming_candle(
        self,
        forming: Candle,
        elapsed_seconds: float,
        candle_duration: float = 60.0
    ) -> Optional[CandleSignal]:
        """
        Called with a live (not yet closed) candle.
        Projects where the candle will close based on:
        - Current body direction and momentum
        - Wick development pattern
        - Time elapsed (confidence grows with time)
        """
        if elapsed_seconds < 10:
            return None  # Too early to judge

        progress = elapsed_seconds / candle_duration
        body_direction = "CALL" if forming.is_bullish else "PUT"

        # Momentum score: strong body forming + wicks on opposite side
        momentum = forming.body_ratio * progress

        # Rejection pattern forming in real-time
        if forming.is_bullish and forming.lower_wick_ratio > 0.40:
            momentum *= 1.25  # Bulls absorbed the dip — increasing momentum

        if forming.is_bearish and forming.upper_wick_ratio > 0.40:
            momentum *= 1.25  # Bears absorbed the rally — increasing momentum

        confidence = 0.45 + (momentum * 0.40) + (progress * 0.10)
        confidence = min(confidence, 0.78)

        if confidence < 0.52:
            return None

        return CandleSignal(
            pattern=f"Live Candle Projection ({int(elapsed_seconds)}s elapsed)",
            direction=body_direction,
            confidence=confidence,
            description=(
                f"Forming candle ({int(progress*100)}% complete): "
                f"{'Bullish' if forming.is_bullish else 'Bearish'} body "
                f"({forming.body_ratio:.0%} of range). "
                f"Momentum: {'Strong' if momentum > 0.4 else 'Moderate'}. "
                f"Projected close: {'UP ↑' if body_direction == 'CALL' else 'DOWN ↓'}."
            ),
            strength="MODERATE",
            emoji="⚡"
        )
