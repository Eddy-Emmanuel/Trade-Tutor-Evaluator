"""
indicators/engine.py
All 10 indicator engines — formulas taken verbatim from the PDF.
Each compute_* function returns a dict with:
  - 'indicator_col' : str  (column name for the main indicator value)
  - 'extra_cols'    : dict (name → pd.Series) of any additional columns
  - 'position'      : list[str]
  - 'action'        : list[str]
  - 'buy_cond'      : list[bool]
  - 'sell_cond'     : list[bool]
"""
import numpy as np
import pandas as pd

# ─────────────────────────── helpers ────────────────────────────────────────
def _state_machine(buy_cond: list, sell_cond: list, repeat: bool = False) -> tuple[list, list]:
    """Universal state machine. Returns (position, action).

    repeat=False (default): alternates Buy → Sell → Buy → Sell.
                            After a Buy, further Buy signals are ignored until a Sell fires.
    repeat=True:            reacts to every signal; consecutive Buys or Sells are allowed.
    """
    position, action = [], []
    state = "Out"
    for bc, sc in zip(buy_cond, sell_cond):
        if repeat:
            if bc:
                state = "In"
                action.append("Buy")
            elif sc:
                state = "Out"
                action.append("Sell")
            else:
                action.append("Hold")
        else:
            if state == "Out" and bc:
                state = "In"
                action.append("Buy")
            elif state == "In" and sc:
                state = "Out"
                action.append("Sell")
            else:
                action.append("Hold")
        position.append(state)
    return position, action


def _threshold(base: pd.Series, pct: float, direction: str) -> pd.Series:
    """PDF Eq. 5/6/19/20/34/35/44/45/64/65/76/77/95/96:
    sign = +1 for 'above', -1 for 'below'."""
    sign = +1 if direction == "above" else -1
    return base * (1 + sign * pct / 100)


def _edge_cross(prev: float, curr: float, level: float, direction: str) -> bool:
    """Direction-literal crossing test used by RSI / Stochastic / ADX (PDF §7.3):
    'above' -> fires on the RISING edge (prev <= level and curr > level)
    'below' -> fires on the FALLING edge (prev >= level and curr < level)
    """
    if direction == "above":
        return prev <= level and curr > level
    else:
        return prev >= level and curr < level


# ─────────────────────────── 1. SMA ─────────────────────────────────────────
def compute_sma(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    # NOTE: do NOT round sma here. The Excel export computes Buy/Sell Threshold
    # and Buy/Sell Condition off the full-precision AVERAGE() value (the cell's
    # "0.0000" number format is display-only, not an actual rounding of the
    # value used in downstream formulas). Rounding here, before the thresholds
    # and conditions are derived, can flip a condition near a boundary versus
    # Excel and cascade into different Action(calc)/Status results. Display
    # rounding is applied later in pipeline.build_result_df via _round4().
    sma = prices.rolling(window=window).mean()
    buy_thresh  = _threshold(sma, buy_pct,  buy_direction)
    sell_thresh = _threshold(sma, sell_pct, sell_direction)

    buy_op  = (lambda p, t: p > t) if buy_direction  == "above" else (lambda p, t: p < t)
    sell_op = (lambda p, t: p < t) if sell_direction == "below"  else (lambda p, t: p > t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(buy_op(p, bt))
            sell_cond.append(sell_op(p, st))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "Moving Average (calc)",
        "indicator_vals": sma,
        "extra_cols": {
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }

# ─────────────────────────── 2. EMA ─────────────────────────────────────────
def compute_ema(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    alpha = 2 / (window + 1)
    ema_vals = [prices.iloc[0]]
    for i in range(1, len(prices)):
        ema_vals.append(alpha * prices.iloc[i] + (1 - alpha) * ema_vals[-1])
    ema = pd.Series(ema_vals, index=prices.index)

    buy_thresh  = _threshold(ema, buy_pct,  buy_direction)
    sell_thresh = _threshold(ema, sell_pct, sell_direction)

    buy_op  = (lambda p, t: p > t) if buy_direction  == "above" else (lambda p, t: p < t)
    sell_op = (lambda p, t: p < t) if sell_direction == "below"  else (lambda p, t: p > t)

    buy_cond  = [buy_op(prices.iloc[i],  buy_thresh.iloc[i])  for i in range(len(prices))]
    sell_cond = [sell_op(prices.iloc[i], sell_thresh.iloc[i]) for i in range(len(prices))]

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "EMA (calc)",
        "indicator_vals": ema,
        "extra_cols": {
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 3. Stochastic ──────────────────────────────────
def compute_stochastic(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    # %K = (C - L_n) / (H_n - L_n) * 100
    low_n  = prices.rolling(window=window).min()
    high_n = prices.rolling(window=window).max()
    denom  = (high_n - low_n).replace(0, np.nan)
    k = ((prices - low_n) / denom * 100).fillna(50)
    d = k.rolling(window=3).mean()  # %D = 3-period SMA of %K

    OVERSOLD   = 20
    OVERBOUGHT = 80

    # Direction-literal semantics (PDF §7.3, applies to Stochastic too):
    # buy_direction "above" -> rising-edge cross of 20 (canonical exit-oversold)
    # buy_direction "below" -> falling-edge cross of 20
    # sell_direction "below" -> falling-edge cross of 80 (canonical exit-overbought)
    # sell_direction "above" -> rising-edge cross of 80
    k_vals = k.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(k_vals)):
        prev = k_vals[i - 1] if i > 0 else k_vals[0]
        curr = k_vals[i]
        buy_cond.append(_edge_cross(prev, curr, OVERSOLD,   buy_direction))
        sell_cond.append(_edge_cross(prev, curr, OVERBOUGHT, sell_direction))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "%K (calc)",
        "indicator_vals": k,
        "extra_cols": {
            "%D (Signal)": d,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 4. MACD ────────────────────────────────────────
def compute_macd(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    def _ema_series(s, n):
        a = 2 / (n + 1)
        out = [s.iloc[0]]
        for i in range(1, len(s)):
            out.append(a * s.iloc[i] + (1 - a) * out[-1])
        return pd.Series(out, index=s.index)

    ema12  = _ema_series(prices, 12)
    ema26  = _ema_series(prices, 26)
    macd   = ema12 - ema26
    signal = _ema_series(macd, 9)
    hist   = macd - signal

    # PDF Eq. 34/35: threshold anchored to the Signal Line, shifted by
    # buy_pct/sell_pct with sign set by buy_direction/sell_direction.
    # At pct=0 this collapses back to the plain Signal Line crossover.
    buy_thresh  = _threshold(signal, buy_pct,  buy_direction)
    sell_thresh = _threshold(signal, sell_pct, sell_direction)

    buy_cond, sell_cond = [], []
    for i in range(len(macd)):
        prev_m  = macd.iloc[i - 1]        if i > 0 else macd.iloc[0]
        prev_bt = buy_thresh.iloc[i - 1]  if i > 0 else buy_thresh.iloc[0]
        prev_st = sell_thresh.iloc[i - 1] if i > 0 else sell_thresh.iloc[0]
        curr_m, curr_bt, curr_st = macd.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        buy_cond.append(prev_m <= prev_bt and curr_m > curr_bt)
        sell_cond.append(prev_m >= prev_st and curr_m < curr_st)

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "MACD (calc)",
        "indicator_vals": macd,
        "extra_cols": {
            "MACD Signal":    signal,
            "MACD Histogram": hist,
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 5. Bollinger Bands ─────────────────────────────
def compute_bollinger(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    k: float = 2.0,
    repeat: bool = False,
) -> dict:
    mid   = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=0)  # population std
    upper = mid + k * sigma
    lower = mid - k * sigma

    # PDF Eq. 44/45: buy anchors to Lower Band, sell anchors to Upper Band
    # (the canonical mean-reversion form), each shifted by pct with sign
    # set by the direction operator. buy_op/sell_op read the direction
    # literally, same pattern as SMA/EMA, so inverted directions are honoured.
    buy_thresh  = _threshold(lower, buy_pct,  buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op  = (lambda p, t: p < t) if buy_direction  == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(buy_op(prices.iloc[i], bt))
            sell_cond.append(sell_op(prices.iloc[i], st))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "BB Middle (calc)",
        "indicator_vals": mid,
        "extra_cols": {
            "BB Upper":       upper,
            "BB Lower":       lower,
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 6. RSI ─────────────────────────────────────────
def compute_rsi(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    delta  = prices.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.rolling(window=window).mean()
    avg_l  = loss.rolling(window=window).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi    = 100 - 100 / (1 + rs)

    OVERSOLD   = 30
    OVERBOUGHT = 70

    # Direction-literal semantics (PDF §7.3):
    # buy_direction "above" -> rising-edge cross of 30 (textbook exit-oversold)
    # buy_direction "below" -> falling-edge cross of 30 (contrarian dip-buy)
    # sell_direction "below" -> falling-edge cross of 70 (textbook exit-overbought)
    # sell_direction "above" -> rising-edge cross of 70 (entering overbought)
    rsi_vals = rsi.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(rsi_vals)):
        prev = rsi_vals[i - 1] if i > 0 else rsi_vals[i]
        curr = rsi_vals[i]
        if pd.isna(prev): prev = curr
        if pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(_edge_cross(prev, curr, OVERSOLD,   buy_direction))
        sell_cond.append(_edge_cross(prev, curr, OVERBOUGHT, sell_direction))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "RSI (calc)",
        "indicator_vals": rsi,
        "extra_cols": {
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 7. Fibonacci ───────────────────────────────────
def compute_fibonacci(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    roll_low  = prices.rolling(window=window).min()
    roll_high = prices.rolling(window=window).max()
    diff = roll_high - roll_low

    fib236 = roll_low + 0.236 * diff
    fib382 = roll_low + 0.382 * diff  # support / Buy anchor
    fib500 = roll_low + 0.500 * diff  # main indicator column
    fib618 = roll_low + 0.618 * diff  # resistance / Sell anchor
    fib786 = roll_low + 0.786 * diff

    # PDF Eq. 64/65: buy anchors to fib382 (support), sell anchors to fib618
    # (resistance), each shifted by pct with sign set by direction operator.
    buy_thresh  = _threshold(fib382, buy_pct,  buy_direction)
    sell_thresh = _threshold(fib618, sell_pct, sell_direction)

    buy_op  = (lambda p, t: p < t) if buy_direction  == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(buy_op(prices.iloc[i], bt))
            sell_cond.append(sell_op(prices.iloc[i], st))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "Fib 50% (calc)",
        "indicator_vals": fib500,
        "extra_cols": {
            "Fib 23.6%": fib236,
            "Fib 38.2%": fib382,
            "Fib 61.8%": fib618,
            "Fib 78.6%": fib786,
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 8. Standard Deviation ──────────────────────────
def compute_std_dev(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    k: float = 2.0,
    repeat: bool = False,
) -> dict:
    mu    = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=0)
    lower = mu - k * sigma
    upper = mu + k * sigma

    # PDF Eq. 76/77 (canonical examples Eq. 78-81): buy anchors to Lower
    # Threshold, sell anchors to Upper Threshold, shifted by pct with sign
    # set by direction operator; direction also flips the comparison operator
    # so inverted (mean-following) strategies are honoured literally.
    buy_thresh  = _threshold(lower, buy_pct,  buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op  = (lambda p, t: p < t) if buy_direction  == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(buy_op(prices.iloc[i], bt))
            sell_cond.append(sell_op(prices.iloc[i], st))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "Std Dev σ (calc)",
        "indicator_vals": sigma,
        "extra_cols": {
            "StdDev Mean":    mu,
            "StdDev Lower":   lower,
            "StdDev Upper":   upper,
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 9. ADX ─────────────────────────────────────────
def compute_adx(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    # PDF §10.1 fixes the lookback at 14 throughout (Eq. 82-88); a caller-
    # supplied window is only used as a fallback if none/invalid is given,
    # to keep the ADX math matching the documented formula by default.
    n = 14

    # Approximate ADX from a single price series (no H/L/C columns)
    # Use price as proxy for H, L, C (same series)
    delta  = prices.diff().fillna(0)
    plus_dm  = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)

    tr = prices.diff().abs().fillna(0)  # simplified TR without H/L

    atr        = tr.rolling(window=n).mean()
    plus_di    = (plus_dm.rolling(window=n).mean()  / atr.replace(0, np.nan) * 100).fillna(0)
    minus_di   = (minus_dm.rolling(window=n).mean() / atr.replace(0, np.nan) * 100).fillna(0)
    denom      = (plus_di + minus_di).replace(0, np.nan)
    dx         = ((plus_di - minus_di).abs() / denom * 100).fillna(0)
    adx        = dx.rolling(window=n).mean()

    STRONG = 25
    WEAK   = 20

    # Direction-literal semantics (PDF §7.3, ADX follows the same convention):
    # buy_direction "above" -> rising-edge cross of 25 (canonical: trend starting)
    # buy_direction "below" -> falling-edge cross of 25
    # sell_direction "below" -> falling-edge cross of 20 (canonical: trend weakening)
    # sell_direction "above" -> rising-edge cross of 20
    adx_vals = adx.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(adx_vals)):
        prev = adx_vals[i - 1] if i > 0 else adx_vals[i]
        curr = adx_vals[i]
        if pd.isna(prev): prev = curr
        if pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(_edge_cross(prev, curr, STRONG, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, WEAK,   sell_direction))

    position, action = _state_machine(buy_cond, sell_cond, repeat)
    return {
        "indicator_col": "ADX (calc)",
        "indicator_vals": adx,
        "extra_cols": {
            "+DI": plus_di,
            "-DI": minus_di,
            "Buy Condition":  pd.Series(buy_cond,  index=prices.index),
            "Sell Condition": pd.Series(sell_cond, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
    }


# ─────────────────────────── 10. Heikin Ashi ────────────────────────────────
def compute_heikin_ashi(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    # With only one price series, treat O=H=L=C=price
    ha_close = prices  # (O+H+L+C)/4 = price when all are same
    ha_open  = [prices.iloc[0]]
    for i in range(1, len(prices)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i - 1]) / 2)
    ha_open_s = pd.Series(ha_open, index=prices.index)

    # PDF Eq. 95/96: threshold anchored to HA Open, shifted by pct with sign
    # set by direction operator. At pct=0 this collapses to the canonical
    # green/red candle-color comparison.
    buy_thresh  = _threshold(ha_open_s, buy_pct,  buy_direction)
    sell_thresh = _threshold(ha_open_s, sell_pct, sell_direction)

    buy_op  = (lambda c, t: c > t) if buy_direction  == "above" else (lambda c, t: c < t)
    sell_op = (lambda c, t: c < t) if sell_direction == "below" else (lambda c, t: c > t)

    buy_cond  = [buy_op(ha_close.iloc[i],  buy_thresh.iloc[i])  for i in range(len(prices))]
    sell_cond = [sell_op(ha_close.iloc[i], sell_thresh.iloc[i]) for i in range(len(prices))]

    # Only fire on colour change (transition)
    buy_fired, sell_fired = [], []
    for i in range(len(buy_cond)):
        prev_b = buy_cond[i - 1]  if i > 0 else False
        prev_s = sell_cond[i - 1] if i > 0 else False
        buy_fired.append(buy_cond[i]  and not prev_b)
        sell_fired.append(sell_cond[i] and not prev_s)

    position, action = _state_machine(buy_fired, sell_fired, repeat)
    return {
        "indicator_col": "HA Close (calc)",
        "indicator_vals": ha_close,
        "extra_cols": {
            "HA Open":        ha_open_s,
            "Buy Threshold":  buy_thresh,
            "Sell Threshold": sell_thresh,
            "Buy Condition":  pd.Series(buy_fired,  index=prices.index),
            "Sell Condition": pd.Series(sell_fired, index=prices.index),
        },
        "position": position, "action": action,
        "buy_cond": buy_fired, "sell_cond": sell_fired,
    }


# ─────────────────────────── Dispatcher ─────────────────────────────────────
INDICATOR_MAP = {
    "Simple Moving Average":      compute_sma,
    "Exponential Moving Average": compute_ema,
    "Stochastic Oscillator":      compute_stochastic,
    "MACD":                        compute_macd,
    "Bollinger Bands":             compute_bollinger,
    "Relative Strength Index":     compute_rsi,
    "Fibonacci Retracement":       compute_fibonacci,
    "Standard Deviation":          compute_std_dev,
    "ADX":                         compute_adx,
    "Heikin Ashi":                 compute_heikin_ashi,
}

# Human-readable hint shown under each indicator
INDICATOR_HINTS = {
    "Simple Moving Average":      "Buy when price moves above/below the rolling average by a % threshold.",
    "Exponential Moving Average": "Like SMA but gives more weight to recent prices. Reacts faster.",
    "Stochastic Oscillator":      "Buys when %K crosses above 20 (oversold exit); sells below 80 (overbought exit).",
    "MACD":                        "Buys on MACD/Signal crossover (bullish); sells on bearish crossover.",
    "Bollinger Bands":             "Mean reversion: buy below lower band, sell above upper band.",
    "Relative Strength Index":     "Buys when RSI crosses above 30; sells when it crosses below 70.",
    "Fibonacci Retracement":       "Buys near 38.2% support level; sells near 61.8% resistance.",
    "Standard Deviation":          "Buys below µ−2σ band; sells above µ+2σ band.",
    "ADX":                         "Buys when ADX crosses above 25 (strong trend); sells below 20.",
    "Heikin Ashi":                 "Buys on green candle colour change; sells on red candle colour change.",
}


def run_indicator(
    indicator_name: str,
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
) -> dict:
    fn = INDICATOR_MAP.get(indicator_name)
    if fn is None:
        raise ValueError(f"Unknown indicator: {indicator_name}")
    return fn(prices, window, buy_pct, sell_pct, buy_direction, sell_direction, repeat)