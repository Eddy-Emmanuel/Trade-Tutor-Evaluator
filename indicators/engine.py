"""
indicators/engine.py
All 10 indicator engines — formulas taken from the indicator PDF.

Each compute_* function returns a dict with:
  - 'indicator_col' : str  (column name for the main indicator value)
  - 'indicator_vals': pd.Series
  - 'extra_cols'    : dict (name -> pd.Series) of any additional columns
  - 'position'      : list[str]
  - 'action'        : list[str]
  - 'buy_cond'      : list[bool]
  - 'sell_cond'     : list[bool]
  - 'warmup'        : int   (leading rows that cannot produce a signal)

QA round 2 — what changed
-------------------------
1. WARM-UP. No indicator fabricates a value to fill a warm-up gap any more.
   Stochastic's `.fillna(50)` and ADX's five `.fillna(0)` calls are gone.
   Every engine now computes `warmup` — the number of leading rows where the
   indicator is not yet defined — and forces buy_cond/sell_cond False there,
   so Position stays "Out" and Action stays "Hold" until the lookback is
   genuinely satisfied. This is the SMA behaviour (a 20-period MA holds for
   the first 20 rows) applied uniformly.

   Crossover indicators (RSI, Stochastic, ADX, MACD) need TWO consecutive
   real values to detect an edge, so their warm-up is one row longer than
   the point where the series first becomes valid. Previously the loops did
   `prev = curr` on the first row, which invented an edge out of nothing.

2. THRESHOLDS vs "How Much (%)". RSI, Stochastic and ADX never read buy_pct /
   sell_pct — they compare against level constants. The sidebar's "How Much
   (%)" was therefore inert for those three. Their levels are now first-class
   buy/sell fields (0-100) instead, matching TT.

3. WINDOW. MACD, ADX and Heikin Ashi ignore the generic `window` argument.
   MACD and ADX have their own named period fields; Heikin Ashi has no period
   at all. INDICATOR_SPEC declares this so the sidebar stops showing a control
   that does nothing.

Threshold constants in the PDF are illustrative defaults, not fixed values —
all of them are overridable here.
"""
import numpy as np
import pandas as pd


# ─────────────────────────── helpers ────────────────────────────────────────
def _state_machine(buy_cond: list, sell_cond: list, repeat: bool = False) -> tuple[list, list]:
    """Universal state machine. Returns (position, action).

    repeat=False (default): alternates Buy -> Sell -> Buy -> Sell.
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
    return prev >= level and curr < level


def _ema_series(s: pd.Series, n: int) -> pd.Series:
    """Recursive EMA seeded on the first observation (PDF Eq. 17).

    The recursion emits a number from row 0 onward, so it carries no NaN to
    mark its own warm-up. Callers must apply an explicit warm-up of n-1 rows.
    """
    a = 2 / (n + 1)
    out = [s.iloc[0]]
    for i in range(1, len(s)):
        out.append(a * s.iloc[i] + (1 - a) * out[-1])
    return pd.Series(out, index=s.index)


def _first_valid_pos(s: pd.Series) -> int:
    """Positional index of the first non-NaN value; len(s) if there is none."""
    idx = s.first_valid_index()
    return len(s) if idx is None else int(s.index.get_loc(idx))


def _blank_warmup(buy_cond: list, sell_cond: list, warmup: int) -> tuple[list, list]:
    """Force no signal for the first `warmup` rows.

    The lookback is not satisfied yet, so any condition computed there is an
    artefact of seeding or of a placeholder fill, never a measurement.
    """
    n = min(max(warmup, 0), len(buy_cond))
    for i in range(n):
        buy_cond[i] = False
        sell_cond[i] = False
    return buy_cond, sell_cond


def _finish(
    indicator_col: str,
    indicator_vals: pd.Series,
    extra_cols: dict,
    buy_cond: list,
    sell_cond: list,
    warmup: int,
    repeat: bool,
    index,
) -> dict:
    """Apply warm-up suppression, run the state machine, assemble the result."""
    buy_cond, sell_cond = _blank_warmup(buy_cond, sell_cond, warmup)
    position, action = _state_machine(buy_cond, sell_cond, repeat)

    extra_cols = dict(extra_cols)
    extra_cols["Buy Condition"] = pd.Series(buy_cond, index=index)
    extra_cols["Sell Condition"] = pd.Series(sell_cond, index=index)

    return {
        "indicator_col": indicator_col,
        "indicator_vals": indicator_vals,
        "extra_cols": extra_cols,
        "position": position,
        "action": action,
        "buy_cond": buy_cond,
        "sell_cond": sell_cond,
        "warmup": int(min(max(warmup, 0), len(buy_cond))),
    }


# ─────────────────────────── 1. SMA ─────────────────────────────────────────
def compute_sma(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    **_ignored,
) -> dict:
    # NOTE: do NOT round sma here. The Excel export computes Buy/Sell Threshold
    # and Buy/Sell Condition off the full-precision AVERAGE() value; rounding
    # before the thresholds are derived can flip a condition near a boundary.
    sma = prices.rolling(window=window).mean()
    warmup = _first_valid_pos(sma)          # rolling() already marks its own gap

    buy_thresh = _threshold(sma, buy_pct, buy_direction)
    sell_thresh = _threshold(sma, sell_pct, sell_direction)

    buy_op = (lambda p, t: p > t) if buy_direction == "above" else (lambda p, t: p < t)
    sell_op = (lambda p, t: p < t) if sell_direction == "below" else (lambda p, t: p > t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    return _finish(
        "Moving Average (calc)", sma,
        {"Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 2. EMA ─────────────────────────────────────────
def compute_ema(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    **_ignored,
) -> dict:
    ema = _ema_series(prices, window)
    # The recursion is seeded from row 0, so it produces numbers immediately.
    # Hold for the same span an SMA of this window would need.
    warmup = max(int(window) - 1, 0)

    buy_thresh = _threshold(ema, buy_pct, buy_direction)
    sell_thresh = _threshold(ema, sell_pct, sell_direction)

    buy_op = (lambda p, t: p > t) if buy_direction == "above" else (lambda p, t: p < t)
    sell_op = (lambda p, t: p < t) if sell_direction == "below" else (lambda p, t: p > t)

    buy_cond = [bool(buy_op(prices.iloc[i], buy_thresh.iloc[i])) for i in range(len(prices))]
    sell_cond = [bool(sell_op(prices.iloc[i], sell_thresh.iloc[i])) for i in range(len(prices))]

    return _finish(
        "EMA (calc)", ema,
        {"Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 3. Stochastic ──────────────────────────────────
def compute_stochastic(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    oversold: float = 20.0,
    overbought: float = 80.0,
    d_window: int = 3,
    **_ignored,
) -> dict:
    # %K = (C - L_n) / (H_n - L_n) * 100
    low_n = prices.rolling(window=window).min()
    high_n = prices.rolling(window=window).max()
    denom = (high_n - low_n).replace(0, np.nan)

    # No .fillna(50). During warm-up %K is undefined and stays NaN; a flat
    # window (high == low) is also genuinely undefined rather than "neutral".
    k = (prices - low_n) / denom * 100
    d = k.rolling(window=int(d_window)).mean()

    # A crossover needs a valid previous AND current value, so the first row
    # that can carry a signal is one past where %K becomes valid.
    warmup = _first_valid_pos(k) + 1

    k_vals = k.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(k_vals)):
        prev = k_vals[i - 1] if i > 0 else np.nan
        curr = k_vals[i]
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(_edge_cross(prev, curr, oversold, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, overbought, sell_direction))

    return _finish(
        "%K (calc)", k, {"%D (Signal)": d},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 4. MACD ────────────────────────────────────────
def compute_macd(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
    **_ignored,
) -> dict:
    ema_fast = _ema_series(prices, int(fast_window))
    ema_slow = _ema_series(prices, int(slow_window))
    macd = ema_fast - ema_slow
    signal = _ema_series(macd, int(signal_window))
    hist = macd - signal

    # Slow EMA must mature, then the signal EMA must mature on top of it,
    # then one more row so a crossover has a real predecessor.
    warmup = max(int(slow_window) - 1, 0) + max(int(signal_window) - 1, 0) + 1

    # PDF Eq. 34/35: threshold anchored to the Signal Line, shifted by
    # buy_pct/sell_pct. At pct=0 this is the plain Signal Line crossover.
    buy_thresh = _threshold(signal, buy_pct, buy_direction)
    sell_thresh = _threshold(signal, sell_pct, sell_direction)

    buy_cond, sell_cond = [], []
    for i in range(len(macd)):
        if i == 0:
            buy_cond.append(False); sell_cond.append(False); continue
        prev_m, prev_bt, prev_st = macd.iloc[i - 1], buy_thresh.iloc[i - 1], sell_thresh.iloc[i - 1]
        curr_m, curr_bt, curr_st = macd.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(prev_m) or pd.isna(curr_m):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(bool(prev_m <= prev_bt and curr_m > curr_bt))
        sell_cond.append(bool(prev_m >= prev_st and curr_m < curr_st))

    return _finish(
        "MACD (calc)", macd,
        {"MACD Signal": signal, "MACD Histogram": hist,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 5. Bollinger Bands ─────────────────────────────
def compute_bollinger(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    k: float = 2.0,
    ddof: int = 0,
    **_ignored,
) -> dict:
    mid = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=int(ddof))
    upper = mid + k * sigma
    lower = mid - k * sigma
    warmup = _first_valid_pos(mid)

    # PDF Eq. 44/45: buy anchors to Lower Band, sell anchors to Upper Band.
    buy_thresh = _threshold(lower, buy_pct, buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op = (lambda p, t: p < t) if buy_direction == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(bool(buy_op(prices.iloc[i], bt)))
            sell_cond.append(bool(sell_op(prices.iloc[i], st)))

    return _finish(
        "BB Middle (calc)", mid,
        {"BB Upper": upper, "BB Lower": lower,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 6. RSI ─────────────────────────────────────────
def compute_rsi(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    oversold: float = 30.0,
    overbought: float = 70.0,
    **_ignored,
) -> dict:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.rolling(window=window).mean()
    avg_l = loss.rolling(window=window).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # diff() costs one row, the rolling mean costs `window`, and the crossover
    # needs a valid predecessor.
    warmup = _first_valid_pos(rsi) + 1

    rsi_vals = rsi.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(rsi_vals)):
        prev = rsi_vals[i - 1] if i > 0 else np.nan
        curr = rsi_vals[i]
        # Previously: `if pd.isna(prev): prev = curr`, which manufactured a
        # non-edge on row 0 and after any gap. Now it simply cannot signal.
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(_edge_cross(prev, curr, oversold, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, overbought, sell_direction))

    return _finish(
        "RSI (calc)", rsi, {},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 7. Fibonacci ───────────────────────────────────
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]


def compute_fibonacci(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    buy_level: float = 0.382,
    sell_level: float = 0.618,
    **_ignored,
) -> dict:
    roll_low = prices.rolling(window=window).min()
    roll_high = prices.rolling(window=window).max()
    diff = roll_high - roll_low
    warmup = _first_valid_pos(diff)

    levels = {lv: roll_low + lv * diff for lv in sorted(set(FIB_LEVELS + [buy_level, sell_level]))}

    # PDF Eq. 64/65: buy anchors to a support level, sell to a resistance level.
    buy_thresh = _threshold(levels[buy_level], buy_pct, buy_direction)
    sell_thresh = _threshold(levels[sell_level], sell_pct, sell_direction)

    buy_op = (lambda p, t: p < t) if buy_direction == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(bool(buy_op(prices.iloc[i], bt)))
            sell_cond.append(bool(sell_op(prices.iloc[i], st)))

    extra = {f"Fib {lv * 100:.1f}%": s for lv, s in levels.items() if lv != 0.500}
    extra["Buy Threshold"] = buy_thresh
    extra["Sell Threshold"] = sell_thresh

    return _finish(
        "Fib 50% (calc)", levels.get(0.500, roll_low + 0.5 * diff), extra,
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 8. Standard Deviation ──────────────────────────
def compute_std_dev(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    k: float = 2.0,
    ddof: int = 0,
    **_ignored,
) -> dict:
    mu = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=int(ddof))
    lower = mu - k * sigma
    upper = mu + k * sigma
    warmup = _first_valid_pos(sigma)

    # PDF Eq. 76/77: buy anchors to the Lower Threshold, sell to the Upper.
    buy_thresh = _threshold(lower, buy_pct, buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op = (lambda p, t: p < t) if buy_direction == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        bt, st = buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False)
        else:
            buy_cond.append(bool(buy_op(prices.iloc[i], bt)))
            sell_cond.append(bool(sell_op(prices.iloc[i], st)))

    return _finish(
        "Std Dev σ (calc)", sigma,
        {"StdDev Mean": mu, "StdDev Lower": lower, "StdDev Upper": upper,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 9. ADX ─────────────────────────────────────────
def compute_adx(
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    adx_window: int | None = None,
    strong: float = 25.0,
    weak: float = 20.0,
    **_ignored,
) -> dict:
    # PDF §10.1 documents a 14-period lookback (Eq. 82-88) as the default.
    # `window` is deliberately unread — ADX has its own named period field.
    n = int(adx_window) if adx_window else 14

    # Approximate ADX from a single price series (no H/L/C columns):
    # price acts as proxy for H, L and C.
    # No .fillna(0) anywhere below — a zero DI/DX during warm-up is a fake
    # measurement that _edge_cross would happily treat as a real crossing.
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)

    tr = prices.diff().abs()

    atr = tr.rolling(window=n).mean()
    plus_di = plus_dm.rolling(window=n).mean() / atr.replace(0, np.nan) * 100
    minus_di = minus_dm.rolling(window=n).mean() / atr.replace(0, np.nan) * 100
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / denom * 100
    adx = dx.rolling(window=n).mean()

    warmup = _first_valid_pos(adx) + 1

    adx_vals = adx.tolist()
    buy_cond, sell_cond = [], []
    for i in range(len(adx_vals)):
        prev = adx_vals[i - 1] if i > 0 else np.nan
        curr = adx_vals[i]
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); continue
        buy_cond.append(_edge_cross(prev, curr, strong, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, weak, sell_direction))

    return _finish(
        "ADX (calc)", adx, {"+DI": plus_di, "-DI": minus_di},
        buy_cond, sell_cond, warmup, repeat, prices.index,
    )


# ─────────────────────────── 10. Heikin Ashi ────────────────────────────────
def compute_heikin_ashi(
    prices: pd.Series,
    window: int = 1,
    buy_pct: float = 0.0,
    sell_pct: float = 0.0,
    buy_direction: str = "above",
    sell_direction: str = "below",
    repeat: bool = False,
    transition_only: bool = True,
    **_ignored,
) -> dict:
    # With only one price series, treat O=H=L=C=price. `window` is unread —
    # Heikin Ashi has no lookback period, so the sidebar hides that field.
    ha_close = prices  # (O+H+L+C)/4 = price when all are the same
    ha_open = [prices.iloc[0]]
    for i in range(1, len(prices)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i - 1]) / 2)
    ha_open_s = pd.Series(ha_open, index=prices.index)

    # Row 0's HA Open is a seed, not a computed candle, so it cannot signal.
    warmup = 1

    # PDF Eq. 95/96: threshold anchored to HA Open. At pct=0 this collapses
    # to the canonical green/red candle-colour comparison.
    buy_thresh = _threshold(ha_open_s, buy_pct, buy_direction)
    sell_thresh = _threshold(ha_open_s, sell_pct, sell_direction)

    buy_op = (lambda c, t: c > t) if buy_direction == "above" else (lambda c, t: c < t)
    sell_op = (lambda c, t: c < t) if sell_direction == "below" else (lambda c, t: c > t)

    buy_cond = [bool(buy_op(ha_close.iloc[i], buy_thresh.iloc[i])) for i in range(len(prices))]
    sell_cond = [bool(sell_op(ha_close.iloc[i], sell_thresh.iloc[i])) for i in range(len(prices))]

    if transition_only:
        # Fire only on colour change, not on every bar of a run.
        buy_fired, sell_fired = [], []
        for i in range(len(buy_cond)):
            prev_b = buy_cond[i - 1] if i > 0 else False
            prev_s = sell_cond[i - 1] if i > 0 else False
            buy_fired.append(buy_cond[i] and not prev_b)
            sell_fired.append(sell_cond[i] and not prev_s)
    else:
        buy_fired, sell_fired = buy_cond, sell_cond

    return _finish(
        "HA Close (calc)", ha_close,
        {"HA Open": ha_open_s, "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_fired, sell_fired, warmup, repeat, prices.index,
    )


# ─────────────────────────── Dispatcher ─────────────────────────────────────
INDICATOR_MAP = {
    "Simple Moving Average":      compute_sma,
    "Exponential Moving Average": compute_ema,
    "Stochastic Oscillator":      compute_stochastic,
    "MACD":                       compute_macd,
    "Bollinger Bands":            compute_bollinger,
    "Relative Strength Index":    compute_rsi,
    "Fibonacci Retracement":      compute_fibonacci,
    "Standard Deviation":         compute_std_dev,
    "ADX":                        compute_adx,
    "Heikin Ashi":                compute_heikin_ashi,
}

INDICATOR_HINTS = {
    "Simple Moving Average":      "Buy when price moves above/below the rolling average by a % threshold.",
    "Exponential Moving Average": "Like SMA but gives more weight to recent prices. Reacts faster.",
    "Stochastic Oscillator":      "Buys when %K crosses the buy threshold; sells when it crosses the sell threshold.",
    "MACD":                       "Buys on MACD/Signal crossover (bullish); sells on bearish crossover.",
    "Bollinger Bands":            "Mean reversion: buy below the lower band, sell above the upper band.",
    "Relative Strength Index":    "Buys when RSI crosses the buy threshold; sells when it crosses the sell threshold.",
    "Fibonacci Retracement":      "Buys near the chosen support level; sells near the chosen resistance level.",
    "Standard Deviation":         "Buys below µ−kσ; sells above µ+kσ.",
    "ADX":                        "Buys when ADX crosses the strong-trend level; sells below the weak level.",
    "Heikin Ashi":                "Buys on green candle colour change; sells on red candle colour change.",
}


# ── Per-indicator field schema — single source of truth for the sidebar ─────
#
# window   : the generic period control, or None when the indicator has no
#            lookback (Heikin Ashi) or supplies its own named ones (MACD, ADX).
# uses_pct : whether "How Much (%)" reaches the maths. False for the three
#            crossover indicators, which compare against threshold levels.
# fields   : extra controls. slot "buy"/"sell" places them in the buy or sell
#            leg (where "How Much (%)" would otherwise sit); "shared" places
#            them under the period control.
#
# Every threshold default below is the PDF's illustrative value, not a fixed
# constant — all are user-overridable, matching TT.
INDICATOR_SPEC: dict[str, dict] = {
    "Simple Moving Average": {
        "window": {"label": "Window (Periods)", "default": 2, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [],
    },
    "Exponential Moving Average": {
        "window": {"label": "Window (Periods)", "default": 2, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [],
    },
    "Stochastic Oscillator": {
        "window": {
            "label": "Lookback Period", "default": 14, "min": 2, "max": 500,
            "help": "Periods used for the %K high/low range. The PDF uses 14. "
                    "Very short lookbacks make %K jump between 0 and 100 and "
                    "produce crossings that carry little information.",
        },
        "uses_pct": False,
        "fields": [
            {"key": "d_window", "label": "%D Period", "type": "int", "slot": "shared",
             "default": 3, "min": 1, "max": 50,
             "help": "Periods used to smooth %K into the %D signal line."},
            {"key": "oversold", "label": "Buy Threshold (%K)", "type": "float", "slot": "buy",
             "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0},
            {"key": "overbought", "label": "Sell Threshold (%K)", "type": "float", "slot": "sell",
             "default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0},
        ],
    },
    "MACD": {
        "window": None,
        "uses_pct": True,
        "fields": [
            {"key": "fast_window", "label": "Fast EMA Period", "type": "int", "slot": "shared",
             "default": 12, "min": 1, "max": 200},
            {"key": "slow_window", "label": "Slow EMA Period", "type": "int", "slot": "shared",
             "default": 26, "min": 2, "max": 400},
            {"key": "signal_window", "label": "Signal EMA Period", "type": "int", "slot": "shared",
             "default": 9, "min": 1, "max": 200},
        ],
    },
    "Bollinger Bands": {
        "window": {"label": "Window (Periods)", "default": 20, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [
            {"key": "k", "label": "Band Width (k × σ)", "type": "float", "slot": "shared",
             "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
            {"key": "ddof", "label": "σ Denominator", "type": "select", "slot": "shared",
             "default": 0, "options": [("Population (n)", 0), ("Sample (n-1)", 1)],
             "help": "Excel STDEV.P → population; STDEV.S → sample. Must match the workbook."},
        ],
    },
    "Relative Strength Index": {
        "window": {
            "label": "Lookback Period", "default": 14, "min": 2, "max": 500,
            "help": "Periods averaged for gains and losses. The PDF uses 14.",
        },
        "uses_pct": False,
        "fields": [
            {"key": "oversold", "label": "Buy Threshold (RSI)", "type": "float", "slot": "buy",
             "default": 30.0, "min": 0.0, "max": 100.0, "step": 1.0},
            {"key": "overbought", "label": "Sell Threshold (RSI)", "type": "float", "slot": "sell",
             "default": 70.0, "min": 0.0, "max": 100.0, "step": 1.0},
        ],
    },
    "Fibonacci Retracement": {
        "window": {"label": "Window (Periods)", "default": 20, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [
            {"key": "buy_level", "label": "Buy Anchor Level", "type": "select", "slot": "buy",
             "default": 0.382,
             "options": [("23.6%", 0.236), ("38.2%", 0.382), ("50.0%", 0.500),
                         ("61.8%", 0.618), ("78.6%", 0.786)]},
            {"key": "sell_level", "label": "Sell Anchor Level", "type": "select", "slot": "sell",
             "default": 0.618,
             "options": [("23.6%", 0.236), ("38.2%", 0.382), ("50.0%", 0.500),
                         ("61.8%", 0.618), ("78.6%", 0.786)]},
        ],
    },
    "Standard Deviation": {
        "window": {"label": "Window (Periods)", "default": 20, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [
            {"key": "k", "label": "Band Width (k × σ)", "type": "float", "slot": "shared",
             "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
            {"key": "ddof", "label": "σ Denominator", "type": "select", "slot": "shared",
             "default": 0, "options": [("Population (n)", 0), ("Sample (n-1)", 1)]},
        ],
    },
    "ADX": {
        "window": None,
        "uses_pct": False,
        "fields": [
            {"key": "adx_window", "label": "ADX Period", "type": "int", "slot": "shared",
             "default": 14, "min": 2, "max": 200,
             "help": "PDF §10.1 uses 14."},
            {"key": "strong", "label": "Buy Threshold (ADX)", "type": "float", "slot": "buy",
             "default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0},
            {"key": "weak", "label": "Sell Threshold (ADX)", "type": "float", "slot": "sell",
             "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0},
        ],
    },
    "Heikin Ashi": {
        "window": None,
        "uses_pct": True,
        "fields": [
            {"key": "transition_only", "label": "Signal on colour change only",
             "type": "bool", "slot": "shared", "default": True,
             "help": "Off = every candle matching the condition fires, not just the first of a run."},
        ],
    },
}

# Back-compat for callers that imported the flat list (e.g. the FastAPI layer).
INDICATOR_PARAMS: dict[str, list[dict]] = {
    name: spec["fields"] for name, spec in INDICATOR_SPEC.items()
}


def spec_for(indicator_name: str) -> dict:
    return INDICATOR_SPEC.get(indicator_name, {"window": None, "uses_pct": True, "fields": []})


def uses_window(indicator_name: str) -> bool:
    return spec_for(indicator_name)["window"] is not None


def uses_pct(indicator_name: str) -> bool:
    return bool(spec_for(indicator_name)["uses_pct"])


def default_params(indicator_name: str) -> dict:
    return {f["key"]: f["default"] for f in spec_for(indicator_name)["fields"]}


def run_indicator(
    indicator_name: str,
    prices: pd.Series,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat: bool = False,
    params: dict | None = None,
) -> dict:
    fn = INDICATOR_MAP.get(indicator_name)
    if fn is None:
        raise ValueError(f"Unknown indicator: {indicator_name}")

    extras = default_params(indicator_name)
    extras.update(params or {})

    # Indicators that don't read pct never see a stale value from the form.
    if not uses_pct(indicator_name):
        buy_pct = sell_pct = 0.0

    return fn(
        prices=prices,
        window=window,
        buy_pct=buy_pct,
        sell_pct=sell_pct,
        buy_direction=buy_direction,
        sell_direction=sell_direction,
        repeat=repeat,
        **extras,
    )