"""
indicators/engine.py
All 10 indicator engines — formulas taken from Algorithmic_Indicators_v3.

Each compute_* function returns a dict with:
  - 'indicator_col'  : str  (column name for the main indicator value)
  - 'indicator_vals' : pd.Series
  - 'extra_cols'     : dict (name -> pd.Series) of any additional columns
  - 'position'       : list[str]
  - 'action'         : list[str]
  - 'buy_cond'       : list[bool]
  - 'sell_cond'      : list[bool]
  - 'warmup'         : int   (leading rows that cannot produce a signal)
  - 'undefined_bars' : int   (bars where at least one comparison operand was NaN)

v3 changes vs QA round 2
-------------------------
1. MACD THRESHOLD (§5.3 Eq. 34/35). Was multiplicative (Signal × (1 ± pct/100)).
   Now additive, volatility-scaled:
     buy_thresh  = Signal + buy_sign  × (buy_pct/100)  × σ_hist(20, ddof=0)
     sell_thresh = Signal + sell_sign × (sell_pct/100) × σ_hist(20, ddof=0)
   σ_hist is the 20-bar rolling population std-dev of (MACD − Signal).
   At pct=0 both thresholds equal the Signal line — canonical crossover.
   The Histogram column is computed internally but hidden from output (§5.1).

2. HEIKIN ASHI — level mode (§1.3, §11.3). The transition_only flag is
   removed. HA fires on every qualifying bar (Green → Buy condition; Red →
   Sell condition), not only on the first of a run. The state machine still
   enforces alternating Buy/Sell when repeat=False.

3. RSI DEGENERATE INPUTS (§1.5).
     Avg Loss = 0 AND Avg Gain > 0  →  RSI = 100
     Avg Loss = 0 AND Avg Gain = 0  →  RSI = 50
   Previously these emitted NaN (pandas 0/0 → NaN).

4. ADX DEGENERATE INPUTS (§1.5).
   ATR = 0  →  +DI = −DI = 0, DX = 0 (reads "no trend"), rather than NaN.

5. UNDEFINED BAR COUNTING (§1.5). Every engine accumulates `undefined_bars`:
   any bar where a comparison operand was NaN. Returned in the result dict
   so the UI can report "N/M bars had an undefined operand".

6. HARD MINIMUMS AND DEFAULTS (§1.3).
   - SMA / EMA default window: 20 (was 2).
   - Stochastic min window: 5.
   - Bollinger / StdDev min window: 6 at k=2 (k ≥ √(n−1) warns, not hard-stops).
   - MACD fast must be strictly < slow (validated in run_indicator).
   - How Much (%) default: 0.0 (was 0.05).

7. ddof SELECTOR REMOVED (§6.1, §9.1). Both Bollinger and StdDev enforce
   ddof=0 (population σ). The Sample (n−1) option is gone.

8. SELL QUANTITY LOCKED TO BUY QUANTITY (§12).

Threshold constants in the PDF are illustrative defaults, not fixed values —
all of them are overridable.
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
    undefined_bars: int = 0,
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
        "undefined_bars": int(undefined_bars),
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
    undef = 0
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st) or pd.isna(p):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    return _finish(
        "Moving Average (calc)", sma,
        {"Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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

    undef = 0
    buy_cond, sell_cond = [], []
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(p) or pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    return _finish(
        "EMA (calc)", ema,
        {"Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    undef = 0
    for i in range(len(k_vals)):
        prev = k_vals[i - 1] if i > 0 else np.nan
        curr = k_vals[i]
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); undef += 1; continue
        buy_cond.append(_edge_cross(prev, curr, oversold, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, overbought, sell_direction))

    return _finish(
        "%K (calc)", k, {"%D (Signal)": d},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    # Histogram computed internally for σ scaling; hidden from output (§5.1).
    hist = macd - signal

    # Slow EMA must mature, then the signal EMA must mature on top of it,
    # then one more row so a crossover has a real predecessor.
    warmup = max(int(slow_window) - 1, 0) + max(int(signal_window) - 1, 0) + 1

    # §5.3 Eq. 34/35: additive, volatility-scaled threshold.
    # σ_hist = rolling population std-dev of histogram over 20 bars (ddof=0).
    # At pct=0 the buffer is zero, thresholds equal the Signal line exactly —
    # canonical MACD/Signal crossover with no warm-up penalty.
    HIST_WINDOW = 20
    sigma_hist = hist.rolling(window=HIST_WINDOW).std(ddof=0)

    buy_sign  = +1 if buy_direction  == "above" else -1
    sell_sign = +1 if sell_direction == "above" else -1

    buy_thresh  = signal + buy_sign  * (buy_pct  / 100) * sigma_hist
    sell_thresh = signal + sell_sign * (sell_pct / 100) * sigma_hist

    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(macd)):
        if i == 0:
            buy_cond.append(False); sell_cond.append(False); continue
        prev_m  = macd.iloc[i - 1]
        curr_m  = macd.iloc[i]
        prev_bt = buy_thresh.iloc[i - 1]
        prev_st = sell_thresh.iloc[i - 1]
        curr_bt = buy_thresh.iloc[i]
        curr_st = sell_thresh.iloc[i]
        if pd.isna(prev_m) or pd.isna(curr_m) or pd.isna(prev_bt) or pd.isna(curr_bt):
            buy_cond.append(False); sell_cond.append(False); undef += 1; continue
        buy_cond.append(bool(prev_m <= prev_bt and curr_m > curr_bt))
        sell_cond.append(bool(prev_m >= prev_st and curr_m < curr_st))

    return _finish(
        "MACD (calc)", macd,
        # Histogram hidden (True) per §5.1; Buy/Sell Threshold visible for QA.
        {"MACD Signal": signal, "MACD Histogram (hidden)": hist,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    **_ignored,
) -> dict:
    # §6.1: population σ (ddof=0) is the spec-mandated denominator. The Sample
    # (n-1) option has been removed from the UI; ddof is hardcoded here.
    mid = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=0)
    upper = mid + k * sigma
    lower = mid - k * sigma
    warmup = _first_valid_pos(mid)

    # §6.3 Eq. 44/45: buy anchors to Lower Band, sell anchors to Upper Band.
    buy_thresh = _threshold(lower, buy_pct, buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op = (lambda p, t: p < t) if buy_direction == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st) or pd.isna(p):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    return _finish(
        "BB Middle (calc)", mid,
        {"BB Upper": upper, "BB Lower": lower,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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

    # §1.5 degenerate inputs — applied after rolling means are computed so
    # NaN (warm-up) rows are not touched.
    #   Avg Loss = 0 AND Avg Gain > 0  →  RSI = 100 (maximally overbought)
    #   Avg Loss = 0 AND Avg Gain = 0  →  RSI = 50  (neutral, no signal)
    rsi = pd.Series(np.nan, index=prices.index)
    valid = avg_g.notna() & avg_l.notna()
    both_zero  = valid & (avg_l == 0) & (avg_g == 0)
    loss_zero  = valid & (avg_l == 0) & (avg_g > 0)
    normal     = valid & (avg_l > 0)
    rsi[both_zero] = 50.0
    rsi[loss_zero] = 100.0
    rs = avg_g[normal] / avg_l[normal]
    rsi[normal] = 100 - 100 / (1 + rs)

    # diff() costs one row, the rolling mean costs `window`, and the crossover
    # needs a valid predecessor.
    warmup = _first_valid_pos(rsi) + 1

    rsi_vals = rsi.tolist()
    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(rsi_vals)):
        prev = rsi_vals[i - 1] if i > 0 else np.nan
        curr = rsi_vals[i]
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); undef += 1; continue
        buy_cond.append(_edge_cross(prev, curr, oversold, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, overbought, sell_direction))

    return _finish(
        "RSI (calc)", rsi, {},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    undef = 0
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st) or pd.isna(p):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    extra = {f"Fib {lv * 100:.1f}%": s for lv, s in levels.items() if lv != 0.500}
    extra["Buy Threshold"] = buy_thresh
    extra["Sell Threshold"] = sell_thresh

    return _finish(
        "Fib 50% (calc)", levels.get(0.500, roll_low + 0.5 * diff), extra,
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    **_ignored,
) -> dict:
    # §9.1: population σ (ddof=0) enforced. The Sample (n-1) option is removed.
    mu = prices.rolling(window=window).mean()
    sigma = prices.rolling(window=window).std(ddof=0)
    lower = mu - k * sigma
    upper = mu + k * sigma
    warmup = _first_valid_pos(sigma)

    # §9.3 Eq. 76/77: buy anchors to the Lower Threshold, sell to the Upper.
    buy_thresh = _threshold(lower, buy_pct, buy_direction)
    sell_thresh = _threshold(upper, sell_pct, sell_direction)

    buy_op = (lambda p, t: p < t) if buy_direction == "below" else (lambda p, t: p > t)
    sell_op = (lambda p, t: p > t) if sell_direction == "above" else (lambda p, t: p < t)

    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(prices)):
        p, bt, st = prices.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(bt) or pd.isna(st) or pd.isna(p):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(p, bt)))
            sell_cond.append(bool(sell_op(p, st)))

    return _finish(
        "Std Dev σ (calc)", sigma,
        {"StdDev Mean": mu, "StdDev Lower": lower, "StdDev Upper": upper,
         "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    # §10.1: 14-period lookback default. `window` unread — ADX has its own period.
    n = int(adx_window) if adx_window else 14

    # Close-only proxy (no per-bar H/L available in single-price series):
    # price acts as proxy for H, L and C.
    delta = prices.diff()
    plus_dm = delta.clip(lower=0)
    minus_dm = (-delta).clip(lower=0)
    tr = prices.diff().abs()

    atr = tr.rolling(window=n).mean()
    plu_avg = plus_dm.rolling(window=n).mean()
    min_avg = minus_dm.rolling(window=n).mean()

    # §1.5 degenerate: ATR = 0  →  +DI = −DI = 0 (reads "no trend").
    # Use fillna on the division result but only where atr is 0 AND defined.
    atr_nonzero = atr.replace(0, np.nan)
    plus_di  = (plu_avg / atr_nonzero * 100).where(atr.notna(), np.nan)
    minus_di = (min_avg / atr_nonzero * 100).where(atr.notna(), np.nan)
    # Where ATR is exactly 0 (valid but flat): DI → 0.
    plus_di  = plus_di.where(atr != 0, 0.0)
    minus_di = minus_di.where(atr != 0, 0.0)

    # DX: |+DI − −DI| / (+DI + −DI) × 100; when both DI are 0, DX = 0.
    di_sum = plus_di + minus_di
    dx = pd.Series(np.nan, index=prices.index)
    di_defined = plus_di.notna() & minus_di.notna()
    dx[di_defined & (di_sum > 0)] = (
        (plus_di - minus_di).abs() / di_sum * 100
    )[di_defined & (di_sum > 0)]
    dx[di_defined & (di_sum == 0)] = 0.0     # §1.5: both DI zero → DX = 0

    adx = dx.rolling(window=n).mean()
    warmup = _first_valid_pos(adx) + 1

    adx_vals = adx.tolist()
    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(adx_vals)):
        prev = adx_vals[i - 1] if i > 0 else np.nan
        curr = adx_vals[i]
        if pd.isna(prev) or pd.isna(curr):
            buy_cond.append(False); sell_cond.append(False); undef += 1; continue
        buy_cond.append(_edge_cross(prev, curr, strong, buy_direction))
        sell_cond.append(_edge_cross(prev, curr, weak, sell_direction))

    return _finish(
        "ADX (calc)", adx, {"+DI": plus_di, "-DI": minus_di},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
    **_ignored,
) -> dict:
    # §1.3, §11.3: HA is a LEVEL-mode indicator. The buy condition fires on
    # every Green bar (HA Close > threshold); the sell on every Red bar.
    # The transition_only flag has been removed — the state machine (repeat=False)
    # already prevents a second Buy while a position is open, so no additional
    # suppression is needed. With repeat=True every qualifying bar executes.
    #
    # With a single price series O=H=L=C=price, so HA Close = price and
    # HA Open = (prev_HA_Open + prev_HA_Close) / 2.  `window` is unread.
    ha_close = prices
    ha_open = [prices.iloc[0]]
    for i in range(1, len(prices)):
        ha_open.append((ha_open[-1] + ha_close.iloc[i - 1]) / 2)
    ha_open_s = pd.Series(ha_open, index=prices.index)

    # Row 0's HA Open is the seed; it cannot produce a signal.
    warmup = 1

    # §11.3 Eq. 95/96: threshold anchored to HA Open. At pct=0 this collapses
    # to the canonical HA Close vs HA Open comparison (Green / Red candle).
    buy_thresh = _threshold(ha_open_s, buy_pct, buy_direction)
    sell_thresh = _threshold(ha_open_s, sell_pct, sell_direction)

    buy_op  = (lambda c, t: c > t) if buy_direction  == "above" else (lambda c, t: c < t)
    sell_op = (lambda c, t: c < t) if sell_direction == "below" else (lambda c, t: c > t)

    buy_cond, sell_cond = [], []
    undef = 0
    for i in range(len(prices)):
        c, bt, st = ha_close.iloc[i], buy_thresh.iloc[i], sell_thresh.iloc[i]
        if pd.isna(c) or pd.isna(bt) or pd.isna(st):
            buy_cond.append(False); sell_cond.append(False); undef += 1
        else:
            buy_cond.append(bool(buy_op(c, bt)))
            sell_cond.append(bool(sell_op(c, st)))

    return _finish(
        "HA Close (calc)", ha_close,
        {"HA Open": ha_open_s, "Buy Threshold": buy_thresh, "Sell Threshold": sell_thresh},
        buy_cond, sell_cond, warmup, repeat, prices.index, undef,
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
        # §1.3: hard min 2, recommended 10-50, default 20.
        "window": {"label": "Window (Periods)", "default": 20, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [],
    },
    "Exponential Moving Average": {
        # §1.3: same as SMA.
        "window": {"label": "Window (Periods)", "default": 20, "min": 2, "max": 500},
        "uses_pct": True,
        "fields": [],
    },
    "Stochastic Oscillator": {
        # §1.3: hard min 5 (flat windows below this produce no %K).
        "window": {
            "label": "Lookback Period", "default": 14, "min": 5, "max": 500,
            "help": "Periods for the %K high/low range. Hard minimum 5 — below that, "
                    "flat windows produce no %K value (§1.3). Default 14.",
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
        # §5.1: fast must be strictly < slow. Validated in run_indicator.
        "window": None,
        "uses_pct": True,
        "fields": [
            {"key": "fast_window", "label": "Fast EMA Period", "type": "int", "slot": "shared",
             "default": 12, "min": 2, "max": 200,
             "help": "Must be strictly less than Slow EMA Period (§5.1)."},
            {"key": "slow_window", "label": "Slow EMA Period", "type": "int", "slot": "shared",
             "default": 26, "min": 2, "max": 400},
            {"key": "signal_window", "label": "Signal EMA Period", "type": "int", "slot": "shared",
             "default": 9, "min": 1, "max": 200},
        ],
    },
    "Bollinger Bands": {
        # §1.3, §6.1: min 6 at k=2 (k ≥ √(n−1) makes signals unreachable).
        # σ is always population (ddof=0); the sample option has been removed.
        "window": {
            "label": "Window (Periods)", "default": 20, "min": 6, "max": 500,
            "help": "Hard minimum 6 at k=2. Minimum rises with k: k=2.5→8, k=3→11 (§1.3).",
        },
        "uses_pct": True,
        "fields": [
            {"key": "k", "label": "Band Width (k × σ)", "type": "float", "slot": "shared",
             "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
             "help": "Standard-deviation multiplier. σ uses population denominator (§6.1)."},
        ],
    },
    "Relative Strength Index": {
        "window": {
            "label": "Lookback Period", "default": 14, "min": 2, "max": 500,
            "help": "Periods averaged for gains and losses. Default 14 (§7.1).",
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
        # §1.3, §9.1: same min window rule as Bollinger. ddof=0 enforced.
        "window": {
            "label": "Window (Periods)", "default": 20, "min": 6, "max": 500,
            "help": "Hard minimum 6 at k=2. σ uses population denominator (§9.1).",
        },
        "uses_pct": True,
        "fields": [
            {"key": "k", "label": "Band Width (k × σ)", "type": "float", "slot": "shared",
             "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
        ],
    },
    "ADX": {
        "window": None,
        "uses_pct": False,
        "fields": [
            {"key": "adx_window", "label": "ADX Period", "type": "int", "slot": "shared",
             "default": 14, "min": 2, "max": 200,
             "help": "ADX needs ≈2n bars before producing a value (§1.3). Default 14 (§10.1)."},
            {"key": "strong", "label": "Buy Threshold (ADX)", "type": "float", "slot": "buy",
             "default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0},
            {"key": "weak", "label": "Sell Threshold (ADX)", "type": "float", "slot": "sell",
             "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0},
        ],
    },
    "Heikin Ashi": {
        # §1.3, §11.3: level-mode indicator; no window, no transition_only.
        "window": None,
        "uses_pct": True,
        "fields": [],
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

    # §5.1 MACD: fast must be strictly < slow. Reversed windows invert every
    # signal silently without producing an error, so catch it here.
    if indicator_name == "MACD":
        fast = int(extras.get("fast_window", 12))
        slow = int(extras.get("slow_window", 26))
        if fast >= slow:
            raise ValueError(
                f"MACD Fast EMA Period ({fast}) must be strictly less than "
                f"Slow EMA Period ({slow}). Swap the values or reduce Fast."
            )

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