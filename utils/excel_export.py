"""
utils/excel_export.py
Builds an .xlsx workbook where every calculated cell contains a LIVE Excel
formula (not a hardcoded value). Change Window / % thresholds / direction on
the Settings sheet and the whole Results sheet recalculates in Excel.

Public entry point: build_workbook(...) -> openpyxl.Workbook
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

BLUE  = Font(color="0000FF")
BLACK = Font(color="000000")
GREY  = Font(color="808080", italic=True)
HDR_FILL = PatternFill("solid", start_color="1F2937", end_color="1F2937")
HDR_FONT = Font(color="FFFFFF", bold=True)
HELP_FILL = PatternFill("solid", start_color="F3F4F6", end_color="F3F4F6")

RAW_COL_ORDER = [
    "S/N", "Symbol", "Transaction Time",
    "Ask Price", "Bid Price", "Mid Price",
    "Moving Average", "Position", "Action",
]

# Settings sheet absolute cell refs
WIN     = "Settings!$B$1"
BUYPCT  = "Settings!$B$2"
SELLPCT = "Settings!$B$3"
BUYDIR  = "Settings!$B$4"
SELLDIR = "Settings!$B$5"
KBAND   = "Settings!$B$6"
ALPHA   = "Settings!$B$7"


def _build_settings_sheet(wb, window, buy_pct, sell_pct, buy_direction, sell_direction, repeat_flag=False):
    ws = wb.active
    ws.title = "Settings"
    rows = [
        ("Window (Periods)",  window),
        ("Buy %",              buy_pct),
        ("Sell %",             sell_pct),
        ("Buy Direction",      buy_direction),
        ("Sell Direction",     sell_direction),
        ("Band k (Bollinger/StdDev)", 2.0),
        ("EMA alpha (auto)",  "=2/(B1+1)"),
        ("Repeat Trade Flag", "TRUE" if repeat_flag else "FALSE"),
    ]
    for i, (label, val) in enumerate(rows, start=1):
        ws.cell(row=i, column=1, value=label).font = Font(bold=True)
        c = ws.cell(row=i, column=2, value=val)
        c.font = BLUE if not (isinstance(val, str) and val.startswith("=")) else BLACK
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 16
    ws["D1"] = ("Edit the blue cells above (Window, %, Direction) — the Results "
                "sheet recalculates automatically. Direction must be exactly "
                "'above' or 'below'. Repeat Trade Flag is baked into the "
                "Position/Action formulas at export time — changing it here "
                "after export will NOT update the formulas; re-export instead.")
    ws["D1"].font = GREY
    return ws


def _offset(price_cell, win_ref=WIN):
    """Trailing rolling window range ending at price_cell, length = Window."""
    return f"OFFSET({price_cell},-({win_ref}-1),0,{win_ref},1)"


def _state_machine_formulas(r, buy_cell, sell_cell, prev_pos_ref, repeat_flag=False):
    """Universal Position(calc)/Action(calc) formulas (used by ALL indicators).

    repeat_flag=False (default): alternates Buy -> Sell -> Buy -> Sell. A new
        Buy is ignored until the previous position is closed with a Sell.
        Mirrors indicators.engine._state_machine(repeat=False).
    repeat_flag=True: reacts to every signal; consecutive Buys/Sells allowed.
        Mirrors indicators.engine._state_machine(repeat=True).
    """
    if repeat_flag:
        action = (f'=IF({buy_cell},"Buy",'
                  f'IF({sell_cell},"Sell","Hold"))')
        position = (f'=IF({buy_cell},"In",'
                    f'IF({sell_cell},"Out",{prev_pos_ref}))')
    else:
        action = (f'=IF(AND({prev_pos_ref}="Out",{buy_cell}),"Buy",'
                  f'IF(AND({prev_pos_ref}="In",{sell_cell}),"Sell","Hold"))')
        position = (f'=IF(AND({prev_pos_ref}="Out",{buy_cell}),"In",'
                    f'IF(AND({prev_pos_ref}="In",{sell_cell}),"Out",{prev_pos_ref}))')
    return position, action


def _status_formula(uploaded_action_cell, action_calc_cell):
    return (f'=IF(TRIM({uploaded_action_cell})="","N/A",'
            f'IF(LOWER(TRIM({uploaded_action_cell}))=LOWER(TRIM({action_calc_cell})),'
            f'"Pass","Fail"))')


# ─────────────────────────── per-indicator column specs ─────────────────────
# Each spec is a list of (col_name, hidden, formula_fn(r, L) -> str_or_value)
# L is a dict {col_name: column_letter}. "price" key always maps to price col.

def _sma_spec():
    def ma(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(p)}))'
    def buyt(r, L):
        m = f'{L["Moving Average (calc)"]}{r}'
        return f'=IF({m}="","",{m}*(1+IF({BUYDIR}="above",1,-1)*{BUYPCT}/100))'
    def sellt(r, L):
        m = f'{L["Moving Average (calc)"]}{r}'
        return f'=IF({m}="","",{m}*(1+IF({SELLDIR}="above",1,-1)*{SELLPCT}/100))'
    def buyc(r, L):
        p = f'{L["price"]}{r}'; bt = f'{L["Buy Threshold"]}{r}'
        return f'=IF({bt}="",FALSE,IF({BUYDIR}="above",{p}>{bt},{p}<{bt}))'
    def sellc(r, L):
        p = f'{L["price"]}{r}'; st = f'{L["Sell Threshold"]}{r}'
        return f'=IF({st}="",FALSE,IF({SELLDIR}="below",{p}<{st},{p}>{st}))'
    return [
        ("Moving Average (calc)", False, ma),
        ("Buy Threshold",         False, buyt),
        ("Sell Threshold",        False, sellt),
        ("Buy Condition",         False, buyc),
        ("Sell Condition",        False, sellc),
    ]


def _ema_spec():
    def ema(r, L):
        if r == 2:
            return f'={L["price"]}{r}'
        p = f'{L["price"]}{r}'; prev = f'{L["EMA (calc)"]}{r-1}'
        return f'={ALPHA}*{p}+(1-{ALPHA})*{prev}'
    def buyt(r, L):
        m = f'{L["EMA (calc)"]}{r}'
        return f'={m}*(1+IF({BUYDIR}="above",1,-1)*{BUYPCT}/100)'
    def sellt(r, L):
        m = f'{L["EMA (calc)"]}{r}'
        return f'={m}*(1+IF({SELLDIR}="above",1,-1)*{SELLPCT}/100)'
    def buyc(r, L):
        p = f'{L["price"]}{r}'; bt = f'{L["Buy Threshold"]}{r}'
        return f'=IF({BUYDIR}="above",{p}>{bt},{p}<{bt})'
    def sellc(r, L):
        p = f'{L["price"]}{r}'; st = f'{L["Sell Threshold"]}{r}'
        return f'=IF({SELLDIR}="below",{p}<{st},{p}>{st})'
    return [
        ("EMA (calc)",      False, ema),
        ("Buy Threshold",   False, buyt),
        ("Sell Threshold",  False, sellt),
        ("Buy Condition",   False, buyc),
        ("Sell Condition",  False, sellc),
    ]


def _stochastic_spec():
    def k(r, L):
        p = f'{L["price"]}{r}'
        off = _offset(p)
        lo = f'MIN({off})'; hi = f'MAX({off})'
        return (f'=IF((ROW()-1)<{WIN},50,'
                f'IF(({hi}-{lo})=0,50,({p}-{lo})/({hi}-{lo})*100))')
    def d(r, L):
        kc = f'{L["%K (calc)"]}{r}'
        if r < 4:
            return f'=""'
        return f'=IF((ROW()-1)<3,"",AVERAGE(OFFSET({kc},-2,0,3,1)))'
    def buyc(r, L):
        if r == 2:
            return "=FALSE"
        prevk = f'{L["%K (calc)"]}{r-1}'; currk = f'{L["%K (calc)"]}{r}'
        return f'=AND({prevk}<=20,{currk}>20)'
    def sellc(r, L):
        if r == 2:
            return "=FALSE"
        prevk = f'{L["%K (calc)"]}{r-1}'; currk = f'{L["%K (calc)"]}{r}'
        return f'=AND({prevk}>=80,{currk}<80)'
    return [
        ("%K (calc)",      False, k),
        ("%D (Signal)",    False, d),
        ("Buy Condition",  False, buyc),
        ("Sell Condition", False, sellc),
    ]


def _macd_spec():
    def ema12(r, L):
        if r == 2:
            return f'={L["price"]}{r}'
        p = f'{L["price"]}{r}'; prev = f'{L["EMA12 (helper)"]}{r-1}'
        return f'=(2/13)*{p}+(1-(2/13))*{prev}'
    def ema26(r, L):
        if r == 2:
            return f'={L["price"]}{r}'
        p = f'{L["price"]}{r}'; prev = f'{L["EMA26 (helper)"]}{r-1}'
        return f'=(2/27)*{p}+(1-(2/27))*{prev}'
    def macd(r, L):
        a = f'{L["EMA12 (helper)"]}{r}'; b = f'{L["EMA26 (helper)"]}{r}'
        return f'={a}-{b}'
    def signal(r, L):
        m = f'{L["MACD (calc)"]}{r}'
        if r == 2:
            return f'={m}'
        prev = f'{L["MACD Signal"]}{r-1}'
        return f'=(2/10)*{m}+(1-(2/10))*{prev}'
    def hist(r, L):
        m = f'{L["MACD (calc)"]}{r}'; s = f'{L["MACD Signal"]}{r}'
        return f'={m}-{s}'
    def buyc(r, L):
        if r == 2:
            return "=FALSE"
        pm = f'{L["MACD (calc)"]}{r-1}'; ps = f'{L["MACD Signal"]}{r-1}'
        cm = f'{L["MACD (calc)"]}{r}'; cs = f'{L["MACD Signal"]}{r}'
        return f'=AND({pm}<={ps},{cm}>{cs})'
    def sellc(r, L):
        if r == 2:
            return "=FALSE"
        pm = f'{L["MACD (calc)"]}{r-1}'; ps = f'{L["MACD Signal"]}{r-1}'
        cm = f'{L["MACD (calc)"]}{r}'; cs = f'{L["MACD Signal"]}{r}'
        return f'=AND({pm}>={ps},{cm}<{cs})'
    return [
        ("EMA12 (helper)",  True,  ema12),
        ("EMA26 (helper)",  True,  ema26),
        ("MACD (calc)",     False, macd),
        ("MACD Signal",     False, signal),
        ("MACD Histogram",  False, hist),
        ("Buy Condition",   False, buyc),
        ("Sell Condition",  False, sellc),
    ]


def _bollinger_spec():
    def mid(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(p)}))'
    def upper(r, L):
        p = f'{L["price"]}{r}'; m = f'{L["BB Middle (calc)"]}{r}'
        return f'=IF({m}="","",{m}+{KBAND}*STDEVP({_offset(p)}))'
    def lower(r, L):
        p = f'{L["price"]}{r}'; m = f'{L["BB Middle (calc)"]}{r}'
        return f'=IF({m}="","",{m}-{KBAND}*STDEVP({_offset(p)}))'
    def buyc(r, L):
        p = f'{L["price"]}{r}'; lo = f'{L["BB Lower"]}{r}'
        return f'=IF({lo}="",FALSE,{p}<{lo})'
    def sellc(r, L):
        p = f'{L["price"]}{r}'; up = f'{L["BB Upper"]}{r}'
        return f'=IF({up}="",FALSE,{p}>{up})'
    return [
        ("BB Middle (calc)", False, mid),
        ("BB Upper",         False, upper),
        ("BB Lower",         False, lower),
        ("Buy Condition",    False, buyc),
        ("Sell Condition",   False, sellc),
    ]


def _rsi_spec():
    def gain(r, L):
        if r == 2:
            return '=""'
        p = f'{L["price"]}{r}'; pp = f'{L["price"]}{r-1}'
        return f'=MAX({p}-{pp},0)'
    def loss(r, L):
        if r == 2:
            return '=""'
        p = f'{L["price"]}{r}'; pp = f'{L["price"]}{r-1}'
        return f'=MAX({pp}-{p},0)'
    def avgg(r, L):
        g = f'{L["Gain (helper)"]}{r}'
        off = _offset(g)
        return f'=IF(COUNT({off})<{WIN},"",AVERAGE({off}))'
    def avgl(r, L):
        l = f'{L["Loss (helper)"]}{r}'
        off = _offset(l)
        return f'=IF(COUNT({off})<{WIN},"",AVERAGE({off}))'
    def rsi(r, L):
        ag = f'{L["Avg Gain (helper)"]}{r}'; al = f'{L["Avg Loss (helper)"]}{r}'
        return (f'=IF(OR({ag}="",{al}=""),"",'
                f'IF({al}=0,"",100-100/(1+{ag}/{al})))')
    def buyc(r, L):
        if r == 2:
            return "=FALSE"
        pr = f'{L["RSI (calc)"]}{r-1}'; cr = f'{L["RSI (calc)"]}{r}'
        return f'=IF({cr}="",FALSE,IF({pr}="",FALSE,AND({pr}<=30,{cr}>30)))'
    def sellc(r, L):
        if r == 2:
            return "=FALSE"
        pr = f'{L["RSI (calc)"]}{r-1}'; cr = f'{L["RSI (calc)"]}{r}'
        return f'=IF({cr}="",FALSE,IF({pr}="",FALSE,AND({pr}>=70,{cr}<70)))'
    return [
        ("Gain (helper)",     True,  gain),
        ("Loss (helper)",     True,  loss),
        ("Avg Gain (helper)", True,  avgg),
        ("Avg Loss (helper)", True,  avgl),
        ("RSI (calc)",        False, rsi),
        ("Buy Condition",     False, buyc),
        ("Sell Condition",    False, sellc),
    ]


def _fibonacci_spec():
    def lo(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",MIN({_offset(p)}))'
    def hi(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",MAX({_offset(p)}))'
    def fib(mult, name):
        def f(r, L):
            l = f'{L["Roll Low (helper)"]}{r}'; h = f'{L["Roll High (helper)"]}{r}'
            return f'=IF(OR({l}="",{h}=""),"",{l}+{mult}*({h}-{l}))'
        return f
    def buyc(r, L):
        p = f'{L["price"]}{r}'; f382 = f'{L["Fib 38.2%"]}{r}'
        return f'=IF({f382}="",FALSE,{p}<{f382})'
    def sellc(r, L):
        p = f'{L["price"]}{r}'; f618 = f'{L["Fib 61.8%"]}{r}'
        return f'=IF({f618}="",FALSE,{p}>{f618})'
    return [
        ("Roll Low (helper)",  True,  lo),
        ("Roll High (helper)", True,  hi),
        ("Fib 23.6%",          False, fib(0.236, "236")),
        ("Fib 38.2%",          False, fib(0.382, "382")),
        ("Fib 50% (calc)",     False, fib(0.500, "500")),
        ("Fib 61.8%",          False, fib(0.618, "618")),
        ("Fib 78.6%",          False, fib(0.786, "786")),
        ("Buy Condition",      False, buyc),
        ("Sell Condition",     False, sellc),
    ]


def _stddev_spec():
    def sigma(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",STDEVP({_offset(p)}))'
    def mu(r, L):
        p = f'{L["price"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(p)}))'
    def lower(r, L):
        m = f'{L["StdDev Mean"]}{r}'; s = f'{L["Std Dev  σ (calc)"]}{r}'
        return f'=IF(OR({m}="",{s}=""),"",{m}-{KBAND}*{s})'
    def upper(r, L):
        m = f'{L["StdDev Mean"]}{r}'; s = f'{L["Std Dev  σ (calc)"]}{r}'
        return f'=IF(OR({m}="",{s}=""),"",{m}+{KBAND}*{s})'
    def buyc(r, L):
        p = f'{L["price"]}{r}'; lo = f'{L["StdDev Lower"]}{r}'
        return f'=IF({lo}="",FALSE,{p}<{lo})'
    def sellc(r, L):
        p = f'{L["price"]}{r}'; up = f'{L["StdDev Upper"]}{r}'
        return f'=IF({up}="",FALSE,{p}>{up})'
    return [
        ("Std Dev  σ (calc)", False, sigma),
        ("StdDev Mean",            False, mu),
        ("StdDev Lower",           False, lower),
        ("StdDev Upper",           False, upper),
        ("Buy Condition",          False, buyc),
        ("Sell Condition",         False, sellc),
    ]


def _adx_spec():
    def delta(r, L):
        if r == 2:
            return "=0"
        p = f'{L["price"]}{r}'; pp = f'{L["price"]}{r-1}'
        return f'={p}-{pp}'
    def plusdm(r, L):
        d = f'{L["Delta (helper)"]}{r}'
        return f'=MAX({d},0)'
    def minusdm(r, L):
        d = f'{L["Delta (helper)"]}{r}'
        return f'=MAX(-{d},0)'
    def tr(r, L):
        d = f'{L["Delta (helper)"]}{r}'
        return f'=ABS({d})'
    def atr(r, L):
        t = f'{L["TR (helper)"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(t)}))'
    def plusdmavg(r, L):
        p = f'{L["PlusDM (helper)"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(p)}))'
    def minusdmavg(r, L):
        m = f'{L["MinusDM (helper)"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(m)}))'
    def plusdi(r, L):
        a = f'{L["ATR (helper)"]}{r}'; pa = f'{L["PlusDM Avg (helper)"]}{r}'
        return f'=IF(OR({a}="",{a}=0),0,{pa}/{a}*100)'
    def minusdi(r, L):
        a = f'{L["ATR (helper)"]}{r}'; ma = f'{L["MinusDM Avg (helper)"]}{r}'
        return f'=IF(OR({a}="",{a}=0),0,{ma}/{a}*100)'
    def dx(r, L):
        pdi = f'{L["+DI"]}{r}'; mdi = f'{L["-DI"]}{r}'
        return f'=IF(({pdi}+{mdi})=0,0,ABS({pdi}-{mdi})/({pdi}+{mdi})*100)'
    def adx(r, L):
        d = f'{L["DX (helper)"]}{r}'
        return f'=IF((ROW()-1)<{WIN},"",AVERAGE({_offset(d)}))'
    def buyc(r, L):
        if r == 2:
            return "=FALSE"
        pr = f'{L["ADX (calc)"]}{r-1}'; cr = f'{L["ADX (calc)"]}{r}'
        return f'=IF({cr}="",FALSE,IF({pr}="",FALSE,AND({pr}<=25,{cr}>25)))'
    def sellc(r, L):
        if r == 2:
            return "=FALSE"
        pr = f'{L["ADX (calc)"]}{r-1}'; cr = f'{L["ADX (calc)"]}{r}'
        return f'=IF({cr}="",FALSE,IF({pr}="",FALSE,AND({pr}>=20,{cr}<20)))'
    return [
        ("Delta (helper)",        True,  delta),
        ("PlusDM (helper)",       True,  plusdm),
        ("MinusDM (helper)",      True,  minusdm),
        ("TR (helper)",           True,  tr),
        ("ATR (helper)",          True,  atr),
        ("PlusDM Avg (helper)",   True,  plusdmavg),
        ("MinusDM Avg (helper)",  True,  minusdmavg),
        ("+DI",                   False, plusdi),
        ("-DI",                   False, minusdi),
        ("DX (helper)",           True,  dx),
        ("ADX (calc)",            False, adx),
        ("Buy Condition",         False, buyc),
        ("Sell Condition",        False, sellc),
    ]


def _heikin_ashi_spec():
    def haclose(r, L):
        return f'={L["price"]}{r}'
    def haopen(r, L):
        if r == 2:
            return f'={L["price"]}{r}'
        prevopen = f'{L["HA Open"]}{r-1}'; prevclose = f'{L["HA Close (calc)"]}{r-1}'
        return f'=({prevopen}+{prevclose})/2'
    def buyc(r, L):
        c = f'{L["HA Close (calc)"]}{r}'; o = f'{L["HA Open"]}{r}'
        if r == 2:
            return f'=({c}>{o})'
        pc = f'{L["HA Close (calc)"]}{r-1}'; po = f'{L["HA Open"]}{r-1}'
        return f'=AND({c}>{o},NOT({pc}>{po}))'
    def sellc(r, L):
        c = f'{L["HA Close (calc)"]}{r}'; o = f'{L["HA Open"]}{r}'
        if r == 2:
            return f'=({c}<{o})'
        pc = f'{L["HA Close (calc)"]}{r-1}'; po = f'{L["HA Open"]}{r-1}'
        return f'=AND({c}<{o},NOT({pc}<{po}))'
    return [
        ("HA Close (calc)", False, haclose),
        ("HA Open",          False, haopen),
        ("Buy Condition",    False, buyc),
        ("Sell Condition",   False, sellc),
    ]


INDICATOR_SPECS = {
    "Simple Moving Average":      _sma_spec,
    "Exponential Moving Average": _ema_spec,
    "Stochastic Oscillator":      _stochastic_spec,
    "MACD":                        _macd_spec,
    "Bollinger Bands":             _bollinger_spec,
    "Relative Strength Index":     _rsi_spec,
    "Fibonacci Retracement":       _fibonacci_spec,
    "Standard Deviation":          _stddev_spec,
    "ADX":                         _adx_spec,
    "Heikin Ashi":                 _heikin_ashi_spec,
}


def build_workbook(
    df_raw,
    indicator_name: str,
    price_col: str,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat_flag: bool = False,
):
    """Returns an openpyxl Workbook with every calculated cell as a live formula."""
    if indicator_name not in INDICATOR_SPECS:
        raise ValueError(f"Unknown indicator: {indicator_name}")

    wb = Workbook()
    _build_settings_sheet(wb, window, buy_pct, sell_pct, buy_direction, sell_direction, repeat_flag)
    ws = wb.create_sheet("Results")

    raw_cols = [c for c in RAW_COL_ORDER if c in df_raw.columns]
    spec = INDICATOR_SPECS[indicator_name]()
    calc_names = [name for name, _, _ in spec] + ["Position (calc)", "Action (calc)", "Status"]
    all_cols = raw_cols + calc_names

    L = {}
    for i, name in enumerate(all_cols, start=1):
        L[name] = get_column_letter(i)
    L["price"] = L[price_col]

    n = len(df_raw)
    last_row = n + 1

    # ── header row ──────────────────────────────────────────────────────────
    hidden_set = {name for name, hidden, _ in spec if hidden}
    for col_idx, name in enumerate(all_cols, start=1):
        c = ws.cell(row=1, column=col_idx, value=name)
        c.font = HDR_FONT
        c.fill = HDR_FILL
        if name in hidden_set:
            ws.column_dimensions[get_column_letter(col_idx)].hidden = True

    # ── raw data rows (static values, blue font = inputs) ──────────────────
    has_action = "Action" in df_raw.columns
    for i, (_, row) in enumerate(df_raw.iterrows()):
        r = i + 2
        for name in raw_cols:
            val = row.get(name, None)
            c = ws.cell(row=r, column=all_cols.index(name) + 1)
            if name == "Transaction Time" and pd_isna_safe(val) is False:
                try:
                    c.value = val.to_pydatetime() if hasattr(val, "to_pydatetime") else val
                    c.number_format = "yyyy-mm-dd hh:mm:ss"
                except Exception:
                    c.value = str(val)
            else:
                c.value = None if pd_isna_safe(val) else val
            c.font = BLUE

    # ── calculated columns: write formulas row by row ──────────────────────
    for r in range(2, last_row + 1):
        for name, hidden, fn in spec:
            formula = fn(r, L)
            c = ws.cell(row=r, column=all_cols.index(name) + 1, value=formula)
            c.font = GREY if hidden else BLACK
            # c.number_format = "0.0000"

        prev_pos_ref = '"Out"' if r == 2 else f'{L["Position (calc)"]}{r-1}'
        buy_cell = f'{L["Buy Condition"]}{r}'
        sell_cell = f'{L["Sell Condition"]}{r}'
        pos_f, act_f = _state_machine_formulas(r, buy_cell, sell_cell, prev_pos_ref, repeat_flag)
        ws.cell(row=r, column=all_cols.index("Position (calc)") + 1, value=pos_f).font = BLACK
        ws.cell(row=r, column=all_cols.index("Action (calc)") + 1, value=act_f).font = BLACK

        if has_action:
            status_f = _status_formula(f'{L["Action"]}{r}', f'{L["Action (calc)"]}{r}')
        else:
            status_f = "N/A"
        ws.cell(row=r, column=all_cols.index("Status") + 1, value=status_f).font = BLACK

    # ── cosmetics ────────────────────────────────────────────────────────────
    for col_idx, name in enumerate(all_cols, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = max(12, min(22, len(name) + 4))
    ws.freeze_panes = "A2"

    # ── summary block (formula-driven) ─────────────────────────────────────
    summary = wb.create_sheet("Summary")
    summary["A1"] = "Metric"; summary["B1"] = "Value"
    summary["A1"].font = HDR_FONT; summary["B1"].font = HDR_FONT
    summary["A1"].fill = HDR_FILL; summary["B1"].fill = HDR_FILL
    action_calc_rng = f'Results!{L["Action (calc)"]}2:{L["Action (calc)"]}{last_row}'
    status_rng = f'Results!{L["Status"]}2:{L["Status"]}{last_row}'
    rows = [
        ("Total Rows", f"=COUNTA({action_calc_rng})"),
        ("Buy Signals", f'=COUNTIF({action_calc_rng},"Buy")'),
        ("Sell Signals", f'=COUNTIF({action_calc_rng},"Sell")'),
        ("Pass", f'=COUNTIF({status_rng},"Pass")'),
        ("Fail", f'=COUNTIF({status_rng},"Fail")'),
        ("N/A", f'=COUNTIF({status_rng},"N/A")'),
    ]
    for i, (label, formula) in enumerate(rows, start=2):
        summary.cell(row=i, column=1, value=label).font = Font(bold=True)
        summary.cell(row=i, column=2, value=formula).font = BLACK
    summary.column_dimensions["A"].width = 18
    summary.column_dimensions["B"].width = 14

    return wb


def pd_isna_safe(val):
    try:
        import pandas as pd
        return bool(pd.isna(val))
    except Exception:
        return val is None