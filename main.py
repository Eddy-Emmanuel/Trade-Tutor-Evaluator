# main.py

import io
from datetime import date, time

import pandas as pd
import streamlit as st

from utils.excel_export import build_workbook
from utils.pipeline import build_result_df, compute_pnl, filter_by_window
from indicators.engine import INDICATOR_MAP, INDICATOR_HINTS, spec_for
from utils.file_loader import load_file, available_price_cols, validate_sma_columns

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeTutor — Strategy Builder",
    page_icon="📊",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
# Light palette matched to the target design: white surfaces, indigo primary
# (#2563EB), muted violet-grey field labels, 8px radii.
st.markdown("""
<style>
:root {
    --tt-primary:  #2563EB;
    --tt-label:    #5B5F7D;
    --tt-border:   #E3E6F0;
    --tt-surface:  #FFFFFF;
    --tt-tint:     #EEF2FF;
    --tt-text:     #1A1D2E;
    --tt-muted:    #8A90AB;
    --tt-up:       #16A34A;
    --tt-down:     #DC2626;
}

/* ── Ticker tape ─────────────────────────────────────────────────── */
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-wrap {
    background: #0A0F1E;
    overflow: hidden;
    border-radius: 6px;
    padding: 6px 0;
    margin-bottom: 4px;
}
.ticker-track { display: inline-flex; white-space: nowrap; animation: ticker 40s linear infinite; }
.ticker-item {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    color: #CCD6F6; margin-right: 36px; font-family: 'Courier New', monospace;
}
.ticker-item .up   { color: #26A65B; }
.ticker-item .down { color: #E74C3C; }
.ticker-arrow { font-size: 0.6rem; margin-left: 2px; }

/* ── Sidebar: field labels + section titles ──────────────────────── */
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: var(--tt-label) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
.tt-section {
    font-size: 0.95rem; font-weight: 700; color: var(--tt-text);
    margin: 4px 0 10px;
}
.tt-sublabel {
    font-size: 0.84rem; font-weight: 500; color: var(--tt-label);
    margin: 10px 0 -6px;
}
.tt-hint {
    font-size: 0.76rem; color: var(--tt-muted);
    line-height: 1.4; margin: -4px 0 12px;
}

/* Cost summary panel under Buy Quantity */
.cost-panel {
    background: var(--tt-tint);
    border-radius: 8px;
    padding: 12px 14px;
    margin: 10px 0 4px;
    font-size: 0.84rem;
    color: var(--tt-label);
}
.cost-row { display: flex; justify-content: space-between; align-items: center; }
.cost-row + .cost-row { margin-top: 6px; }
.cost-row .val { font-weight: 700; color: var(--tt-text); }

/* ── Main area cards ─────────────────────────────────────────────── */
.metric-card {
    background: var(--tt-surface);
    border: 1px solid var(--tt-border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label {
    color: var(--tt-muted); font-size: 0.72rem;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px;
}
.metric-value { color: var(--tt-text); font-size: 1.5rem; font-weight: 700; }
.metric-value.green { color: var(--tt-up); }
.metric-value.red   { color: var(--tt-down); }
.metric-value.amber { color: #D97706; }

.section-header {
    font-size: 1rem; font-weight: 600; color: var(--tt-text);
    border-left: 3px solid var(--tt-primary);
    padding-left: 10px; margin: 22px 0 10px;
}
.chip {
    display: inline-block; border-radius: 5px; padding: 5px 10px;
    margin-bottom: 4px; font-size: 0.82rem;
}
.chip.ok   { background: #ECFDF5; border: 1px solid #A7F3D0; color: #047857; }
.chip.bad  { background: #FEF2F2; border: 1px solid #FECACA; color: #B91C1C; }
</style>
""", unsafe_allow_html=True)


# ── Ticker tape ────────────────────────────────────────────────────────────────
TICKERS = [
    ("ATALAYA MINING", -3.72), ("PARTNERS GRP E", -3.61),
    ("ALLIANZ TECH",   -3.59), ("POLAR CAP",      -3.58),
    ("VESUVIUS",       -3.56), ("BABCOCK INTL",   +3.52),
    ("FIDELITY E.M.LD", -3.46), ("PACIFIC HORIZON", -6.23),
    ("ANTOFAGASTA",    -6.12), ("GLENCORE",       -5.13),
    ("BUNZL",          +5.11), ("ANGLO AMERICAN", -5.02),
    ("TEMPLETON EMRG", -5.00), ("RASPBERRY PI",  -12.67),
]


def _ticker_html(tickers):
    items = ""
    for name, pct in tickers:
        cls = "up" if pct > 0 else "down"
        arrow = "▲" if pct > 0 else "▼"
        items += (
            f'<span class="ticker-item">{name} '
            f'<span class="{cls}">{pct:+.2f}%'
            f'<span class="ticker-arrow">{arrow}</span></span></span>'
        )
    return f'<div class="ticker-wrap"><div class="ticker-track">{items}{items}</div></div>'


st.markdown(_ticker_html(TICKERS), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar helpers
# ══════════════════════════════════════════════════════════════════════════════
INDICATOR_NAMES = list(INDICATOR_MAP.keys())
PRICE_OPTIONS = ["Mid Price", "Ask Price", "Bid Price"]
INSTRUMENT_TYPES = ["FTSE 350", "ETF", "ETN"]

# Fallback universes, used only when the uploaded file has no Symbol column.
INSTRUMENT_UNIVERSE = {
    "FTSE 350": ["TESCO PLC", "BARCLAYS PLC", "GLENCORE PLC", "VESUVIUS PLC", "BUNZL PLC"],
    "ETF":      ["ISHARES CORE FTSE 100", "VANGUARD FTSE 250", "SPDR S&P 500"],
    "ETN":      ["IPATH B S&P 500 VIX", "ELEMENTS MSCI"],
}


def _segmented(label: str, options: list[str], key: str, help_text: str | None = None) -> str:
    """st.segmented_control where available (Streamlit ≥1.40), radio otherwise."""
    st.markdown(f'<div class="tt-sublabel">{label} ⓘ</div>', unsafe_allow_html=True)
    if hasattr(st, "segmented_control"):
        picked = st.segmented_control(
            label, options, default=options[0], key=key,
            label_visibility="collapsed", help=help_text,
        )
        return picked or options[0]
    return st.radio(
        label, options, horizontal=True, key=key,
        label_visibility="collapsed", help=help_text,
    )


def _render_field(field: dict, indicator_name: str) -> tuple[str, object]:
    """Render one control from its declaration in INDICATOR_SPEC."""
    wkey = f"fld_{indicator_name}_{field['key']}"
    label, kind, helptext = field["label"], field["type"], field.get("help")

    if kind == "int":
        val = st.number_input(
            f"{label} ⓘ",
            min_value=int(field.get("min", 1)), max_value=int(field.get("max", 1000)),
            value=int(field["default"]), step=int(field.get("step", 1)),
            key=wkey, help=helptext,
        )
    elif kind == "float":
        val = st.number_input(
            f"{label} ⓘ",
            min_value=float(field.get("min", 0.0)), max_value=float(field.get("max", 1000.0)),
            value=float(field["default"]), step=float(field.get("step", 0.1)),
            key=wkey, help=helptext,
        )
    elif kind == "bool":
        val = st.toggle(f"{label} ⓘ", value=bool(field["default"]), key=wkey, help=helptext)
    elif kind == "select":
        opts = field["options"]                       # list[(label, value)]
        labels = [o[0] for o in opts]
        values = [o[1] for o in opts]
        idx = values.index(field["default"]) if field["default"] in values else 0
        choice = st.selectbox(f"{label} ⓘ", labels, index=idx, key=wkey, help=helptext)
        val = values[labels.index(choice)]
    else:
        raise ValueError(f"Unknown field type: {kind}")

    return field["key"], val


def _render_slot(indicator_name: str, slot: str, into: dict) -> None:
    """Render every field this indicator declares for a given slot.

    slot "buy"/"sell" fields sit where "How Much (%)" would otherwise be, so
    RSI / Stochastic / ADX show their threshold levels instead of a control
    the backend never reads.
    """
    for field in spec_for(indicator_name)["fields"]:
        if field.get("slot", "shared") == slot:
            key, value = _render_field(field, indicator_name)
            into[key] = value


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — field order mirrors the design
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="tt-section">Build Your Strategy</div>', unsafe_allow_html=True)
    st.divider()

    uploaded = st.file_uploader(
        "Upload trade file (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        help="Expected columns: Transaction Time, Mid/Ask/Bid Price, Moving Average, Position, Action",
    )

    st.divider()
    st.markdown('<div class="tt-section">Build your Entry Strategy!</div>', unsafe_allow_html=True)

    # ── Indicator Type ────────────────────────────────────────────────────────
    indicator_name = st.selectbox(
        "Indicator Type ⓘ", INDICATOR_NAMES, index=0,
        help="Technical indicator used to generate signals.",
    )
    st.markdown(f'<div class="tt-hint">{INDICATOR_HINTS[indicator_name]}</div>',
                unsafe_allow_html=True)

    # ── Instrument Type / Instrument ──────────────────────────────────────────
    instrument_type = _segmented(
        "Instrument Type", INSTRUMENT_TYPES, key="instrument_type",
        help_text="Asset class the strategy trades.",
    )

    # Prefer symbols found in the uploaded file; fall back to the static list.
    file_symbols: list[str] = st.session_state.get("file_symbols", [])
    instrument_options = file_symbols or INSTRUMENT_UNIVERSE[instrument_type]
    instrument = st.selectbox(
        "Instrument ⓘ", instrument_options, index=0,
        help="Ticker or company the signals are evaluated against.",
    )

    # ── Buy leg ───────────────────────────────────────────────────────────────
    buy_price = st.selectbox(
        "Buy When ⓘ", PRICE_OPTIONS, index=0, key="buy_price",
        help="Price column compared against the indicator to trigger entries.",
    )

    buy_dir_label = st.selectbox(
        "Buy Direction ⓘ", ["Is Above", "Is Below"], index=0, key="buy_dir",
    )
    buy_direction = "above" if buy_dir_label == "Is Above" else "below"

    st.markdown(f'<div class="tt-sublabel">{buy_dir_label} ⓘ</div>', unsafe_allow_html=True)
    st.selectbox(
        "Buy reference", [indicator_name], key="buy_ref",
        label_visibility="collapsed", disabled=True,
    )

    spec = spec_for(indicator_name)
    params: dict = {}

    # "How Much (%)" only where the engine actually reads it. RSI, Stochastic
    # and ADX get their threshold level here instead — the field QA found to
    # be inert was this one.
    if spec["uses_pct"]:
        buy_pct = st.number_input(
            "How Much (%) ⓘ", min_value=0.0, max_value=100.0,
            value=0.05, step=0.01, format="%.4f", key="buy_pct",
            help="Percentage offset applied to the indicator to form the buy threshold.",
        )
    else:
        buy_pct = 0.0
    _render_slot(indicator_name, "buy", params)

    # Period controls. MACD and ADX declare their own named periods; Heikin
    # Ashi has none, so no generic Window is shown for any of the three.
    if spec["window"]:
        w = spec["window"]
        window = st.number_input(
            f"{w['label']} ⓘ",
            min_value=int(w["min"]), max_value=int(w["max"]),
            value=int(w["default"]), step=1, key=f"window_{indicator_name}",
            help=w.get("help", "Lookback length for the indicator."),
        )
    else:
        window = 1
    _render_slot(indicator_name, "shared", params)

    buy_qty = st.number_input(
        "Buy Quantity ⓘ", min_value=1, value=500, step=1, key="buy_qty",
    )

    cost_panel = st.empty()   # filled after the file loads

    # ── Sell leg ──────────────────────────────────────────────────────────────
    sell_price = st.selectbox(
        "Sell When ⓘ", PRICE_OPTIONS, index=0, key="sell_price",
        help="Price column used to mark exits. Can differ from the buy column.",
    )

    sell_dir_label = st.selectbox(
        "Sell Direction ⓘ", ["Is Below", "Is Above"], index=0, key="sell_dir",
    )
    sell_direction = "below" if sell_dir_label == "Is Below" else "above"

    st.markdown(f'<div class="tt-sublabel">{sell_dir_label} ⓘ</div>', unsafe_allow_html=True)
    st.selectbox(
        "Sell reference", [indicator_name], key="sell_ref",
        label_visibility="collapsed", disabled=True,
    )

    if spec["uses_pct"]:
        sell_pct = st.number_input(
            "How Much (%) ⓘ", min_value=0.0, max_value=100.0,
            value=0.05, step=0.01, format="%.4f", key="sell_pct",
        )
    else:
        sell_pct = 0.0
    _render_slot(indicator_name, "sell", params)

    sell_qty = st.number_input(
        "Sell Quantity ⓘ", min_value=1, value=500, step=1, key="sell_qty",
    )

    # ── Date / time window ────────────────────────────────────────────────────
    d1, d2 = st.columns(2)
    with d1:
        date_from = st.date_input("Date From", value=date(2026, 1, 5), key="date_from")
    with d2:
        date_to = st.date_input("Date To", value=date(2026, 7, 25), key="date_to")

    t1, t2 = st.columns(2)
    with t1:
        start_time = st.time_input("Start Time", value=time(8, 0), key="start_time")
    with t2:
        end_time = st.time_input("End Time", value=time(16, 30), key="end_time")

    if date_from and date_to and date_from > date_to:
        st.warning("Date From is after Date To — no rows will match.")

    # ── Flags ─────────────────────────────────────────────────────────────────
    repeat_flag = st.toggle(
        "Repeat Trade Flag ⓘ", value=False,
        help=(
            "Off — alternates Buy → Sell → Buy → Sell. A new Buy is ignored until "
            "the open position is closed with a Sell.\n\n"
            "On — reacts to every signal, so consecutive Buys or Sells are allowed."
        ),
    )

    core_analysis = st.toggle(
        "CORE Analysis ⓘ", value=False,
        help="Adds the extended performance breakdown below the results table.",
    )

    st.write("")
    generate = st.button("Generate", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Trading Signal Evaluator")
st.caption(
    "Configure the strategy in the sidebar, then press Generate. Results include "
    "buy/sell signals, Status validation (Pass / Fail / N/A), and P&L."
)

if uploaded is None:
    st.info("Upload an Excel or CSV file in the sidebar to get started.")
    with st.expander("Expected column format"):
        st.markdown("""
| Column | Required | Notes |
|---|---|---|
| `Transaction Time` | Recommended | Timestamp or Excel serial number. Needed for the date/time window. |
| `Mid Price` / `Ask Price` / `Bid Price` | **Yes** | At least one price column |
| `Moving Average` | Optional | Pre-computed; will be recalculated |
| `Position` | Optional | `In` / `Out` |
| `Action` | Optional | `Buy` / `Sell` / `Hold` — used for **Status** comparison |
| `Symbol` | Optional | Populates the Instrument dropdown |
""")
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
try:
    df_loaded = load_file(uploaded)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

# Feed the Instrument dropdown from the file on the next rerun.
if "Symbol" in df_loaded.columns:
    syms = sorted({str(s) for s in df_loaded["Symbol"].dropna().unique()})
    if syms and syms != st.session_state.get("file_symbols"):
        st.session_state["file_symbols"] = syms
        st.rerun()

avail_prices = available_price_cols(df_loaded)
if not avail_prices:
    st.error("No price column found. Provide Mid Price, Ask Price, or Bid Price.")
    st.stop()

# Both legs are now honoured; each falls back to the first available column.
price_col = buy_price if buy_price in avail_prices else avail_prices[0]
sell_col = sell_price if sell_price in avail_prices else price_col

# Restrict to the selected instrument, then to the date/time window.
df_raw = df_loaded
if "Symbol" in df_raw.columns and instrument in set(df_raw["Symbol"].astype(str)):
    df_raw = df_raw[df_raw["Symbol"].astype(str) == instrument].reset_index(drop=True)

rows_before = len(df_raw)
df_raw = filter_by_window(df_raw, date_from, date_to, start_time, end_time)
rows_after = len(df_raw)

if rows_after == 0:
    st.warning(
        f"The selected window removed all {rows_before:,} rows. "
        "Widen the date range or the session times."
    )
    st.stop()

# ── Instrument + cost panel ────────────────────────────────────────────────────
symbol = str(df_raw["Symbol"].iloc[0]) if "Symbol" in df_raw.columns else instrument
short_sym = symbol[:4].upper()
last_px = float(df_raw[price_col].iloc[-1])

cost_panel.markdown(
    f'<div class="cost-panel">'
    f'<div class="cost-row"><span>Price per share ({short_sym})</span>'
    f'<span class="val">£{last_px:,.4f}</span></div>'
    f'<div class="cost-row"><span>Total cost ({short_sym} × {buy_qty:,})</span>'
    f'<span class="val">£{last_px * buy_qty:,.2f}</span></div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.success(
    f"Loaded **{rows_after:,}** of {rows_before:,} rows after filtering  ·  "
    f"Instrument: **{symbol}** ({instrument_type})  ·  "
    f"Buy on **{price_col}**, sell on **{sell_col}**"
)

# ── Column validation ──────────────────────────────────────────────────────────
val = validate_sma_columns(df_raw, price_col)
with st.expander(
    f"Column validation — {len(val['present'])} present / {len(val['missing'])} missing",
    expanded=len(val["missing"]) > 0,
):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Present**")
        for col in val["present"]:
            st.markdown(f'<div class="chip ok">✓ {col}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**Missing**")
        for col in val["missing"]:
            note = " (Status → N/A)" if col == "Action" else ""
            st.markdown(f'<div class="chip bad">✗ {col}{note}</div>', unsafe_allow_html=True)

# ── Generate gate ──────────────────────────────────────────────────────────────
if generate:
    st.session_state["has_run"] = True

if not st.session_state.get("has_run"):
    st.info("Press **Generate** in the sidebar to run the strategy.")
    st.stop()

# ── Calculate ──────────────────────────────────────────────────────────────────
try:
    df_result = build_result_df(
        df_raw, indicator_name, price_col,
        window, buy_pct, sell_pct, buy_direction, sell_direction,
        repeat_flag, params=params,
    )
except Exception as e:
    st.error(f"Calculation error: {e}")
    st.stop()

pnl = compute_pnl(df_result, price_col, buy_qty, sell_qty, sell_price_col=sell_col)

# ── Warm-up notice ─────────────────────────────────────────────────────────────
warmup = int(df_result.attrs.get("warmup", 0))
if warmup >= len(df_result):
    st.error(
        f"The lookback needs {warmup:,} rows to become defined, but only "
        f"{len(df_result):,} are in the selected window. Every row is held — "
        "shorten the period or widen the date range."
    )
elif warmup > 0:
    st.caption(
        f"Warm-up: the first {warmup:,} row(s) are held at Out / Hold because "
        f"the lookback is not satisfied yet. Signals start at row {warmup + 1:,}."
    )

# ── KPI cards ──────────────────────────────────────────────────────────────────
n_pass = int((df_result["Status"] == "Pass").sum()) if "Status" in df_result.columns else 0
n_fail = int((df_result["Status"] == "Fail").sum()) if "Status" in df_result.columns else 0

kpis = [
    ("Rows",      f"{len(df_result):,}", ""),
    ("Trades",    f"{pnl['trades']}", ""),
    ("Net P&L",   f"£{pnl['total']:,.2f}", "green" if pnl["total"] >= 0 else "red"),
    ("Pass",      f"{n_pass}", "green" if n_pass else ""),
    ("Fail",      f"{n_fail}", "red" if n_fail else ""),
]
for col, (lbl, val_str, color) in zip(st.columns(len(kpis)), kpis):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
            f'<div class="metric-value {color}">{val_str}</div></div>',
            unsafe_allow_html=True,
        )

# ── Results table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Calculated results</div>', unsafe_allow_html=True)

always_first = ["Transaction Time", price_col]
indicator_cols = [
    c for c in df_result.columns
    if c not in always_first
    and c not in ["S/N", "Symbol", "Position", "Moving Average"]
    and "Condition" not in c
]
display_cols = [c for c in always_first + indicator_cols if c in df_result.columns]


def _highlight(row):
    if row.get("Status", "") == "Fail":
        return ["background-color: #FEF2F2"] * len(row)
    action = row.get("Action (calc)", "")
    if action == "Buy":
        return ["background-color: #ECFDF5"] * len(row)
    if action == "Sell":
        return ["background-color: #FFF7ED"] * len(row)
    return [""] * len(row)


show_df = df_result[display_cols].copy()
MAX_CELLS = 262_144
total_cells = show_df.shape[0] * show_df.shape[1]

if total_cells <= MAX_CELLS:
    st.dataframe(show_df.style.apply(_highlight, axis=1), use_container_width=True, height=430)
else:
    st.caption(f"Large dataset ({total_cells:,} cells) — row highlighting disabled.")
    st.dataframe(show_df, use_container_width=True, height=430)

# ── Trade log ──────────────────────────────────────────────────────────────────
if pnl["trades"] > 0:
    st.markdown('<div class="section-header">Round-trip trade log</div>', unsafe_allow_html=True)
    trades_df = pd.DataFrame({
        "#": range(1, pnl["trades"] + 1),
        f"Buy ({price_col})": pnl["buys"],
        f"Sell ({sell_col})": pnl["sells"],
        "P&L (£)": pnl["pnl_list"],
    })
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
else:
    st.info("No completed round trips with the current settings.")

# ── CORE analysis ──────────────────────────────────────────────────────────────
if core_analysis:
    st.markdown('<div class="section-header">CORE analysis</div>', unsafe_allow_html=True)
    if pnl["trades"] == 0:
        st.caption("No closed trades to analyse.")
    else:
        wins = [x for x in pnl["pnl_list"] if x > 0]
        losses = [x for x in pnl["pnl_list"] if x <= 0]
        in_bars = int((df_result["Position (calc)"] == "In").sum())
        stats = {
            "Win rate": f"{len(wins) / pnl['trades'] * 100:,.1f}%",
            "Average win": f"£{(sum(wins) / len(wins)) if wins else 0:,.2f}",
            "Average loss": f"£{(sum(losses) / len(losses)) if losses else 0:,.2f}",
            "Best trade": f"£{max(pnl['pnl_list']):,.2f}",
            "Worst trade": f"£{min(pnl['pnl_list']):,.2f}",
            "Time in market": f"{in_bars / len(df_result) * 100:,.1f}% of rows",
            "Buy signals": int((df_result["Action (calc)"] == "Buy").sum()),
            "Sell signals": int((df_result["Action (calc)"] == "Sell").sum()),
        }
        st.dataframe(
            pd.DataFrame({"Metric": list(stats), "Value": [str(v) for v in stats.values()]}),
            use_container_width=True, hide_index=True,
        )

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

try:
    wb = build_workbook(
        df_raw, indicator_name, price_col, window,
        buy_pct, sell_pct, buy_direction, sell_direction,
        repeat_flag, params=params,
    )
    buf = io.BytesIO()
    wb.save(buf)
    st.download_button(
        "Download results (Excel — live formulas)",
        data=buf.getvalue(),
        file_name="trade_results_formulas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception as e:
    st.error(f"Excel export error: {e}")

st.caption(
    "The Excel file recalculates live: every indicator value, threshold, "
    "Buy/Sell Condition, Position, Action, and Status cell is a real formula "
    "(see the **Settings** sheet to tweak Window / % / Direction)."
)