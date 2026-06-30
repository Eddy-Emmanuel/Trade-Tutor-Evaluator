import io
import pandas as pd
import streamlit as st

from indicators.engine import INDICATOR_MAP, INDICATOR_HINTS
from utils.file_loader import load_file, available_price_cols, validate_sma_columns
from utils.pipeline import build_result_df, compute_pnl
from utils.excel_export import build_workbook

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeTutor — Strategy Builder",
    page_icon="📊",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Ticker tape ─────────────────────────────────────────────────── */
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-wrap {
    background: #0a0f1e;
    overflow: hidden;
    border-bottom: 1px solid #1e2d50;
    padding: 6px 0;
    margin-bottom: 0;
}
.ticker-track {
    display: inline-flex;
    white-space: nowrap;
    animation: ticker 40s linear infinite;
}
.ticker-item {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #ccd6f6;
    margin-right: 36px;
    font-family: 'Courier New', monospace;
}
.ticker-item .up   { color: #26a65b; }
.ticker-item .down { color: #e74c3c; }
.ticker-arrow { font-size: 0.6rem; margin-left: 2px; }

/* Price-per-share footer bar */
.price-bar {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 6px;
    padding: 8px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    color: #8892b0;
    margin-top: 4px;
}
.price-bar .price-val { font-weight: 700; color: #64ffda; }

/* ── Main area ───────────────────────────────────────────────────── */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-label {
    color: #8892b0;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-value { color: #ccd6f6; font-size: 1.5rem; font-weight: 700; }
.metric-value.green { color: #64ffda; }
.metric-value.red   { color: #ff6b6b; }
.metric-value.amber { color: #ffb347; }

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #ccd6f6;
    border-left: 3px solid #4c8bf5;
    padding-left: 10px;
    margin: 22px 0 10px;
}

/* Status pill colours inside dataframe */
</style>
""", unsafe_allow_html=True)


# ── Ticker tape ────────────────────────────────────────────────────────────────
TICKERS = [
    ("ATALAYA MINING",   -3.72), ("PARTNERS GRP E",  -3.61),
    ("ALLIANZ TECH",     -3.59), ("POLAR CAP",       -3.58),
    ("VESUVIUS",         -3.56), ("BABCOCK INTL",    +3.52),
    ("FIDELITY E.M.LD", -3.46), ("PACIFIC HORIZON",  -6.23),
    ("ANTOFAGASTA",     -6.12), ("GLENCORE",         -5.13),
    ("BUNZL",           +5.11), ("ANGLO AMERICAN",  -5.02),
    ("TEMPLETON EMRG",  -5.00), ("RASPBERRY PI",   -12.67),
]

def _ticker_html(tickers):
    items = ""
    for name, pct in tickers:
        cls   = "up" if pct > 0 else "down"
        arrow = "▲" if pct > 0 else "▼"
        items += (
            f'<span class="ticker-item">'
            f'{name} <span class="{cls}">{pct:+.2f}%'
            f'<span class="ticker-arrow">{arrow}</span></span>'
            f'</span>'
        )
    # duplicate for seamless loop
    return (
        f'<div class="ticker-wrap">'
        f'<div class="ticker-track">{items}{items}</div>'
        f'</div>'
    )

st.markdown(_ticker_html(TICKERS), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — exact TradeTutor layout
# ══════════════════════════════════════════════════════════════════════════════
INDICATOR_NAMES = list(INDICATOR_MAP.keys())
PRICE_OPTIONS   = ["Mid Price", "Ask Price", "Bid Price"]
DIR_OPTIONS     = ["Is Above", "Is Below"]

with st.sidebar:
    st.markdown("## Build Your Strategy")
    st.markdown("---")

    # ── File upload ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload trade file (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        help="Columns expected: Transaction Time, Mid/Ask/Bid Price, Moving Average, Position, Action",
    )

    st.markdown("---")
    st.markdown("**Build your Entry Strategy!**")

    # ── Indicator Type ───────────────────────────────────────────────────────
    indicator_name = st.selectbox(
        "Indicator Type ⓘ",
        INDICATOR_NAMES,
        index=0,
        help="Choose the technical indicator used to generate signals.",
    )
    st.markdown(
        f'<div style="font-size:0.78rem;color:#8892b0;margin-bottom:10px;line-height:1.4;">'
        f'{INDICATOR_HINTS[indicator_name]}</div>',
        unsafe_allow_html=True,
    )

    # ── Instrument (read from file or placeholder) ───────────────────────────
    instrument_display = "—"

    # ── Buy When ─────────────────────────────────────────────────────────────
    st.markdown("**Buy When** ⓘ")

    buy_price = st.selectbox(
        "Price column",
        PRICE_OPTIONS,
        index=0,
        key="buy_price",
        label_visibility="collapsed",
    )

    buy_dir_label = st.selectbox(
        "Buy Direction ⓘ",
        DIR_OPTIONS,
        index=0,
        key="buy_dir",
    )
    buy_direction = "above" if buy_dir_label == "Is Above" else "below"

    # "Is Above ⓘ" sub-label + indicator reference dropdown
    dir_label_text = buy_dir_label  # e.g. "Is Above ⓘ"
    st.markdown(
        f'<div style="font-size:0.82rem;font-weight:500;color:#444c6e;margin-top:6px;">'
        f'{buy_dir_label} ⓘ</div>',
        unsafe_allow_html=True,
    )
    buy_ref = st.selectbox(
        "Buy reference indicator",
        [indicator_name],
        key="buy_ref",
        label_visibility="collapsed",
        disabled=True,
    )

    buy_pct = st.number_input(
        "How Much (%) ⓘ",
        min_value=0.0, max_value=100.0,
        value=0.05, step=0.01, format="%.4f",
        key="buy_pct",
    )

    window = st.number_input(
        "Window (Periods) ⓘ",
        min_value=2, max_value=500, value=2, step=1,
    )

    buy_qty = st.number_input(
        "Buy Quantity ⓘ",
        min_value=1, value=500, step=1,
        key="buy_qty",
    )

    # Price per share bar (filled once file is loaded — placeholder for now)
    price_bar_placeholder = st.empty()

    st.markdown("---")

    # ── Sell When ─────────────────────────────────────────────────────────────
    st.markdown("**Sell When** ⓘ")

    sell_price = st.selectbox(
        "Price column (sell)",
        PRICE_OPTIONS,
        index=0,
        key="sell_price",
        label_visibility="collapsed",
    )

    sell_dir_label = st.selectbox(
        "Sell Direction ⓘ",
        ["Is Below", "Is Above"],
        index=0,
        key="sell_dir",
    )
    sell_direction = "below" if sell_dir_label == "Is Below" else "above"

    st.markdown(
        f'<div style="font-size:0.82rem;font-weight:500;color:#444c6e;margin-top:6px;">'
        f'{sell_dir_label} ⓘ</div>',
        unsafe_allow_html=True,
    )
    sell_ref = st.selectbox(
        "Sell reference indicator",
        [indicator_name],
        key="sell_ref",
        label_visibility="collapsed",
        disabled=True,
    )

    sell_pct = st.number_input(
        "How Much (%) ⓘ",
        min_value=0.0, max_value=100.0,
        value=0.05, step=0.01, format="%.4f",
        key="sell_pct",
    )

    sell_qty = st.number_input(
        "Sell Quantity ⓘ",
        min_value=1, value=500, step=1,
        key="sell_qty",
    )

    st.markdown("---")

    # ── Repeat Trade Flag ─────────────────────────────────────────────────────
    repeat_flag = st.toggle(
        "Repeat Trade Flag ⓘ",
        value=False,
        help=(
            "OFF — Strategy alternates: Buy → Sell → Buy → Sell. "
            "A new Buy is ignored until the previous position is closed with a Sell.\n\n"
            "ON — Strategy reacts to every signal. Consecutive Buys or Sells are allowed "
            "(e.g. Buy → Buy → Sell → Sell → Buy)."
        ),
    )

    st.markdown("---")
    run_btn = st.button("▶ Calculate", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Trading Signal Evaluator")
st.caption(
    "Upload a trade file and configure your strategy using the sidebar. "
    "Results include buy/sell signals, Status validation (Pass / Fail / N/A), and P&L."
)

if uploaded is None:
    st.info("👈  Upload an Excel or CSV file to get started.")
    with st.expander("Expected column format"):
        st.markdown("""
| Column | Required | Notes |
|---|---|---|
| `Transaction Time` | Recommended | Timestamp or Excel serial number |
| `Mid Price` / `Ask Price` / `Bid Price` | **Yes** | At least one price column |
| `Moving Average` | Optional | Pre-computed; will be recalculated |
| `Position` | Optional | `In` / `Out` |
| `Action` | Optional | `Buy` / `Sell` / `Hold` — used for **Status** comparison |
| `Symbol` | Optional | Instrument ticker |
""")
    st.stop()

# ── Load ───────────────────────────────────────────────────────────────────────
try:
    df_raw = load_file(uploaded)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

avail_prices = available_price_cols(df_raw)
if not avail_prices:
    st.error("No price column found (need Mid Price, Ask Price, or Bid Price).")
    st.stop()

# Resolve price column: use Buy selection if available, else first available
price_col = buy_price if buy_price in avail_prices else avail_prices[0]

# ── Instrument + price bar ─────────────────────────────────────────────────────
symbol    = df_raw["Symbol"].iloc[0] if "Symbol" in df_raw.columns else "—"
last_px   = df_raw[price_col].iloc[-1] if price_col in df_raw.columns else None
short_sym = symbol[:4].upper() if symbol != "—" else "—"

if last_px is not None:
    price_bar_placeholder.markdown(
        f'<div class="price-bar">'
        f'<span>Price per share ({short_sym})</span>'
        f'<span class="price-val">£{last_px:.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.success(
    f"Loaded **{len(df_raw):,}** rows  ·  Instrument: **{symbol}**  ·  Price column: **{price_col}**"
)

# ── Column validation ──────────────────────────────────────────────────────────
val = validate_sma_columns(df_raw, price_col)
with st.expander(
    f"📋 Column Validation — {len(val['present'])} present / {len(val['missing'])} missing",
    expanded=len(val["missing"]) > 0,
):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Present**")
        for col in val["present"]:
            st.markdown(
                f'<div style="background:#0d2b1f;border:1px solid #1a5c3a;border-radius:5px;'
                f'padding:5px 10px;margin-bottom:4px;font-size:0.82rem;color:#64ffda;">✓ {col}</div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("**❌ Missing**")
        for col in val["missing"]:
            note = " (Status → N/A)" if col == "Action" else ""
            st.markdown(
                f'<div style="background:#2b0d0d;border:1px solid #5c1a1a;border-radius:5px;'
                f'padding:5px 10px;margin-bottom:4px;font-size:0.82rem;color:#ff6b6b;">✗ {col}{note}</div>',
                unsafe_allow_html=True,
            )

# ── Calculate ──────────────────────────────────────────────────────────────────
if run_btn or True:
    try:
        df_result = build_result_df(
            df_raw, indicator_name, price_col,
            window, buy_pct, sell_pct, buy_direction, sell_direction,
            repeat_flag,
        )
    except Exception as e:
        st.error(f"Calculation error: {e}")
        st.stop()

    pnl = compute_pnl(df_result, price_col, buy_qty)

    # ── KPI cards ──────────────────────────────────────────────────────────────
    n_pass = (df_result["Status"] == "Pass").sum() if "Status" in df_result.columns else 0
    n_fail = (df_result["Status"] == "Fail").sum() if "Status" in df_result.columns else 0

    cols = st.columns(4)
    kpis = [
        ("Total Rows",    f"{len(df_result):,}",              ""),
        ("Indicator",     indicator_name.split()[0],           ""),
        ("Pass",          f"{n_pass}",
         "green" if n_pass > 0 else ""),
        ("Fail",          f"{n_fail}",
         "red" if n_fail > 0 else ""),
    ]
    for col, (lbl, val_str, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value {color}">{val_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Results table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Calculated Results</div>',
                unsafe_allow_html=True)

    # Build display column list dynamically
    always_first = ["Transaction Time", price_col]
    indicator_cols = [
        c for c in df_result.columns
        if c not in always_first
        and c not in ["S/N", "Symbol", "Position", "Moving Average"]
        and "Condition" not in c
    ]
    display_cols = [c for c in always_first + indicator_cols if c in df_result.columns]

    def _highlight(row):
        action = row.get("Action (calc)", "")
        status = row.get("Status", "")
        if status == "Fail":
            return ["background-color: #3a1a2e"] * len(row)
        if action == "Buy":
            return ["background-color: #112a1a"] * len(row)
        if action == "Sell":
            return ["background-color: #2a1111"] * len(row)
        return [""] * len(row)

    show_df = df_result[display_cols].copy()

    # Format numeric columns
    numeric_display = [
        c for c in show_df.columns
        if pd.api.types.is_float_dtype(show_df[c]) and "Condition" not in c
    ]
    for c in numeric_display:
        show_df[c] = show_df[c].map(
            lambda x: f"{x:.4f}" if pd.notna(x) else ""
        )

    MAX_CELLS = 262_144
    total_cells = show_df.shape[0] * show_df.shape[1]

    if total_cells <= MAX_CELLS:
        st.dataframe(
            show_df.style.apply(_highlight, axis=1),
            use_container_width=True,
            height=430,
        )
    else:
        st.caption(f"⚠️ Large dataset ({total_cells:,} cells) — highlighting disabled.")
        st.dataframe(show_df, use_container_width=True, height=430)

    # ── Trade log ──────────────────────────────────────────────────────────────
    if pnl["trades"] > 0:
        st.markdown('<div class="section-header">Round-Trip Trade Log</div>',
                    unsafe_allow_html=True)
        trades_df = pd.DataFrame({
            "#":          range(1, pnl["trades"] + 1),
            "Buy Price":  [f"{b:.4f}" for b in pnl["buys"]],
            "Sell Price": [f"{s:.4f}" for s in pnl["sells"]],
            "P&L (£)":    [f"{p:+.4f}" for p in pnl["pnl_list"]],
        })
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed round trips with the current settings.")

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

    try:
        wb = build_workbook(
            df_raw, indicator_name, price_col, window,
            buy_pct, sell_pct, buy_direction, sell_direction,
            repeat_flag,
        )
        buf = io.BytesIO()
        wb.save(buf)
        st.download_button(
            "⬇ Download Results (Excel — live formulas)",
            data=buf.getvalue(),
            file_name="trade_results_formulas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Excel export error: {e}")

    st.caption(
        "The Excel file recalculates live: every indicator value, threshold, "
        "Buy/Sell Condition, Position, Action, and Status cell is a real Excel "
        "formula (see the **Settings** sheet to tweak Window / % / Direction)."
    )