# utils/pipeline.py

from datetime import date, time

import pandas as pd

from indicators.engine import run_indicator


# ── Date / time window ────────────────────────────────────────────────────────
def filter_by_window(
    df: pd.DataFrame,
    date_from: date | None = None,
    date_to: date | None = None,
    start_time: time | None = None,
    end_time: time | None = None,
    time_col: str = "Transaction Time",
) -> pd.DataFrame:
    """Trim rows to the Date From/To and Start/End Time window from the sidebar.

    Dates are inclusive on both ends. The time filter is applied per row
    (a session window such as 08:00–16:30), not as a single continuous span,
    which is what an intraday trading session means.
    """
    if time_col not in df.columns:
        return df

    ts = pd.to_datetime(df[time_col], errors="coerce")
    mask = ts.notna()

    if date_from is not None:
        mask &= ts.dt.date >= date_from
    if date_to is not None:
        mask &= ts.dt.date <= date_to
    if start_time is not None:
        mask &= ts.dt.time >= start_time
    if end_time is not None:
        mask &= ts.dt.time <= end_time

    return df[mask].reset_index(drop=True)


# ── Status column ─────────────────────────────────────────────────────────────
def compute_status(df: pd.DataFrame) -> pd.Series:
    """Pass / Fail / N/A by comparing Action (calc) vs uploaded Action."""
    if "Action" not in df.columns:
        return pd.Series(["N/A"] * len(df), index=df.index)
    out = []
    for _, row in df.iterrows():
        uploaded = row.get("Action", None)
        calc = row.get("Action (calc)", None)
        if pd.isna(uploaded) or str(uploaded).strip() == "":
            out.append("N/A")
        elif str(uploaded).strip().lower() == str(calc).strip().lower():
            out.append("Pass")
        else:
            out.append("Fail")
    return pd.Series(out, index=df.index)


# ── PnL ───────────────────────────────────────────────────────────────────────
def compute_pnl(
    df: pd.DataFrame,
    price_col: str,
    buy_qty: int,
    sell_qty: int | None = None,
    sell_price_col: str | None = None,
) -> dict:
    """Round-trip P&L.

    sell_qty defaults to buy_qty (the old behaviour, where Sell Quantity was
    collected in the sidebar and then silently ignored). sell_price_col lets
    the exit be marked against a different price column than the entry, e.g.
    buy on Ask, sell on Bid.
    """
    sell_qty = buy_qty if sell_qty is None else sell_qty
    sell_price_col = sell_price_col or price_col

    buys = df.loc[df["Action (calc)"] == "Buy", price_col].tolist()
    sells = df.loc[df["Action (calc)"] == "Sell", sell_price_col].tolist()
    pairs = min(len(buys), len(sells))

    pnl_list = [sells[i] * sell_qty - buys[i] * buy_qty for i in range(pairs)]
    total = sum(pnl_list)
    wins = sum(1 for x in pnl_list if x > 0)

    return {
        "total": total,
        "trades": pairs,
        "wins": wins,
        "losses": pairs - wins,
        "pnl_list": pnl_list,
        "buys": buys[:pairs],
        "sells": sells[:pairs],
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
    }


# ── Build result dataframe ────────────────────────────────────────────────────
def build_result_df(
    df_raw: pd.DataFrame,
    indicator_name: str,
    price_col: str,
    window: int,
    buy_pct: float,
    sell_pct: float,
    buy_direction: str,
    sell_direction: str,
    repeat_flag: bool = False,
    params: dict | None = None,
) -> pd.DataFrame:
    if price_col not in df_raw.columns:
        raise ValueError(f"Price column '{price_col}' not in data.")

    prices = df_raw[price_col]

    result = run_indicator(
        indicator_name=indicator_name,
        prices=prices,
        window=window,
        buy_pct=buy_pct,
        sell_pct=sell_pct,
        buy_direction=buy_direction,
        sell_direction=sell_direction,
        repeat=repeat_flag,
        params=params,
    )

    df = df_raw.copy()
    df[result["indicator_col"]] = result["indicator_vals"]

    for col_name, col_series in result["extra_cols"].items():
        df[col_name] = col_series

    df["Position (calc)"] = result["position"]
    df["Action (calc)"] = result["action"]
    df["Status"] = compute_status(df)

    # Leading rows where the lookback is not yet satisfied. These are held at
    # Out/Hold by the engine; the UI reports the count so a run that produces
    # no trades because the window swallowed the data is obvious.
    df.attrs["warmup"] = result.get("warmup", 0)
    return df