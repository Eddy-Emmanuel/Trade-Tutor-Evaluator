import io
import pandas as pd
from indicators.engine import run_indicator


# ── Status column ─────────────────────────────────────────────────────────────
def compute_status(df: pd.DataFrame) -> pd.Series:
    """Pass / Fail / N/A by comparing Action (calc) vs uploaded Action."""
    if "Action" not in df.columns:
        return pd.Series(["N/A"] * len(df), index=df.index)
    out = []
    for _, row in df.iterrows():
        uploaded = row.get("Action", None)
        calc     = row.get("Action (calc)", None)
        if pd.isna(uploaded) or str(uploaded).strip() == "":
            out.append("N/A")
        elif str(uploaded).strip().lower() == str(calc).strip().lower():
            out.append("Pass")
        else:
            out.append("Fail")
    return pd.Series(out, index=df.index)


# ── PnL ───────────────────────────────────────────────────────────────────────
def compute_pnl(df: pd.DataFrame, price_col: str, shares: int) -> dict:
    buys  = df[df["Action (calc)"] == "Buy" ][price_col].tolist()
    sells = df[df["Action (calc)"] == "Sell"][price_col].tolist()
    pairs = min(len(buys), len(sells))
    pnl_list = [(sells[i] - buys[i]) * shares for i in range(pairs)]
    total = sum(pnl_list)
    wins  = sum(1 for x in pnl_list if x > 0)
    return {
        "total": total, "trades": pairs,
        "wins": wins, "losses": pairs - wins,
        "pnl_list": pnl_list,
        "buys": buys[:pairs], "sells": sells[:pairs],
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
) -> pd.DataFrame:
    if price_col not in df_raw.columns:
        raise ValueError(f"Price column '{price_col}' not in data.")

    prices = df_raw[price_col]
    result = run_indicator(
        indicator_name, prices, window,
        buy_pct, sell_pct, buy_direction, sell_direction,
    )

    df = df_raw.copy()
    df[result["indicator_col"]] = result["indicator_vals"].round(7)
    for col_name, col_series in result["extra_cols"].items():
        df[col_name] = col_series if "Condition" in col_name else col_series.round(7)
    df["Position (calc)"] = result["position"]
    df["Action (calc)"]   = result["action"]
    df["Status"]          = compute_status(df)
    return df