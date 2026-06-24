import pandas as pd


def parse_timestamp(series: pd.Series) -> pd.Series:
    def _convert(v):
        if pd.isna(v):
            return pd.NaT
        try:
            fv = float(v)
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=fv)
        except (ValueError, TypeError):
            pass
        return pd.to_datetime(v, errors="coerce")
    return series.apply(_convert)


def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if cl in ("s/n", "sn", "s_n", "#"):            rename[c] = "S/N"
        elif cl == "symbol":                            rename[c] = "Symbol"
        elif "transaction" in cl and "time" in cl:     rename[c] = "Transaction Time"
        elif "ask" in cl and "price" in cl:             rename[c] = "Ask Price"
        elif "bid" in cl and "price" in cl:             rename[c] = "Bid Price"
        elif "mid" in cl and "price" in cl:             rename[c] = "Mid Price"
        elif "moving" in cl and "average" in cl:        rename[c] = "Moving Average"
        elif cl == "position":                          rename[c] = "Position"
        elif cl == "action":                            rename[c] = "Action"
    df = df.rename(columns=rename)

    keep = [c for c in [
        "S/N", "Symbol", "Transaction Time",
        "Ask Price", "Bid Price", "Mid Price",
        "Moving Average", "Position", "Action",
    ] if c in df.columns]
    df = df[keep].copy()

    if "Transaction Time" in df.columns:
        df["Transaction Time"] = parse_timestamp(df["Transaction Time"])

    for pc in ["Ask Price", "Bid Price", "Mid Price"]:
        if pc in df.columns:
            df[pc] = pd.to_numeric(df[pc], errors="coerce")

    price_cols = [c for c in ["Ask Price", "Bid Price", "Mid Price"] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how="all")

    return df.reset_index(drop=True)


def available_price_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["Mid Price", "Ask Price", "Bid Price"] if c in df.columns]


def validate_sma_columns(df: pd.DataFrame, price_col: str) -> dict:
    required = ["Transaction Time", price_col, "Moving Average", "Position", "Action"]
    return {
        "present": [c for c in required if c in df.columns],
        "missing": [c for c in required if c not in df.columns],
        "has_action": "Action" in df.columns,
    }