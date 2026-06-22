# SMA Trading Signal Calculator

A Streamlit app for loading trade data, calculating Simple Moving Average (SMA) buy/sell thresholds, and producing trading signals with an Excel export.

## Features

- Upload `.xlsx`, `.xls`, or `.csv` trade files
- Normalize common columns such as `S/N`, `Symbol`, `Transaction Time`, and `Mid Price`
- Recompute SMA, buy threshold, sell threshold, position state, and buy/sell/hold actions
- Display calculated results in a styled Streamlit dashboard
- Estimate P&L using completed buy/sell round-trip trades
- Download an Excel workbook with live formulas for SMA and trading signals

## Requirements

- Python 3.8+
- `streamlit`
- `pandas`
- `numpy`
- `openpyxl`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run the app

From the project folder:

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal.

## Expected input columns

The app will normalize and use the following columns if present:

- `S/N` — row number
- `Symbol` — ticker / instrument
- `Transaction Time` — timestamp or Excel serial date
- `Mid Price` — price used for SMA and signals
- `Moving Average` — optional pre-computed value (will be recalculated)
- `Position` — optional `In` / `Out`
- `Action` — optional `Buy` / `Sell` / `Hold`

## How it works

1. Load the uploaded file and clean the data.
2. Compute the SMA for the chosen window.
3. Create buy/sell thresholds based on percentage offsets.
4. Generate a position state machine and trade actions.
5. Show KPIs, signals, and round-trip trades.
6. Export an Excel file with formulas so the model can be reviewed live in Excel.

## Notes

- The app currently auto-runs calculations on upload.
- `Transaction Time` values can be Excel serial dates or datetime strings.
- P&L is estimated using completed Buy→Sell pairs and a configurable shares-per-trade value.
