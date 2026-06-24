# Trade Model Evaluator

A Streamlit app for uploading trade data, evaluating technical indicator-based trading rules, and comparing calculated actions with uploaded trade actions.

## Features

- Upload `.xlsx`, `.xls`, or `.csv` trade files
- Normalize common columns such as `S/N`, `Symbol`, `Transaction Time`, and `Mid Price` / `Ask Price` / `Bid Price`
- Choose from multiple technical indicators, including SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, Fibonacci, Standard Deviation, ADX, and Heikin Ashi
- Configure buy and sell thresholds, price source, direction, and trade quantity
- Generate calculated position and action columns from the selected strategy
- Compare calculated actions with the uploaded `Action` column and show `Pass`, `Fail`, or `N/A` status
- Display KPI summaries and estimated P&L from completed Buy → Sell round-trip trades

## Project structure

- `main.py` — Streamlit application UI and layout
- `indicators/engine.py` — indicator implementations and signal generation
- `utils/file_loader.py` — file parsing and column normalization
- `utils/pipeline.py` — result dataframe creation, status evaluation, and P&L calculation

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
- `Symbol` — ticker or instrument
- `Transaction Time` — timestamp or Excel serial date
- `Mid Price`, `Ask Price`, or `Bid Price` — price data used for analysis
- `Moving Average` — optional pre-computed value
- `Position` — optional `In` / `Out`
- `Action` — optional `Buy` / `Sell` / `Hold`

## How it works

1. Load and clean the uploaded file.
2. Select an indicator and configure the buy/sell logic.
3. Compute indicator values and threshold-based conditions.
4. Generate calculated positions and actions.
5. Compare results with uploaded actions and display status and P&L.

## Notes

- `Transaction Time` values can be Excel serial dates or datetime strings.
- The app supports multiple price columns and will use the selected price column when available.
- P&L is estimated using completed Buy → Sell pairs and a configurable shares-per-trade value.
