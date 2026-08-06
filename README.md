# Financial Valuation Dashboard

An interactive equity valuation and scenario-analysis dashboard for exploring a company’s two-stage discounted cash flow (DCF), automatic or manual WACC, reverse DCF, sensitivity analysis, and comparable-company context.

Live application: [valuationdashboard-ryankelley.streamlit.app](https://valuationdashboard-ryankelley.streamlit.app/)

## Features

- Two-stage growth DCF by default, plus optional custom annual FCF forecasting with adjustable terminal growth, debt, cash, and shares outstanding.
- Automatic WACC using CAPM, estimated cost of debt, capital structure weights, and tax adjustment, plus a transparent manual mode.
- Projected and discounted annual free cash flows, enterprise value, equity value, implied share price, and terminal-value contribution.
- WACC/terminal-growth sensitivity heatmap and reverse DCF for the Stage 1 growth rate implied by the current price.
- Comparable-company table with normalized tickers, missing-value handling, and per-ticker failure isolation.
- Live company and financial statement data from Yahoo Finance with labeled fallbacks for important inputs.

## Valuation methodology

The model sums the latest four quarterly free cash flow values when available, or calculates operating cash flow less capital expenditures using alternate Yahoo Finance labels. The default two-stage method projects FCF through Stage 1 and Stage 2; optional custom annual FCF mode exists so analysts can model nonlinear cash-flow paths that constant growth rates cannot represent well. Both methods discount each forecast year and the Gordon-growth terminal value using WACC, then adjust enterprise value for debt and cash.

Automatic WACC uses CAPM for cost of equity and estimates pretax cost of debt from interest expense divided by debt when those fields are available. Missing or unstable inputs are labeled in the dashboard and may cause a clearly identified fallback.

## Technology stack

Python, Streamlit, yfinance, Pandas, NumPy, and Plotly.

## Project structure

~~~
app.py                    Streamlit layout, inputs, and presentation
valuation.py              Pure DCF, WACC, sensitivity, and reverse-DCF logic
data_service.py           Yahoo Finance retrieval and data normalization
tests/test_valuation.py   Deterministic model tests
~~~

## Local setup

~~~
git clone https://github.com/rkkelley/ValuationDashboard.git
cd ValuationDashboard
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
streamlit run app.py
~~~

## Testing

~~~
pytest -q
python -m pytest -q
python -m compileall .
~~~

The valuation tests are deterministic and do not require internet access. Live Yahoo Finance responses are intentionally not part of the test suite.

## Limitations and educational-use disclaimer

Yahoo Finance data may be delayed, revised, incomplete, or inconsistent between issuers. DCF outputs are highly sensitive to growth, WACC, terminal growth, and the quality of the underlying FCF data. Comparable-company multiples provide market context rather than a standalone valuation conclusion. This project is for educational and analytical use only and is not investment advice or a recommendation to buy or sell securities.
