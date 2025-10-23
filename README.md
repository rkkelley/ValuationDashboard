# Valuation Dashboard

## Overview
AI-assisted interactive financial dashboard performing **Discounted Cash Flow (DCF) valuations** and **Comparable Company Analysis (Comps)** for any stock.
Built with **Python**, **Streamlit**, and **yfinance**. Designed to be **demo-ready**, reusable, and fully interactive.

## Features
- Pulls financial statement data for any stock ticker
- Performs simple DCF valuation with user inputs for revenue growth and WACC
- Displays comparable company metrics: P/E, P/S, EV/EBITDA
- Interactive sidebar inputs for ticker and assumptions
- Demo-ready web interface via Streamlit
- Optional: expandable sections for charts and AI-generated valuation summaries

## Installation
1. Clone this repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Install dependencies:

pip install -r requirements.txt


3. Run the app:

streamlit run app.py
