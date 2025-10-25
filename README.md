# Financial Valuation Dashboard 📈📊

## Overview

This project is an interactive web application designed to perform comprehensive financial valuations of publicly traded companies. It automates key analysis tasks like Discounted Cash Flow (DCF) modeling and Comparable Company Analysis (Comps), providing insights into a stock's intrinsic value and market expectations.

The tool leverages Python libraries for data fetching and analysis, with a user-friendly interface built using Streamlit. It includes advanced features like automatic WACC calculation, sensitivity analysis, and reverse DCF to assess implied growth rates.

**Live Demo URL:** [(your_streamlit_app_url)](https://valuationdashboard-ryankelley.streamlit.app/)

---

## Key Features

* **Interactive Interface:** Built with Streamlit for easy input and visualization.
* **Data Fetching:** Pulls real-time stock data, company info, and financials using `yfinance`.
* **2-Stage DCF Model:** Performs a 10-year Discounted Cash Flow valuation with user-adjustable growth assumptions (Years 1-5 and 6-10) and a terminal value.
* **WACC Calculation:**
    * **Automatic Mode:** Calculates Weighted Average Cost of Capital (WACC) using CAPM (Cost of Equity based on Beta, Risk-Free Rate, ERP) and Cost of Debt (estimated from financials).
    * **Manual Mode:** Allows users to input their own WACC.
* **Comparable Company Analysis (Comps):** Displays key financial metrics and valuation multiples (P/E, P/S, EV/Revenue, EV/EBITDA, Margins, Growth) for the target company and user-defined peers in a highlighted table.
* **Sensitivity Analysis:** Visualizes how the Implied Share Price changes based on variations in WACC and the Terminal Growth Rate using an interactive Plotly heatmap.
* **Reverse DCF:** Calculates the implied short-term (Years 1-5) FCF growth rate required to justify the current market stock price, given the other DCF assumptions.
* **Robust Data Handling:** Includes error checks and fallbacks for missing or inconsistent data from the `yfinance` API.

---

## Technologies Used

* **Python:** Core programming language.
* **Streamlit:** Web application framework.
* **yfinance:** API for fetching stock data from Yahoo Finance.
* **Pandas:** Data manipulation and analysis.
* **Plotly:** Interactive visualizations (Sensitivity Analysis heatmap).
* **NumPy / math:** Numerical operations and handling special values (NaN/inf).

---

## Demo

*(Replace the image link with your actual GIF)*
![Financial Valuation Dashboard Demo](link_to_your_gif.gif)

A brief walkthrough showing the main features: entering a ticker, adjusting DCF assumptions, viewing the Comps table, and interpreting the Sensitivity Analysis and Reverse DCF.

---

## Installation & Usage

To run this application locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-github-username>/<repo-name>.git
    cd <repo-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure your `requirements.txt` file lists `streamlit`, `yfinance`, `pandas`, `plotly`, `numpy`)*
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py 
    ```
    The application will open in your default web browser.

---

## (Optional) Future Enhancements

* Integration with a more robust financial data API (e.g., FMP, EODHD) for greater data reliability.
* Addition of historical financial trends charts (Revenue, EPS, FCF over time).
* Monte Carlo simulation for DCF outputs.
* Saving/loading user assumptions.
* More sophisticated fallback logic for missing data points.
