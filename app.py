import streamlit as st
import yfinance as yf
import pandas as pd
import math
import numpy as np # For infinity check
import plotly.express as px
import plotly.graph_objects as go


# --- Page Configuration ---
st.set_page_config(
    page_title="Valuation Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Robust Helper Function (Catches str, None, nan, inf) ---
def safe_float(value):
    """Converts a value to float, returning 0.0 if it fails or is inf/nan."""
    if value is None:
        return 0.0
    try:
        f_val = float(value)
        # Check if the value is 'Not a Number' (nan) or infinity
        if math.isnan(f_val) or math.isinf(f_val):
            return 0.0
        return f_val
    except (ValueError, TypeError, AttributeError):
        return 0.0
    

def format_number(value, decimals=2, is_dollar=True, is_shares=False):
    """Formats a number safely with commas and specified decimals."""
    cleaned_value = safe_float(value) # Use safe_float first

    # Handle zero/invalid values
    if cleaned_value == 0.0 and (value != 0 or not isinstance(value, (int, float))):
        return "N/A"
    if cleaned_value == 0.0 and value == 0:
        return "$0.00" if is_dollar else ("0" if is_shares else "0.00")


    try:
        if is_shares:
            # Format shares as integer with commas
            return f"{cleaned_value:,.0f}"
        else:
            # Format currency or general number
            prefix = "$" if is_dollar else ""
            return f"{prefix}{cleaned_value:,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A (Format Error)"

# --- Helper Function for Comps (Uses safe_float) ---
@st.cache_data(ttl=600)
def get_comps_data(tickers):
    """Pulls key metrics for a list of competitor tickers."""
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            metrics = {
                "Ticker": ticker,
                "Company Name": info.get('shortName', 'N/A'),
                "Market Cap (B)": safe_float(info.get('marketCap')) / 1e9,
                "Revenue Growth (TTM)": safe_float(info.get('revenueGrowth')),
                "Gross Margin (TTM)": safe_float(info.get('grossMargins')),
                "EBITDA Margin (TTM)": safe_float(info.get('ebitdaMargins')),
                "P/E (Forward)": safe_float(info.get('forwardPE')),
                "P/S (TTM)": safe_float(info.get('priceToSalesTrailing12Months')), # Corrected P/S
                "EV/Revenue (TTM)": safe_float(info.get('enterpriseToRevenue')),
                "EV/EBITDA (TTM)": safe_float(info.get('enterpriseToEbitda'))
            }
            data.append(metrics)
        except Exception as e:
            st.warning(f"Could not pull data for {ticker}. Skipping. Error: {e}")

    column_order = [
        "Company Name", "Market Cap (B)",
        "Revenue Growth (TTM)", "Gross Margin (TTM)", "EBITDA Margin (TTM)",
        "P/E (Forward)", "P/S (TTM)", "EV/Revenue (TTM)", "EV/EBITDA (TTM)"
    ]

    df = pd.DataFrame(data).set_index("Ticker")
    df = df.reindex(columns=column_order, fill_value=0.0)
    return df

# --- Helper Function for WACC (Uses safe_float) ---
# --- Helper Function for WACC (Uses safe_float AND better Rd fallback) ---
# @st.cache_data(ttl=600)
def get_wacc_components(_stock, _financials, _balance_sheet, rf_rate, erp, tax_rate):
    """Calculates the components of WACC with improved Rd fallback."""
    info = _stock.info

    beta = safe_float(info.get('beta', 1.0))
    if beta == 0.0: beta = 1.0
    cost_of_equity = rf_rate + beta * erp

    # --- Improved Cost of Debt (Rd) Calculation ---
    total_debt = 0.0
    # Try multiple keys for debt
    if 'Total Debt' in _balance_sheet.index:
        total_debt = safe_float(_balance_sheet.loc['Total Debt'].iloc[0])
    elif 'longTermDebt' in info and info['longTermDebt'] is not None:
         total_debt = safe_float(info['longTermDebt'])
    elif 'Long Term Debt' in _balance_sheet.index:
         total_debt = safe_float(_balance_sheet.loc['Long Term Debt'].iloc[0])

    interest_expense = 0.0
    # Try multiple keys for interest expense
    if 'Interest Expense' in _financials.index:
        interest_expense = abs(safe_float(_financials.loc['Interest Expense'].iloc[0]))
    elif 'interestExpense' in info and info['interestExpense'] is not None:
         interest_expense = abs(safe_float(info['interestExpense']))

    # Default Rd assumption: Risk-Free Rate + 1.5% Spread
    cost_of_debt_fallback = rf_rate + 0.015
    cost_of_debt = cost_of_debt_fallback # Start with fallback

    if total_debt > 0 and interest_expense > 0:
        try:
            calculated_rd = interest_expense / total_debt
            calculated_rd = min(calculated_rd, 0.20) # Cap at 20%

            # **NEW FALLBACK LOGIC:** Only use calculated Rd if it's reasonable (>= rf_rate)
            if calculated_rd >= rf_rate:
                cost_of_debt = calculated_rd
            else:
                 # If calculated Rd is too low, keep the fallback but log a warning (optional)
                 print(f"Warning: Calculated Rd ({calculated_rd:.2%}) for {info.get('symbol', '')} is below Rf ({rf_rate:.2%}). Using fallback Rd {cost_of_debt_fallback:.2%}.")
                 cost_of_debt = cost_of_debt_fallback # Explicitly ensure fallback is used

        except ZeroDivisionError:
             pass # Keep the initial fallback if division fails
    # --- End Improved Rd Calculation ---


    market_cap = safe_float(info.get('marketCap'))
    total_value = market_cap + total_debt

    if total_value == 0.0:
        weight_equity = 1.0
        weight_debt = 0.0
    else:
        weight_equity = market_cap / total_value
        weight_debt = total_debt / total_value

    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    wacc = max(0.01, min(wacc, 0.30)) # Ensure WACC is within bounds

    return wacc, cost_of_equity, cost_of_debt, beta, rf_rate, erp


# --- NEW: Helper Function for Sensitivity Analysis ---
def calculate_dcf_price(fcf, g_rate_1, g_rate_2, t_rate, wacc, shares_outstanding, total_debt, cash):
    """Recalculates DCF Implied Price for given assumptions."""
    try:
        # --- Basic Validations ---
        if any(pd.isna(x) or x <= 0 for x in [fcf, shares_outstanding]) or wacc <= 0:
            return np.nan

        # Prevent unrealistic scenario: WACC too close or below terminal growth
        if wacc - t_rate < 0.005:  # at least 0.5% spread
            return np.nan

        # --- 1. Project FCF for 10 years ---
        projected_fcf = []
        fcf_year = fcf
        for year in range(1, 11):
            # First 5 years: higher growth, next 5 years: moderate growth
            growth = g_rate_1 if year <= 5 else g_rate_2
            fcf_year *= (1 + growth)
            projected_fcf.append(fcf_year)

        # --- 2. Calculate Terminal Value ---
        terminal_value = projected_fcf[-1] * (1 + t_rate) / (wacc - t_rate)

        # --- 3. Discount All Cash Flows ---
        discounted_fcf = [
            projected_fcf[i] / ((1 + wacc) ** (i + 1)) for i in range(10)
        ]
        discounted_tv = terminal_value / ((1 + wacc) ** 10)

        # --- 4. Enterprise Value ---
        enterprise_value = sum(discounted_fcf) + discounted_tv

        # --- 5. Equity Value ---
        equity_value = enterprise_value - total_debt + cash

        # --- 6. Implied Price ---
        implied_price = equity_value / shares_outstanding

        # --- 7. Cap unrealistic results ---
        # if implied_price > 500:
        #     implied_price = np.nan
        return safe_float(implied_price)
    
    except Exception as e:
        print(f"Error in calculate_dcf_price: {e}")
        return np.nan
    

@st.cache_data(ttl=600)
# --- Reverse DCF Helper Function ---
def reverse_dcf(current_price, fcf, g_rate_2, t_rate, wacc, shares_outstanding, total_debt, cash, years=5):
    """
    Calculate the implied short-term growth rate (g_rate_1) required to justify current_price.
    Uses binary search.
    """
    # Ensure inputs are valid floats
    current_price = safe_float(current_price)
    fcf = safe_float(fcf)
    g_rate_2 = safe_float(g_rate_2)
    t_rate = safe_float(t_rate)
    wacc = safe_float(wacc)
    shares_outstanding = safe_float(shares_outstanding)
    total_debt = safe_float(total_debt)
    cash = safe_float(cash)

    # Basic validation
    if fcf <= 0 or shares_outstanding <= 0 or wacc <= t_rate or current_price <= 0:
        print(f"Reverse DCF Input Validation Failed: fcf={fcf}, shares={shares_outstanding}, wacc={wacc}, t_rate={t_rate}, price={current_price}")
        return None # Return None for invalid base inputs

    target_ev = current_price * shares_outstanding + total_debt - cash
    if target_ev <= 0: # Target EV must be positive for binary search to work logically
        print(f"Reverse DCF Target EV Calculation Failed or Non-positive: target_ev={target_ev}")
        return None


    # Binary search for g_rate_1
    low, high = -0.50, 2.00  # Allow negative growth up to 200% growth
    implied_rate = None

    for iteration in range(100):  # Max 100 iterations for convergence
        mid = (low + high) / 2

        # --- Recalculate EV based on 'mid' as g_rate_1 ---
        try: # Add try-except for potential math errors in projection
            projected_fcf = []
            last_fcf = fcf
            # Stage 1: High growth using 'mid' rate
            for _ in range(years):
                last_fcf *= (1 + mid)
                projected_fcf.append(last_fcf)
            # Stage 2: Stable growth using g_rate_2
            for _ in range(years): # Years 6-10
                last_fcf *= (1 + g_rate_2)
                projected_fcf.append(last_fcf)

            # Clean projected FCF list and check validity
            projected_fcf = [safe_float(val) for val in projected_fcf]
            if not projected_fcf or projected_fcf[-1] == 0 or any(math.isinf(f) or math.isnan(f) for f in projected_fcf):
                 # If projection fails or leads to non-finite numbers, adjust search range
                 # print(f"Iter {iteration}: Invalid FCF projection for mid={mid:.4f}. Adjusting range.")
                 if mid > 0 : high = mid # If positive growth failed, try lower
                 else: low = mid      # If negative growth failed, try higher (less negative)
                 continue # Skip rest of loop for this iteration


            terminal_value = (projected_fcf[-1] * (1 + t_rate)) / (wacc - t_rate)
            terminal_value = safe_float(terminal_value) # Clean (handles potential inf/nan)

            discounted_values = [fcf_year / (1 + wacc)**(i + 1) for i, fcf_year in enumerate(projected_fcf)]
            discounted_values = [safe_float(val) for val in discounted_values] # Clean

            discounted_terminal_value = terminal_value / (1 + wacc)**(10) # Discount over 10 years
            discounted_terminal_value = safe_float(discounted_terminal_value) # Clean

            enterprise_value = sum(discounted_values) + discounted_terminal_value
            enterprise_value = safe_float(enterprise_value) # Clean

            # If EV calculation results in 0 (e.g., due to extreme safe_float cleaning), treat as failure
            if enterprise_value == 0:
                # print(f"Iter {iteration}: Calculated EV is zero for mid={mid:.4f}. Adjusting range.")
                # This likely means the growth rate 'mid' combined with wacc leads to issues
                # Adjust based on whether we are trying to increase or decrease EV
                if low < mid : high = mid # If EV was likely too high and got cleaned to 0, reduce upper bound
                else: low = mid        # If EV was likely too low/negative, increase lower bound
                continue


        except (OverflowError, ValueError, ZeroDivisionError) as e_calc:
             # print(f"Iter {iteration}: Calculation Error for mid={mid:.4f}: {e_calc}. Adjusting range.")
             # If calculation fails, assume 'mid' growth rate was too extreme
             if mid > 0 : high = mid # Try lower rates
             else: low = mid      # Try higher rates (less negative)
             continue # Skip comparison for this iteration


        # --- Binary Search Logic ---
        # Check if calculated EV is close enough to target EV
        tolerance = target_ev * 0.001 # 0.1% tolerance
        if abs(enterprise_value - target_ev) < tolerance:
            implied_rate = mid
            # print(f"Converged at iteration {iteration} with rate {implied_rate:.4f}")
            break
        elif enterprise_value < target_ev:
            low = mid # Need higher growth, increase lower bound
        else:
            high = mid # Need lower growth, decrease upper bound

        # Break if range is too small (indicates convergence or unable to converge)
        if abs(high - low) < 1e-6:
             implied_rate = mid # Converged enough, or best estimate
             # print(f"Range too small at iteration {iteration}. Best rate {implied_rate:.4f}")
             break

    # Final check on the validity of the found rate
    if implied_rate is not None and -0.50 <= implied_rate <= 2.00:
        return implied_rate
    else:
        # print(f"Failed to converge or result out of bounds. Final mid: {mid}, low: {low}, high: {high}")
        return None # Return None if no suitable rate found


# --- Sidebar ---
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
run_button = st.sidebar.button("Run Analysis")

# --- DCF Assumptions ---
st.sidebar.subheader("DCF Assumptions")
st.sidebar.markdown("""
* **High-growth (e.g., NVDA):** Use high short-term growth (20-50%).
* **Mature (e.g., XOM):** Use low, stable growth (2-8%).
""")

g_rate_1_percent = st.sidebar.slider(
    "Growth Rate (Years 1-5):", 1, 50, 30, format="%d%%",
    help="The expected FCF growth rate for the first 5 years."
)
g_rate_2_percent = st.sidebar.slider(
    "Growth Rate (Years 6-10):", 1, 20, 10, format="%d%%",
    help="The FCF growth rate for the next 5 years (stable growth phase)."
)
t_rate_percent = st.sidebar.slider(
    "Perpetual Growth Rate (Terminal):", 1, 5, 2, format="%d%%",
    help="The long-term growth rate of FCF after Year 10."
)

# --- WACC Calculation Section ---
st.sidebar.subheader("Discount Rate (WACC)")
wacc_mode = st.sidebar.radio("WACC Mode", ["Automatic", "Manual"], index=0, help="Choose 'Automatic' to calculate WACC based on market data (CAPM) or 'Manual' to set it yourself.")

wacc_percent = 0.0 # Initialize

if wacc_mode == "Manual":
    wacc_percent = st.sidebar.slider(
        "Manual WACC:", 1, 30, 10, format="%d%%",
        help="Manually set the Weighted Average Cost of Capital."
    )
else:
    st.sidebar.markdown("**WACC Auto-Calculation Inputs:**")
    erp_percent = st.sidebar.slider(
        "Equity Risk Premium (ERP):", 3.0, 10.0, 5.5, step=0.1, format="%.1f%%",
        help="The excess return that investing in the stock market provides over a risk-free rate. (Default: 5.5%)"
    )
    tax_rate_percent = st.sidebar.slider(
        "Effective Tax Rate:", 0, 50, 21, format="%d%%",
        help="The company's effective corporate tax rate. (Default: 21%)"
    )

# --- Comps Tickers Input ---
st.sidebar.subheader("Comparable Companies")
peers_input = st.sidebar.text_input(
    "Competitor Tickers (comma-separated):", "MSFT, GOOG, META, AMZN"
)

# --- Main Page Title ---
st.title(f"Financial Valuation Dashboard: {ticker}")

# --- App Logic ---
if run_button:

    g_rate_1 = g_rate_1_percent / 100.0
    g_rate_2 = g_rate_2_percent / 100.0
    t_rate = t_rate_percent / 100.0

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        try: financials = stock.financials
        except Exception: financials = pd.DataFrame()
        try: balance_sheet = stock.balance_sheet
        except Exception: balance_sheet = pd.DataFrame()
        try: cash_flow = stock.cashflow
        except Exception: cash_flow = pd.DataFrame()


        # --- WACC Calculation Logic ---
        wacc = 0.0
        re = 0.0; rd = 0.0; beta = 0.0; rf = 0.0; erp_val = 0.0

        if wacc_mode == "Manual":
            wacc = wacc_percent / 100.0
        else:
            try:
                rf_stock = yf.Ticker("^TNX")
                rf_rate_raw = rf_stock.info.get('regularMarketPrice') or rf_stock.info.get('previousClose')
                rf_rate = safe_float(rf_rate_raw) / 100.0
                if rf_rate == 0.0: rf_rate = 0.04
            except Exception:
                rf_rate = 0.04

            erp = erp_percent / 100.0
            tax_rate = tax_rate_percent / 100.0

            if not financials.empty and not balance_sheet.empty:
                 wacc, re, rd, beta, rf, erp_val = get_wacc_components(stock, financials, balance_sheet, rf_rate, erp, tax_rate)
            else:
                 st.warning("Could not retrieve necessary financial data for WACC calculation. Using fallback WACC of 10%.")
                 wacc = 0.10

            with st.sidebar.expander("View Auto-WACC Details", expanded=True):
                st.write(f"Risk-Free Rate (10-Yr): {rf_rate:.2%}") # Use rf_rate directly
                st.write(f"Equity Risk Premium: {erp:.2%}") # Use erp directly
                st.write(f"Company Beta: {beta:.2f}")
                st.write(f"Cost of Equity (Re): {re:.2%}")
                st.write(f"Cost of Debt (Rd): {rd:.2%}")
                st.write(f"Effective Tax Rate: {tax_rate:.2%}")
                st.markdown(f"**Calculated WACC: {wacc:.2%}**")


        # Display Basic Info
        st.subheader(f"Company Profile: {info.get('longName', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Website:** {info.get('website', 'N/A')}")

        with st.expander("View Company Summary"):
            st.write(info.get('longBusinessSummary', 'No summary available.'))

        # --- DCF Valuation ---
        st.subheader("Discounted Cash Flow (DCF) Valuation")
        st.write(f"Using Discount Rate (WACC): **{wacc:.2%}**")

        # Get FCF
        fcf = 0.0
        try:
            fcf = safe_float(info.get('freeCashflow'))
            if fcf == 0 and not cash_flow.empty:
                op_cash = 0.0
                if 'Total Cash From Operating Activities' in cash_flow.index:
                    op_cash = safe_float(cash_flow.loc['Total Cash From Operating Activities'].iloc[0])

                cap_ex = 0.0
                if 'Capital Expenditures' in cash_flow.index:
                    cap_ex = safe_float(cash_flow.loc['Capital Expenditures'].iloc[0])

                fcf = op_cash + cap_ex

            if fcf == 0:
                 st.warning("Could not find TTM Free Cash Flow data. DCF result will be zero.")

        except Exception as e:
            st.error(f"Error getting FCF: {e}")
            fcf = 0

        # Initialize DCF variables outside the 'if fcf > 0' block
        projected_fcf = [0.0] * 10
        discounted_values = [0.0] * 10
        terminal_value = 0.0
        discounted_terminal_value = 0.0
        enterprise_value = 0.0
        equity_value = 0.0
        implied_share_price = 0.0

        if fcf > 0:
            # 1. Project FCF
            temp_projected_fcf = [] # Use a temporary list
            last_fcf = fcf
            for i in range(1, 6): last_fcf *= (1 + g_rate_1); temp_projected_fcf.append(last_fcf)
            for i in range(1, 6): last_fcf *= (1 + g_rate_2); temp_projected_fcf.append(last_fcf)
            # --- NEW: Clean projected_fcf list ---
            projected_fcf = [safe_float(val) for val in temp_projected_fcf]


            # 2. Calculate Terminal Value
            if wacc <= t_rate:
                st.error("WACC must be greater than the Perpetual Growth Rate. DCF cannot be calculated.")
                # Keep values at 0
            else:
                # Ensure last projected FCF is valid before calculating terminal value
                last_valid_fcf = projected_fcf[-1] if projected_fcf else 0.0
                if last_valid_fcf > 0:
                     terminal_value = (last_valid_fcf * (1 + t_rate)) / (wacc - t_rate)
                     terminal_value = safe_float(terminal_value) # Clean terminal value

                     # 3. Discount FCF and Terminal Value
                     temp_discounted_values = [fcf_year / (1 + wacc)**(i + 1) for i, fcf_year in enumerate(projected_fcf)]
                     # --- NEW: Clean discounted_values list ---
                     discounted_values = [safe_float(val) for val in temp_discounted_values]

                     discounted_terminal_value = terminal_value / (1 + wacc)**10
                     discounted_terminal_value = safe_float(discounted_terminal_value) # Clean dTV

                     # 4. Calculate Enterprise Value
                     enterprise_value = sum(discounted_values) + discounted_terminal_value
                     enterprise_value = safe_float(enterprise_value) # Clean EV
                else:
                    st.error("Could not project FCF. DCF cannot be calculated.")


            # 5. Calculate Equity Value and Implied Share Price
            try:
                # Ensure base values are floats
                total_debt = safe_float(info.get('totalDebt'))
                cash = safe_float(info.get('totalCash'))
                shares_outstanding = safe_float(info.get('sharesOutstanding'))
                current_price = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))

                st.write(f"Using TTM FCF: **${safe_float(fcf):,.2f}**") # Added safe_float here too

                col1, col2 = st.columns(2)

                equity_value = 0.0
                implied_share_price = 0.0

                # Ensure enterprise_value exists and is calculated
                if 'enterprise_value' not in locals():
                     enterprise_value = 0.0 # Initialize if WACC condition failed

                if shares_outstanding > 0 and enterprise_value > 0 : # Check enterprise_value
                    equity_value = enterprise_value - total_debt + cash
                    equity_value = safe_float(equity_value) # Clean Equity Value
                    implied_share_price = equity_value / shares_outstanding
                    implied_share_price = safe_float(implied_share_price) # Clean Implied Price

                    with col1:
                        st.success(f"Implied Share Price: **${implied_share_price:,.2f}**")
                    with col2:
                        st.info(f"Current Share Price: **${current_price:,.2f}**")

                    if implied_share_price > 0 and current_price > 0:
                        if implied_share_price > current_price:
                            st.write(f"**Result:** Based on this 2-stage DCF, the stock appears **Undervalued**.")
                        else:
                            st.write(f"**Result:** Based on this 2-stage DCF, the stock appears **Overvalued**.")
                    else:
                         st.warning("Cannot determine valuation due to missing price data or invalid DCF result.")

                else:
                    st.warning("Could not calculate implied share price (missing shares or invalid DCF inputs).")
                    with col1:
                        st.error("Implied Share Price: N/A")
                    with col2:
                        st.info(f"Current Share Price: **${current_price:,.2f}**")


                # --- Display Calculation Steps Expander ---
                with st.expander("View DCF Calculation Steps"):
                    try:
                        # --- Create df_fcf INSIDE the try block AFTER values are calculated ---
                         df_fcf_display = pd.DataFrame({
                             'Year': [f"Year {i+1}" for i in range(10)],
                             'Projected FCF': projected_fcf, # Use cleaned list
                             'Discounted FCF': discounted_values # Use cleaned list
                         })

                         # --- Apply formatting safely column by column using format_number ---
                         formatted_df = df_fcf_display.copy()
                         for col in ['Projected FCF', 'Discounted FCF']:
                             if col in formatted_df.columns:
                                 # Use format_number for DataFrame columns
                                 formatted_df[col] = formatted_df[col].apply(lambda x: format_number(x, decimals=2, is_dollar=True))

                         st.dataframe(formatted_df) # Display the manually formatted DataFrame

                         # --- Display other values safely using format_number ---
                         st.write(f"Terminal Value (at Year 10): **{format_number(terminal_value, decimals=2, is_dollar=True)}**")
                         st.write(f"Discounted Terminal Value: **{format_number(discounted_terminal_value, decimals=2, is_dollar=True)}**")
                         st.write(f"Enterprise Value: **{format_number(enterprise_value, decimals=2, is_dollar=True)}**")
                         st.write(f"Equity Value: **{format_number(equity_value, decimals=2, is_dollar=True)}** (EV - Debt + Cash)")
                         st.write(f"Shares Outstanding: **{format_number(shares_outstanding, decimals=0, is_dollar=False, is_shares=True)}**") # Format shares

                    except Exception as e_expander:
                         st.error(f"Error displaying calculation steps: {e_expander}")

            except Exception as e_price:
                st.error(f"Error during final price display section. Debug: {e_price}")

        # --- This runs if FCF <= 0 ---
        else:
             st.warning("Cannot run DCF: Most recent Free Cash Flow is zero or negative.")
             # Display empty expander to avoid errors
             with st.expander("View DCF Calculation Steps"):
                 st.write("DCF calculation skipped due to zero or negative FCF.")


        # --- Comparable Company Analysis ---
        st.subheader("Comparable Company Analysis (Comps)")

        peer_list = [p.strip().upper() for p in peers_input.split(',') if p.strip()]
        all_tickers = [ticker] + peer_list
        unique_tickers = list(dict.fromkeys(all_tickers)) # Removes duplicates

        comps_df = get_comps_data(unique_tickers)

        # Added formatting for new columns and corrected P/S
        st.dataframe(comps_df.style
            .format({
                "Market Cap (B)": "${:,.2f}B",
                "Revenue Growth (TTM)": "{:,.2%}",
                "Gross Margin (TTM)": "{:,.2%}",
                "EBITDA Margin (TTM)": "{:,.2%}",
                "P/E (Forward)": "{:,.2f}x",
                "P/S (TTM)": "{:,.2f}x",
                "EV/Revenue (TTM)": "{:,.2f}x",
                "EV/EBITDA (TTM)": "{:,.2f}x"
            }, na_rep='N/A') # Added na_rep
            .highlight_max(axis=0, subset=[
                "Revenue Growth (TTM)", "Gross Margin (TTM)", "EBITDA Margin (TTM)",
                "P/E (Forward)", "P/S (TTM)", "EV/Revenue (TTM)", "EV/EBITDA (TTM)"
                ], color="#0DAF43")
            .highlight_min(axis=0, subset=[
                "Revenue Growth (TTM)", "Gross Margin (TTM)", "EBITDA Margin (TTM)",
                "P/E (Forward)", "P/S (TTM)", "EV/Revenue (TTM)", "EV/EBITDA (TTM)"
                ], color='#FFA0A0')
        )

                # --- NEW: Sensitivity Analysis ---
        # --- Sensitivity Analysis (Using Dictionary Approach) ---
        st.subheader("Sensitivity Analysis (Implied Share Price $)")
        st.caption(
    "This table shows how the **implied share price** changes under different assumptions for the "
    "**WACC** (discount rate) and **terminal growth rate**. "
    "Each cell represents the fair value per share estimated by the DCF model. "
    "Higher terminal growth or lower WACC generally lead to higher valuations."
    )

        wacc_range = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
        t_rate_range = [t_rate - 0.005, t_rate - 0.0025, t_rate, t_rate + 0.0025, t_rate + 0.005]

        wacc_range = [max(0.01, val) for val in wacc_range]
        t_rate_range = [max(0.001, val) for val in t_rate_range]

        # Use a dictionary to store results
        sensitivity_results = {}

        base_fcf = safe_float(fcf)
        base_shares = safe_float(info.get('sharesOutstanding'))
        base_debt = safe_float(info.get('totalDebt'))
        base_cash = safe_float(info.get('totalCash'))

        if base_fcf > 0 and base_shares > 0:
            for w_sens in wacc_range:
                # Use WACC formatted string as the key for the outer dictionary
                w_key = f"{w_sens:.2%}"
                sensitivity_results[w_key] = {} # Create inner dictionary for this WACC row

                for t_sens in t_rate_range:
                    # Use T-Rate formatted string as the key for the inner dictionary
                    t_key = f"{t_sens:.2%}"

                    # print(f"DEBUG Loop: Trying w_sens={w_sens:.4f}, t_sens={t_sens:.4f}") # Keep if needed
                    if w_sens > t_sens:
                        # Calculate price for this specific combination
                        calculated_price = calculate_dcf_price(
                            base_fcf, g_rate_1, g_rate_2, t_sens, w_sens,
                            base_shares, base_debt, base_cash
                        )
                        # print(f"    => Price: {calculated_price:.2f}") # Keep if needed

                        # Assign the result to the dictionary
                        sensitivity_results[w_key][t_key] = calculated_price
                    else:
                        # Assign np.nan for invalid combinations
                        sensitivity_results[w_key][t_key] = np.nan

            # Convert the dictionary to a DataFrame AFTER the loops complete
            sensitivity_df = pd.DataFrame.from_dict(sensitivity_results, orient='index')
            sensitivity_df.index.name = "WACC"
            sensitivity_df.columns.name = "Terminal Growth Rate"

            # Melt for Plotly
            x_labels = sensitivity_df.columns.tolist()
            y_labels = sensitivity_df.index.tolist()
            z_values = sensitivity_df.replace([None], np.nan).values # Use np.nan for gaps

            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlGn', # Green=High, Red=Low
                # Custom hovertemplate to show formatted price
                hovertemplate='WACC: %{y}<br>T. Growth: %{x}<br>Price: $%{z:,.2f}<extra></extra>',
                # Add text to cells, formatting it manually
                text=[[format_number(val, decimals=2, is_dollar=True) for val in row] for row in z_values],
                texttemplate="%{text}", # Display the manually formatted text
                showscale=True, # Show the color bar legend
                colorbar={"title": 'Implied Share Price ($)'}
            ))

            fig.update_layout(
                title='DCF Sensitivity Analysis',
                xaxis_title="Terminal Growth Rate",
                yaxis_title="WACC",
                # Order Y axis correctly (higher WACC at top)
                yaxis={'categoryorder':'array', 'categoryarray': sorted(y_labels, key=lambda x: float(x.strip('%'))/100.0, reverse=True)},
                # Order X axis correctly
                xaxis={'categoryorder':'array', 'categoryarray': sorted(x_labels, key=lambda x: float(x.strip('%'))/100.0)}
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green = Higher Implied Price; Red = Lower Implied Price. N/A for invalid combinations (WACC <= T-Growth).")
            # --- End Heatmap Creation ---

        else:
            st.warning("Cannot perform sensitivity analysis due to missing FCF or Share data.")


        # --- Reverse DCF Section (Simplified Interpretation) ---
        st.subheader("Reverse DCF (Implied Growth Analysis)")
        st.caption("This calculation finds the short-term FCF growth rate (Years 1-5) that the market price implies, given your WACC, long-term growth, and terminal growth assumptions.")

        # Ensure necessary base variables exist and are valid floats
        current_price_rev = safe_float(info.get('currentPrice') or info.get('regularMarketPrice'))
        base_fcf_rev = safe_float(fcf if 'fcf' in locals() and fcf > 0 else 0.0)
        base_shares_rev = safe_float(info.get('sharesOutstanding'))
        base_debt_rev = safe_float(info.get('totalDebt'))
        base_cash_rev = safe_float(info.get('totalCash'))
        g_rate_2_rev = safe_float(g_rate_2 if 'g_rate_2' in locals() else 0.05)
        t_rate_rev = safe_float(t_rate if 't_rate' in locals() else 0.02)
        wacc_rev = safe_float(wacc if 'wacc' in locals() and wacc > 0 else 0.10)

        if current_price_rev > 0 and base_fcf_rev > 0 and base_shares_rev > 0 and wacc_rev > t_rate_rev:
            implied_growth = reverse_dcf(
                current_price=current_price_rev,
                fcf=base_fcf_rev,
                g_rate_2=g_rate_2_rev,
                t_rate=t_rate_rev,
                wacc=wacc_rev,
                shares_outstanding=base_shares_rev,
                total_debt=base_debt_rev,
                cash=base_cash_rev,
                years=5
            )

            if implied_growth is not None:
                st.metric(label="Current Market Price", value=f"${current_price_rev:,.2f}")
                st.metric(label="Implied FCF Growth (Yrs 1-5)", value=f"{implied_growth:.2%}")

                # --- Simplified Interpretation ---
                interpretation = f"The current market price implies **{implied_growth:.1%}** annual FCF growth for the next 5 years. "

                # Add qualitative assessment based on magnitude
                if implied_growth > 0.30:
                    st.warning(interpretation + "This required growth rate is **very aggressive**.")
                elif implied_growth > 0.15:
                    st.info(interpretation + "This required growth rate is **optimistic**.")
                elif implied_growth > 0.05:
                    st.success(interpretation + "This required growth rate seems **plausible** for a growing company.")
                elif implied_growth >= 0:
                     st.success(interpretation + "This required growth rate is **conservative**.")
                else: # Negative growth
                     st.error(interpretation + "The market appears to be pricing in a **decline** in FCF.")
                # --- End Simplified Interpretation ---

            else:
                st.warning("Could not converge on an implied growth rate. The current price might be too high/low for the model assumptions (WACC, long-term growth, terminal growth), input data might be missing, or the calculation encountered an error.")
        else:
            st.warning("Could not calculate implied growth due to missing/invalid data (Price, FCF, Shares) or WACC <= Terminal Rate.")

    except Exception as e:
        st.error(f"A critical error occurred loading data for {ticker}.")
        st.error(f"Debug info: {e}")