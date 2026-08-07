"""Streamlit interface for the Financial Valuation Dashboard."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_service import (
    DataFetchError,
    fetch_company_data,
    get_comps_data,
    get_risk_free_rate,
    normalize_metrics,
    parse_peer_tickers,
)
from valuation import (
    DEFAULT_FALLBACK_WACC,
    DCFResult,
    WACCResult,
    calculate_dcf,
    calculate_dcf_from_fcf,
    calculate_wacc,
    project_two_stage_fcf,
    reverse_dcf,
)


st.set_page_config(
    page_title="Financial Valuation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.25rem; padding-bottom: 3rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        .section-kicker {
            color: #6b7280; font-size: .74rem; font-weight: 700;
            letter-spacing: .11em; text-transform: uppercase; margin-bottom: -.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(value: object, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"${number:,.{decimals}f}" if math.isfinite(number) else "N/A"


def format_large_currency(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    absolute = abs(number)
    if absolute >= 1e12:
        return f"${number / 1e12:,.2f}T"
    if absolute >= 1e9:
        return f"${number / 1e9:,.2f}B"
    if absolute >= 1e6:
        return f"${number / 1e6:,.2f}M"
    return format_currency(number, 0)


def format_percent(value: object, decimals: int = 1, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:.{decimals}%}"


def display_value(value: object, formatter=format_currency) -> str:
    return formatter(value) if value is not None else "N/A"


DEFAULT_TWO_STAGE_SETTINGS = {
    "stage_1_percent": 30.0,
    "stage_2_percent": 10.0,
    "duration_1": 5,
    "duration_2": 5,
}


def seed_custom_fcf_values(fcf_value: object, forecast_years: int) -> list[object]:
    """Create one-time custom-mode defaults without coupling later edits to DCF inputs."""

    settings = st.session_state.get("last_two_stage_settings", DEFAULT_TWO_STAGE_SETTINGS)
    seed_path = project_two_stage_fcf(
        fcf_value,
        settings["stage_1_percent"] / 100.0,
        settings["stage_2_percent"] / 100.0,
        settings["duration_1"],
        settings["duration_2"],
    )
    seeded_values = [value / 1e9 for value in seed_path]
    return [
        seeded_values[year - 1] if year <= len(seeded_values) else None
        for year in range(1, forecast_years + 1)
    ]


def resize_custom_fcf_values(
    values: list[object],
    forecast_years: int,
    fallback_values: list[object],
) -> list[object]:
    resized = list(values[:forecast_years])
    for index in range(len(resized), forecast_years):
        resized.append(fallback_values[index] if index < len(fallback_values) else None)
    return resized


def parse_custom_fcf_values(edited_forecast: pd.DataFrame) -> tuple[tuple[float, ...] | None, str | None]:
    values: list[float] = []
    invalid_years: list[str] = []
    years = edited_forecast["Year"].tolist()
    for year, raw_value in zip(years, edited_forecast["FCF ($B)"].tolist()):
        if raw_value is None or isinstance(raw_value, bool):
            invalid_years.append(str(year))
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            invalid_years.append(str(year))
            continue
        if not math.isfinite(value):
            invalid_years.append(str(year))
            continue
        values.append(value)

    if invalid_years:
        years_label = ", ".join(f"Year {year}" for year in invalid_years)
        return None, f"Enter a finite numeric FCF ($B) value for every forecast year. Missing or invalid value(s): {years_label}."
    return tuple(value * 1e9 for value in values), None


def render_custom_fcf_editor(
    ticker: str,
    fcf_value: object,
    forecast_years: int,
) -> tuple[tuple[float, ...] | None, str | None]:
    state_key = f"custom_fcf_values_{ticker}"
    fallback_values = seed_custom_fcf_values(fcf_value, forecast_years)
    stored_values = st.session_state.get(state_key)
    if stored_values is None:
        stored_values = fallback_values
    else:
        stored_values = resize_custom_fcf_values(
            list(stored_values),
            forecast_years,
            fallback_values,
        )

    length_key = f"custom_fcf_editor_length_{ticker}"
    revision_key = f"custom_fcf_editor_revision_{ticker}"
    if st.session_state.get(length_key) != forecast_years:
        st.session_state[length_key] = forecast_years
        st.session_state[revision_key] = st.session_state.get(revision_key, 0) + 1
    editor_key = f"custom_fcf_editor_{ticker}_{st.session_state[revision_key]}"

    forecast_df = pd.DataFrame(
        {
            "Year": list(range(1, forecast_years + 1)),
            "FCF ($B)": stored_values,
        }
    )
    edited_forecast = st.data_editor(
        forecast_df,
        key=editor_key,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Year": st.column_config.NumberColumn("Year", format="%d", disabled=True),
            "FCF ($B)": st.column_config.NumberColumn(
                "FCF ($B)",
                format="%.3f",
                help="Enter the analyst-supplied annual free cash flow in $ billions. Zero and negative values are allowed.",
            ),
        },
    )

    edited_values = edited_forecast["FCF ($B)"].tolist()
    stored_values = resize_custom_fcf_values(
        list(stored_values),
        forecast_years,
        fallback_values,
    )
    stored_values[:forecast_years] = edited_values
    st.session_state[state_key] = stored_values
    return parse_custom_fcf_values(edited_forecast)


@st.cache_data(ttl=600, show_spinner=False)
def load_comps(tickers: tuple[str, ...]):
    return get_comps_data(tickers)


def manual_wacc_result(value: float) -> WACCResult:
    return WACCResult(valid=True, wacc=value, fallback_used=False)


def render_wacc_details(result: WACCResult, mode: str, source_fallbacks: list[str]) -> None:
    st.markdown('<div class="section-kicker">Discount rate</div>', unsafe_allow_html=True)
    st.subheader("WACC assumptions")
    if mode == "Manual":
        st.info(f"Manual WACC selected: **{format_percent(result.wacc)}**")
        return

    values = {
        "Risk-free rate": format_percent(result.risk_free_rate),
        "Beta": f"{result.beta:.2f}" if result.beta is not None else "N/A",
        "Equity risk premium": format_percent(result.equity_risk_premium),
        "Cost of equity": format_percent(result.cost_of_equity),
        "Cost of debt": format_percent(result.cost_of_debt),
        "Equity weight": format_percent(result.equity_weight),
        "Debt weight": format_percent(result.debt_weight),
        "Tax rate": format_percent(result.tax_rate),
        "Final WACC": format_percent(result.wacc),
    }
    details = pd.DataFrame([values]).T.rename(columns={0: "Value"})
    st.dataframe(details, use_container_width=True, hide_index=False)
    notes = list(result.fallback_notes) + source_fallbacks
    if result.fallback_used or source_fallbacks:
        st.caption("Fallbacks used: " + " ".join(dict.fromkeys(notes)) if notes else "Fallback WACC used.")
    else:
        st.caption("All displayed WACC components came from the selected inputs and available company data.")


def render_sensitivity(
    annual_fcf: tuple[float, ...],
    terminal_growth: float,
    wacc: float,
    shares: float,
    debt: float,
    cash: float,
    forecast_method: str,
) -> None:
    st.markdown('<div class="section-kicker">Scenario analysis</div>', unsafe_allow_html=True)
    st.subheader("WACC and terminal-growth sensitivity")
    st.caption(
        f"Each cell holds the {forecast_method.lower()} FCF forecast constant and calls the shared DCF logic. "
        "Cells without a safe WACC-to-terminal-growth spread are shown as unavailable."
    )

    wacc_values = sorted({max(0.01, wacc + offset) for offset in (-0.01, -0.005, 0, 0.005, 0.01)})
    terminal_values = sorted({max(-0.01, terminal_growth + offset) for offset in (-0.005, -0.0025, 0, 0.0025, 0.005)})
    wacc_labels = [format_percent(value, 2) for value in wacc_values]
    terminal_labels = [format_percent(value, 2) for value in terminal_values]
    prices: list[list[float]] = []
    cell_text: list[list[str]] = []
    for scenario_wacc in wacc_values:
        row: list[float] = []
        text_row: list[str] = []
        for scenario_terminal in terminal_values:
            result = calculate_dcf_from_fcf(
                annual_fcf,
                scenario_terminal,
                scenario_wacc,
                shares,
                debt,
                cash,
            )
            price = result.implied_share_price if result.valid else np.nan
            row.append(price if price is not None else np.nan)
            label = format_currency(price)
            if abs(scenario_wacc - wacc) < 1e-9 and abs(scenario_terminal - terminal_growth) < 1e-9:
                label = f"★ {label}"
            text_row.append(label)
        prices.append(row)
        cell_text.append(text_row)

    figure = go.Figure(
        data=go.Heatmap(
            z=prices,
            x=terminal_labels,
            y=wacc_labels,
            text=cell_text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            hovertemplate="WACC: %{y}<br>Terminal growth: %{x}<br>Implied price: $%{z:,.2f}<extra></extra>",
            colorbar={"title": "Price ($)"},
            hoverongaps=False,
        )
    )
    figure.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        xaxis_title="Terminal growth",
        yaxis_title="WACC",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        f"★ Base case: WACC {format_percent(wacc, 2)}, terminal growth {format_percent(terminal_growth, 2)}. "
        "Prices are scenario outputs under the selected assumptions."
    )


def render_forecast_chart(
    result: DCFResult,
    forecast_method: str,
    duration_1: int | None,
) -> None:
    """Show projected and discounted FCF without interpreting the forecast."""

    years = list(range(1, len(result.projected_fcf) + 1))
    projected_billions = [value / 1e9 for value in result.projected_fcf]
    discounted_billions = [value / 1e9 for value in result.discounted_fcf]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=years,
            y=projected_billions,
            mode="lines+markers",
            name="Projected FCF",
            line={"color": "#6366f1", "width": 3},
            marker={"size": 7},
            hovertemplate="Year %{x}<br>Projected FCF: $%{y:,.3f}B<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=years,
            y=discounted_billions,
            mode="lines+markers",
            name="Discounted FCF",
            line={"color": "#14b8a6", "width": 2, "dash": "dot"},
            marker={"size": 6},
            hovertemplate="Year %{x}<br>Discounted FCF: $%{y:,.3f}B<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color="#9ca3af")
    if forecast_method == "Two-stage growth" and duration_1 is not None:
        figure.add_vline(
            x=duration_1 + 0.5,
            line_width=1,
            line_dash="dash",
            line_color="#9ca3af",
        )

    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 20},
        hovermode="x unified",
        xaxis={"title": "Forecast year", "dtick": 1},
        yaxis={"title": "Free cash flow ($B)", "tickprefix": "$", "ticksuffix": "B"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)
    if forecast_method == "Two-stage growth" and duration_1 is not None:
        st.caption("The dotted vertical line marks the Stage 1 / Stage 2 boundary. The zero line is shown for reference.")
    else:
        st.caption("The zero line is shown for reference; custom annual FCF values are analyst-supplied assumptions.")


def render_comps(ticker: str, peers_input: str) -> None:
    st.markdown('<div class="section-kicker">Market context</div>', unsafe_allow_html=True)
    st.subheader("Comparable companies")
    tickers = parse_peer_tickers(peers_input, ticker)
    comps_df, skipped = load_comps(tuple(tickers))
    if skipped:
        st.caption("Skipped unavailable ticker(s): " + ", ".join(skipped))
    if comps_df.empty:
        st.info("Comparable-company data was unavailable for the selected tickers.")
        return
    formatters = {
        "Market Cap (B)": "${:,.2f}B",
        "Revenue Growth (TTM)": "{:.1%}",
        "Gross Margin (TTM)": "{:.1%}",
        "EBITDA Margin (TTM)": "{:.1%}",
        "P/E (Forward)": "{:.2f}x",
        "P/S (TTM)": "{:.2f}x",
        "EV/Revenue (TTM)": "{:.2f}x",
        "EV/EBITDA (TTM)": "{:.2f}x",
    }
    st.dataframe(
        comps_df.style.format(formatters, na_rep="N/A"),
        use_container_width=True,
        height=min(520, 90 + 36 * len(comps_df)),
    )


def render_methodology() -> None:
    st.markdown('<div class="section-kicker">Reference</div>', unsafe_allow_html=True)
    st.subheader("Methodology and limitations")
    with st.expander("How to read this dashboard"):
        st.markdown(
            """
            The default model starts with trailing-twelve-month free cash flow, projects it through a higher-growth Stage 1 and a lower-growth Stage 2, and discounts those cash flows plus a Gordon-growth terminal value using WACC. Optional custom annual FCF mode accepts analyst-supplied explicit yearly values, allowing nonlinear cash-flow paths without adding a three-statement model. In either mode, enterprise value is adjusted for debt and cash to produce equity value and an implied share price; the final explicit FCF must be positive for terminal value.

            Automatic WACC uses CAPM for the cost of equity and an estimated pretax cost of debt weighted by market-value equity and debt. Missing inputs are labeled when a fallback is used. Reverse DCF solves for the Stage 1 growth rate implied by the current market price under the remaining assumptions.

            Results are highly sensitive to growth, WACC, terminal value, and data quality. Yahoo Finance values can be delayed, revised, incomplete, or inconsistent across issuers. Comparable-company multiples are point-in-time context and do not establish fair value.
            """
        )


st.sidebar.title("Valuation inputs")
ticker = st.sidebar.text_input(
    "Stock ticker",
    "AAPL",
    help="Yahoo Finance symbol, for example AAPL or MSFT.",
).strip().upper()
run_button = st.sidebar.button("Load company data", type="primary", use_container_width=True)

with st.sidebar.expander("DCF assumptions", expanded=True):
    forecast_method = st.radio(
        "Forecast method",
        ["Two-stage growth", "Custom annual FCF"],
        help="Two-stage growth is the default fast scenario. Custom annual FCF lets you enter each explicit forecast year in $ billions.",
    )
    if forecast_method == "Two-stage growth":
        st.caption("Fast scenario-analysis mode using Stage 1 and Stage 2 growth rates.")
        stage_settings = st.session_state.get("last_two_stage_settings", DEFAULT_TWO_STAGE_SETTINGS)
        stage_1_percent = st.slider(
            "Stage 1 FCF growth",
            min_value=-50.0,
            max_value=100.0,
            value=stage_settings["stage_1_percent"],
            step=1.0,
            format="%.0f%%",
            key="stage_1_percent",
            help="Annual free-cash-flow growth during the first projection stage.",
        )
        stage_2_percent = st.slider(
            "Stage 2 FCF growth",
            min_value=-20.0,
            max_value=30.0,
            value=stage_settings["stage_2_percent"],
            step=1.0,
            format="%.0f%%",
            key="stage_2_percent",
            help="Annual free-cash-flow growth during the second projection stage.",
        )
        duration_1 = st.slider(
            "Stage 1 duration",
            1,
            10,
            stage_settings["duration_1"],
            format="%d years",
            key="duration_1",
        )
        duration_2 = st.slider(
            "Stage 2 duration",
            1,
            10,
            stage_settings["duration_2"],
            format="%d years",
            key="duration_2",
        )
        st.session_state["last_two_stage_settings"] = {
            "stage_1_percent": stage_1_percent,
            "stage_2_percent": stage_2_percent,
            "duration_1": duration_1,
            "duration_2": duration_2,
        }
    else:
        st.caption("Enter analyst-supplied annual FCF after the company data loads. Zero and negative years are allowed.")
        explicit_forecast_years = st.slider(
            "Explicit forecast years",
            min_value=5,
            max_value=15,
            value=st.session_state.get("last_custom_forecast_years", 10),
            format="%d years",
            key="explicit_forecast_years",
            help="Number of annual free-cash-flow values in the custom forecast.",
        )
        st.session_state["last_custom_forecast_years"] = explicit_forecast_years
        stage_1_growth = None
        stage_2_growth = None
        duration_1 = None
        duration_2 = None
    terminal_percent = st.slider(
        "Terminal growth",
        min_value=-1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        format="%.1f%%",
        help="Perpetual growth rate after the explicit projection period.",
    )

with st.sidebar.expander("Discount rate (WACC)", expanded=True):
    wacc_mode = st.radio(
        "WACC mode",
        ["Automatic", "Manual"],
        help="Automatic uses CAPM and capital structure data; Manual lets you set the discount rate directly.",
    )
    if wacc_mode == "Manual":
        manual_wacc_percent = st.slider("Manual WACC", 1.0, 30.0, 10.0, 0.1, format="%.1f%%")
        erp_percent = 5.5
        tax_percent = 21.0
    else:
        manual_wacc_percent = 10.0
        erp_percent = st.slider(
            "Equity risk premium",
            3.0,
            10.0,
            5.5,
            0.1,
            format="%.1f%%",
            help="Expected market return above the risk-free rate.",
        )
        tax_percent = st.slider("Tax rate", 0.0, 50.0, 21.0, 1.0, format="%.0f%%")

with st.sidebar.expander("Comparable companies"):
    peers_input = st.text_input(
        "Peer tickers",
        "MSFT, GOOG, META, AMZN",
        help="Comma-separated Yahoo Finance symbols.",
    )

if forecast_method == "Two-stage growth":
    stage_1_growth = stage_1_percent / 100.0
    stage_2_growth = stage_2_percent / 100.0
terminal_growth = terminal_percent / 100.0

st.title("Financial Valuation Dashboard")
st.caption("Interactive DCF and scenario analysis using live Yahoo Finance data")

reuse_cached_analysis = (
    not run_button
    and st.session_state.get("analysis_ticker") == ticker
    and st.session_state.get("analysis_company") is not None
)
if not run_button and not reuse_cached_analysis:
    st.info("Enter a ticker and select Load company data to load the company profile, valuation, scenarios, and comps.")
    if forecast_method == "Custom annual FCF":
        st.markdown('<div class="section-kicker">Custom forecast</div>', unsafe_allow_html=True)
        st.subheader("Custom annual FCF")
        st.write("Enter your own free cash flow forecast for each explicit forecast year.")
        st.info("Load company data to initialize the annual FCF forecast.")
    render_methodology()
    st.stop()

try:
    if reuse_cached_analysis:
        company = st.session_state["analysis_company"]
        metrics = st.session_state["analysis_metrics"]
    else:
        with st.spinner(f"Retrieving financial data for {ticker or 'the selected ticker'}…"):
            company = fetch_company_data(ticker)
            metrics = normalize_metrics(company)
        st.session_state["analysis_ticker"] = company.ticker
        st.session_state["analysis_company"] = company
        st.session_state["analysis_metrics"] = metrics
except DataFetchError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error(f"The analysis could not be loaded for {ticker or 'this ticker'}. Please try again or use another symbol.")
    st.stop()

info = company.info
company_name = info.get("longName") or info.get("shortName") or company.ticker
st.markdown('<div class="section-kicker">Company overview</div>', unsafe_allow_html=True)
st.header(f"{company_name} ({company.ticker})")
overview_left, overview_right = st.columns([2, 1])
with overview_left:
    sector = info.get("sector") or "Sector unavailable"
    industry = info.get("industry") or "Industry unavailable"
    st.write(f"**{sector}** · {industry}")
    summary = info.get("longBusinessSummary")
    if summary:
        with st.expander("Company summary"):
            st.write(summary)
with overview_right:
    website = info.get("website")
    if website:
        st.markdown(f"[Company website]({website})")

debt_value = metrics.total_debt.value if metrics.total_debt.value is not None else 0.0
cash_value = metrics.cash.value if metrics.cash.value is not None else 0.0
market_cap_value = metrics.market_cap.value
shares_value = metrics.shares_outstanding.value
current_price = metrics.current_price.value
fcf_value = metrics.free_cash_flow.value

custom_fcf_values: tuple[float, ...] | None = None
custom_fcf_error: str | None = None
if forecast_method == "Custom annual FCF":
    st.markdown('<div class="section-kicker">Custom forecast</div>', unsafe_allow_html=True)
    st.subheader("Custom annual FCF")
    st.write("Enter your own free cash flow forecast for each explicit forecast year.")
    st.caption("Values are entered in $ billions. Positive, zero, and negative intermediate FCF values are supported.")
    custom_fcf_values, custom_fcf_error = render_custom_fcf_editor(
        company.ticker,
        fcf_value,
        explicit_forecast_years,
    )
    if custom_fcf_error:
        st.error(custom_fcf_error)

forecast_fcf = (
    custom_fcf_values
    if forecast_method == "Custom annual FCF"
    else project_two_stage_fcf(fcf_value, stage_1_growth, stage_2_growth, duration_1, duration_2)
)

risk_free_point = None
if wacc_mode == "Manual":
    wacc_result = manual_wacc_result(manual_wacc_percent / 100.0)
else:
    risk_free_point = get_risk_free_rate()
    wacc_result = calculate_wacc(
        risk_free_point.value,
        metrics.beta.value,
        erp_percent / 100.0,
        market_cap_value,
        debt_value,
        metrics.interest_expense.value,
        tax_percent / 100.0,
        fallback_wacc=DEFAULT_FALLBACK_WACC,
    )

st.markdown('<div class="section-kicker">Primary result</div>', unsafe_allow_html=True)
st.subheader("Valuation snapshot")
if not wacc_result.valid or wacc_result.wacc is None:
    st.error(wacc_result.error or "WACC could not be calculated.")
    dcf_result = DCFResult.invalid("WACC is unavailable.")
else:
    if shares_value is None:
        dcf_result = DCFResult.invalid("Shares outstanding was unavailable; the DCF was not run.")
    elif forecast_method == "Custom annual FCF" and forecast_fcf is None:
        dcf_result = DCFResult.invalid(custom_fcf_error or "Enter a finite FCF value for every explicit forecast year.")
    elif forecast_method == "Custom annual FCF":
        dcf_result = calculate_dcf_from_fcf(
            forecast_fcf,
            terminal_growth,
            wacc_result.wacc,
            shares_value,
            debt_value,
            cash_value,
        )
    elif fcf_value is None:
        dcf_result = DCFResult.invalid(
            "Reliable trailing-twelve-month free cash flow was unavailable; the DCF was not run."
        )
    else:
        dcf_result = calculate_dcf(
            fcf_value,
            stage_1_growth,
            stage_2_growth,
            terminal_growth,
            wacc_result.wacc,
            shares_value,
            debt_value,
            cash_value,
            duration_1,
            duration_2,
        )

    if not dcf_result.valid and not (forecast_method == "Custom annual FCF" and custom_fcf_error):
        st.error(dcf_result.error or "The DCF could not be calculated.")
    elif current_price is None:
        st.warning("The DCF is complete, but a current market price was unavailable for the comparison.")

    upside = None
    if dcf_result.valid and dcf_result.implied_share_price is not None and current_price not in (None, 0):
        upside = dcf_result.implied_share_price / current_price - 1.0
    snapshot = st.columns(6)
    snapshot[0].metric("Current price", format_currency(current_price))
    snapshot[1].metric("Implied price", format_currency(dcf_result.implied_share_price if dcf_result.valid else None))
    snapshot[2].metric("Upside / downside", format_percent(upside, signed=True))
    snapshot[3].metric("Enterprise value", format_large_currency(dcf_result.enterprise_value if dcf_result.valid else None))
    snapshot[4].metric("Equity value", format_large_currency(dcf_result.equity_value if dcf_result.valid else None))
    snapshot[5].metric("WACC", format_percent(wacc_result.wacc))
    if dcf_result.valid and current_price is not None and upside is not None:
        direction = "above" if upside >= 0 else "below"
        st.caption(f"The selected assumptions produce an implied value {direction} the current market price.")
    st.caption(
        f"Terminal value contribution to enterprise value: "
        f"{format_percent(dcf_result.terminal_value_pct_enterprise_value) if dcf_result.valid else 'N/A'} · "
        "Outputs are assumption-sensitive."
    )

if metrics.free_cash_flow.fallback:
    st.info(f"FCF source fallback used: {metrics.free_cash_flow.source}")
neutral_adjustments = []
if metrics.total_debt.value is None:
    neutral_adjustments.append("debt unavailable; $0 used for the EV-to-equity adjustment")
if metrics.cash.value is None:
    neutral_adjustments.append("cash unavailable; $0 used for the EV-to-equity adjustment")
if neutral_adjustments:
    st.info("Data note: " + "; ".join(neutral_adjustments) + ".")

st.markdown('<div class="section-kicker">Model inputs</div>', unsafe_allow_html=True)
st.subheader("DCF assumptions and WACC")
assumption_columns = st.columns(5)
if forecast_method == "Two-stage growth":
    assumption_columns[0].metric("Forecast method", forecast_method)
    assumption_columns[1].metric("Stage 1 growth", format_percent(stage_1_growth))
    assumption_columns[2].metric("Stage 2 growth", format_percent(stage_2_growth))
    assumption_columns[3].metric("Terminal growth", format_percent(terminal_growth))
    assumption_columns[4].metric("Projection period", f"{duration_1 + duration_2} years")
else:
    assumption_columns[0].metric("Forecast method", forecast_method)
    assumption_columns[1].metric("Terminal growth", format_percent(terminal_growth))
    assumption_columns[2].metric("Explicit forecast years", f"{explicit_forecast_years} years")
    assumption_columns[3].metric("FCF input units", "$ billions")
    assumption_columns[4].metric("TTM FCF reference", format_large_currency(fcf_value))
source_fallbacks = []
if wacc_mode == "Automatic" and risk_free_point is not None and risk_free_point.fallback:
    source_fallbacks.append(f"Risk-free rate: {risk_free_point.source}")
for label, point in (
    ("Price", metrics.current_price),
    ("Shares", metrics.shares_outstanding),
    ("Debt", metrics.total_debt),
    ("Cash", metrics.cash),
    ("Market cap", metrics.market_cap),
    ("Beta", metrics.beta),
    ("Interest expense", metrics.interest_expense),
):
    if point.fallback:
        source_fallbacks.append(f"{label}: {point.source}")
with st.expander("View WACC details and data provenance"):
    render_wacc_details(wacc_result, wacc_mode, source_fallbacks)
    provenance = pd.DataFrame(
        [
            {"Metric": "Free cash flow", "Value": format_large_currency(fcf_value), "Source": metrics.free_cash_flow.source},
            {"Metric": "Shares outstanding", "Value": display_value(shares_value, lambda value: f"{value:,.0f}"), "Source": metrics.shares_outstanding.source},
            {"Metric": "Debt", "Value": format_large_currency(metrics.total_debt.value), "Source": metrics.total_debt.source},
            {"Metric": "Cash", "Value": format_large_currency(metrics.cash.value), "Source": metrics.cash.source},
            {"Metric": "Share price", "Value": format_currency(current_price), "Source": metrics.current_price.source},
            {"Metric": "Forecast", "Value": forecast_method, "Source": "Analyst-supplied annual FCF" if forecast_method == "Custom annual FCF" else "Two-stage growth assumptions"},
        ]
    )
    st.dataframe(provenance, hide_index=True, use_container_width=True)

st.markdown('<div class="section-kicker">Explicit forecast</div>', unsafe_allow_html=True)
st.subheader("Projected cash flows")
if dcf_result.valid:
    st.caption(f"Forecast method: **{forecast_method}**")
    render_forecast_chart(dcf_result, forecast_method, duration_1)
    forecast_data = {
        "Year": [f"Year {year}" for year in range(1, len(dcf_result.projected_fcf) + 1)],
        "Projected FCF": list(dcf_result.projected_fcf),
        "Discounted FCF": list(dcf_result.discounted_fcf),
    }
    if forecast_method == "Two-stage growth":
        forecast_data = {
            "Year": forecast_data["Year"],
            "Stage": [
                "Stage 1" if year <= duration_1 else "Stage 2"
                for year in range(1, len(dcf_result.projected_fcf) + 1)
            ],
            "Projected FCF": forecast_data["Projected FCF"],
            "Discounted FCF": forecast_data["Discounted FCF"],
        }
    with st.expander("View detailed forecast"):
        cash_flow_table = pd.DataFrame(forecast_data)
        st.dataframe(
            cash_flow_table.style.format({"Projected FCF": "${:,.0f}", "Discounted FCF": "${:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
    dcf_details = st.columns(4)
    dcf_details[0].metric("Terminal value", format_large_currency(dcf_result.terminal_value))
    dcf_details[1].metric("Discounted terminal value", format_large_currency(dcf_result.discounted_terminal_value))
    dcf_details[2].metric("Enterprise value", format_large_currency(dcf_result.enterprise_value))
    dcf_details[3].metric("Equity value", format_large_currency(dcf_result.equity_value))
else:
    st.info("Projected cash flows are unavailable until the DCF inputs are valid.")

if dcf_result.valid and shares_value is not None and forecast_fcf and wacc_result.wacc is not None:
    render_sensitivity(
        forecast_fcf,
        terminal_growth,
        wacc_result.wacc,
        shares_value,
        debt_value,
        cash_value,
        forecast_method,
    )
else:
    st.subheader("WACC and terminal-growth sensitivity")
    st.info("Sensitivity analysis requires a valid DCF, explicit forecast, shares outstanding, and WACC.")

st.markdown('<div class="section-kicker">Market-implied assumptions</div>', unsafe_allow_html=True)
st.subheader("Reverse DCF")
st.caption("Solves for the Stage 1 FCF growth rate implied by the current market price while holding the other assumptions constant.")
if forecast_method != "Two-stage growth":
    st.info("Reverse DCF is available only in Two-stage growth mode because it solves for Stage 1 growth.")
elif current_price is None or fcf_value is None or shares_value is None or wacc_result.wacc is None:
    st.info("Reverse DCF requires current price, reliable FCF, shares outstanding, and WACC.")
else:
    reverse_result = reverse_dcf(
        current_price,
        fcf_value,
        stage_2_growth,
        terminal_growth,
        wacc_result.wacc,
        shares_value,
        debt_value,
        cash_value,
        duration_1,
        duration_2,
    )
    if reverse_result.converged and reverse_result.implied_growth is not None:
        reverse_columns = st.columns(2)
        reverse_columns[0].metric("Current market price", format_currency(current_price))
        reverse_columns[1].metric(
            f"Implied Stage 1 growth (Years 1–{duration_1})",
            format_percent(reverse_result.implied_growth),
        )
        st.caption("This is the growth rate implied by the selected model assumptions.")
    else:
        st.warning(reverse_result.error or "Reverse DCF did not converge within the supported range.")

render_comps(company.ticker, peers_input)
render_methodology()
