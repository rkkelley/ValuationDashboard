"""Yahoo Finance retrieval and intentional data normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Optional

import pandas as pd
import yfinance as yf


class DataFetchError(RuntimeError):
    """Raised when a ticker cannot provide enough data to analyze."""


@dataclass(frozen=True)
class DataPoint:
    """A normalized value with provenance for display in the UI."""

    value: Optional[float]
    source: str
    fallback: bool = False


@dataclass(frozen=True)
class CompanyData:
    ticker: str
    stock: Any
    info: dict[str, Any]
    financials: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    quarterly_cash_flow: pd.DataFrame


@dataclass(frozen=True)
class CompanyMetrics:
    free_cash_flow: DataPoint
    current_price: DataPoint
    shares_outstanding: DataPoint
    total_debt: DataPoint
    cash: DataPoint
    market_cap: DataPoint
    beta: DataPoint
    interest_expense: DataPoint


COMPS_COLUMNS = [
    "Company Name",
    "Market Cap (B)",
    "Revenue Growth (TTM)",
    "Gross Margin (TTM)",
    "EBITDA Margin (TTM)",
    "P/E (Forward)",
    "P/S (TTM)",
    "EV/Revenue (TTM)",
    "EV/EBITDA (TTM)",
]


def optional_float(value: object) -> Optional[float]:
    """Convert a value only when it is a finite number; preserve missingness."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_ticker(ticker: str) -> str:
    normalized = str(ticker or "").strip().upper()
    if not normalized or not re.fullmatch(r"[A-Z0-9.\-\^=]{1,15}", normalized):
        raise DataFetchError("Enter a valid Yahoo Finance ticker symbol.")
    return normalized


def parse_peer_tickers(raw: str, target_ticker: str) -> list[str]:
    """Normalize comma-separated peers, deduplicate, and include the target."""

    values = [target_ticker]
    values.extend(str(raw or "").split(","))
    normalized: list[str] = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)
    return normalized


def _safe_frame(loader: Any) -> pd.DataFrame:
    try:
        result = loader()
    except Exception:
        return pd.DataFrame()
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()


def fetch_company_data(ticker: str) -> CompanyData:
    """Retrieve the statements used by the dashboard, with a clean error path."""

    normalized = normalize_ticker(ticker)
    stock = yf.Ticker(normalized)
    try:
        info = stock.info or {}
    except Exception as exc:
        raise DataFetchError(
            f"Yahoo Finance could not return data for {normalized}. Check the ticker and try again."
        ) from exc

    # Some valid tickers return a sparse info payload.  A short history check
    # lets us distinguish that case from an unsupported ticker without showing
    # a provider traceback in the app.
    has_company_identity = any(
        info.get(key)
        for key in ("symbol", "shortName", "longName", "regularMarketPrice", "marketCap")
    )
    if not has_company_identity:
        history = _safe_frame(lambda: stock.history(period="5d"))
        if history.empty:
            raise DataFetchError(
                f"No Yahoo Finance data was found for {normalized}. Check the ticker and try again."
            )

    financials = _safe_frame(lambda: stock.financials)
    balance_sheet = _safe_frame(lambda: stock.balance_sheet)
    cash_flow = _safe_frame(lambda: stock.cashflow)
    quarterly_cash_flow = _safe_frame(lambda: stock.get_cashflow(freq="quarterly"))
    return CompanyData(
        ticker=normalized,
        stock=stock,
        info=dict(info),
        financials=financials,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        quarterly_cash_flow=quarterly_cash_flow,
    )


def _first_info_value(info: dict[str, Any], keys: Iterable[str], label: str) -> DataPoint:
    for position, key in enumerate(keys):
        value = optional_float(info.get(key))
        if value is not None:
            return DataPoint(value, f"Yahoo Finance: {key}", fallback=position > 0)
    return DataPoint(None, f"Unavailable: {label}")


def _series_for_labels(statement: pd.DataFrame, labels: Iterable[str]) -> tuple[Optional[pd.Series], Optional[str]]:
    if statement is None or statement.empty:
        return None, None
    index_lookup = {str(value).strip().casefold(): value for value in statement.index}
    column_lookup = {str(value).strip().casefold(): value for value in statement.columns}
    for label in labels:
        key = str(label).strip().casefold()
        if key in index_lookup:
            series = statement.loc[index_lookup[key]]
            return pd.Series(series), str(label)
        if key in column_lookup:
            series = statement[column_lookup[key]]
            return pd.Series(series), str(label)
    return None, None


def _numeric_series(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values.map(lambda item: math.isfinite(float(item)))]
    if values.empty:
        return None
    # Yahoo normally returns newest periods first.  Sort actual dates so this
    # remains correct if a provider response changes its column order.
    try:
        dates = pd.to_datetime(values.index)
        values = values.iloc[dates.argsort()[::-1]]
    except (TypeError, ValueError):
        pass
    return values


def _latest_four(series: Optional[pd.Series]) -> Optional[pd.Series]:
    values = _numeric_series(series)
    if values is None or len(values) < 4:
        return None
    return values.iloc[:4]


def get_ttm_fcf(quarterly_cash_flow: pd.DataFrame) -> DataPoint:
    """Prefer quarterly FreeCashFlow, then calculate OCF minus capex."""

    direct_labels = ("FreeCashFlow", "Free Cash Flow")
    direct_series, direct_label = _series_for_labels(quarterly_cash_flow, direct_labels)
    direct_values = _latest_four(direct_series)
    if direct_values is not None:
        return DataPoint(
            float(direct_values.sum()),
            f"Latest four quarters: {direct_label}",
            fallback=direct_label != "FreeCashFlow",
        )

    ocf_labels = (
        "OperatingCashFlow",
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    )
    capex_labels = (
        "CapitalExpenditure",
        "Capital Expenditure",
        "Capital Expenditures",
        "Capital Expenditure Reported",
        "Purchase Of PPE",
        "Purchase Of Property Plant And Equipment",
        "Purchase Of Property, Plant And Equipment",
    )
    ocf_series, ocf_label = _series_for_labels(quarterly_cash_flow, ocf_labels)
    capex_series, capex_label = _series_for_labels(quarterly_cash_flow, capex_labels)
    ocf_values = _numeric_series(ocf_series)
    capex_values = _numeric_series(capex_series)
    if ocf_values is None or capex_values is None:
        return DataPoint(None, "Unavailable: quarterly free cash flow")

    aligned = pd.concat([ocf_values.rename("ocf"), capex_values.rename("capex")], axis=1).dropna().head(4)
    if len(aligned) < 4:
        return DataPoint(None, "Unavailable: four complete quarters of free cash flow")
    # Yahoo reports capex as a negative cash outflow in most statements.  The
    # sign-aware calculation also handles sources that report the positive use.
    fcf_values = aligned.apply(
        lambda row: row["ocf"] + row["capex"] if row["capex"] < 0 else row["ocf"] - row["capex"],
        axis=1,
    )
    return DataPoint(
        float(fcf_values.sum()),
        f"Fallback: {ocf_label} minus {capex_label}, latest four quarters",
        fallback=True,
    )


def _latest_statement_value(
    statement: pd.DataFrame,
    labels: Iterable[str],
    label: str,
) -> DataPoint:
    series, matched = _series_for_labels(statement, labels)
    values = _numeric_series(series)
    if values is not None:
        return DataPoint(float(values.iloc[0]), f"Yahoo Finance: {matched}", fallback=True)
    return DataPoint(None, f"Unavailable: {label}")


def normalize_metrics(data: CompanyData) -> CompanyMetrics:
    """Normalize the small set of metrics needed by the dashboard."""

    info = data.info
    current_price = _first_info_value(
        info,
        ("currentPrice", "regularMarketPrice", "previousClose", "regularMarketPreviousClose"),
        "current share price",
    )
    shares = _first_info_value(
        info,
        ("sharesOutstanding", "impliedSharesOutstanding"),
        "shares outstanding",
    )
    market_cap = _first_info_value(info, ("marketCap",), "market capitalization")
    if shares.value is None and market_cap.value is not None and current_price.value not in (None, 0):
        shares = DataPoint(
            market_cap.value / current_price.value,
            "Fallback: market capitalization divided by share price",
            fallback=True,
        )
    if market_cap.value is None and shares.value is not None and current_price.value is not None:
        market_cap = DataPoint(
            shares.value * current_price.value,
            "Fallback: shares outstanding multiplied by share price",
            fallback=True,
        )

    total_debt = _first_info_value(info, ("totalDebt", "longTermDebt"), "total debt")
    if total_debt.value is None:
        total_debt = _latest_statement_value(
            data.balance_sheet,
            ("Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"),
            "total debt",
        )
    cash = _first_info_value(info, ("totalCash", "cash"), "cash")
    if cash.value is None:
        cash = _latest_statement_value(
            data.balance_sheet,
            (
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "CashAndCashEquivalents",
                "Cash",
            ),
            "cash",
        )
    beta = _first_info_value(info, ("beta", "beta3Year"), "beta")
    if beta.value is None:
        beta = DataPoint(1.0, "Fallback: beta of 1.0", fallback=True)
    interest = _first_info_value(info, ("interestExpense",), "interest expense")
    if interest.value is None:
        interest = _latest_statement_value(
            data.financials,
            ("Interest Expense Non Operating", "Interest Expense", "InterestExpenseNonOperating"),
            "interest expense",
        )
    return CompanyMetrics(
        free_cash_flow=get_ttm_fcf(data.quarterly_cash_flow),
        current_price=current_price,
        shares_outstanding=shares,
        total_debt=total_debt,
        cash=cash,
        market_cap=market_cap,
        beta=beta,
        interest_expense=interest,
    )


def get_risk_free_rate(default: float = 0.04) -> DataPoint:
    """Read ^TNX and return a decimal rate, with a visible fallback."""

    try:
        info = yf.Ticker("^TNX").info or {}
    except Exception:
        info = {}
    value = _first_info_value(info, ("regularMarketPrice", "previousClose"), "10-year Treasury yield")
    if value.value is not None:
        return DataPoint(
            value.value / 100.0,
            f"{value.source}, converted from percentage",
            fallback=value.fallback,
        )
    return DataPoint(default, "Fallback: 4.0% risk-free rate", fallback=True)


def get_comps_data(tickers: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return normalized comps while skipping only the tickers that fail."""

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-\^=]{1,15}", ticker):
            skipped.append(ticker or "(blank)")
            continue
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if not info or not (
                info.get("symbol") or info.get("shortName") or info.get("longName")
                or info.get("regularMarketPrice") or info.get("marketCap")
            ):
                raise DataFetchError("no company data")
            rows.append(
                {
                    "Ticker": ticker,
                    "Company Name": info.get("shortName") or info.get("longName") or "N/A",
                    "Market Cap (B)": _scale(_first_info_value(info, ("marketCap",), "market cap").value, 1e9),
                    "Revenue Growth (TTM)": optional_float(info.get("revenueGrowth")),
                    "Gross Margin (TTM)": optional_float(info.get("grossMargins")),
                    "EBITDA Margin (TTM)": optional_float(info.get("ebitdaMargins")),
                    "P/E (Forward)": optional_float(info.get("forwardPE")),
                    "P/S (TTM)": optional_float(info.get("priceToSalesTrailing12Months")),
                    "EV/Revenue (TTM)": optional_float(info.get("enterpriseToRevenue")),
                    "EV/EBITDA (TTM)": optional_float(info.get("enterpriseToEbitda")),
                }
            )
        except Exception:
            skipped.append(ticker)

    if not rows:
        return pd.DataFrame(columns=COMPS_COLUMNS).rename_axis("Ticker"), skipped
    frame = pd.DataFrame(rows).set_index("Ticker")
    return frame.reindex(columns=COMPS_COLUMNS), skipped


def _scale(value: Optional[float], divisor: float) -> Optional[float]:
    return None if value is None else value / divisor
