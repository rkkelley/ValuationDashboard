"""Pure financial valuation calculations.

This module intentionally has no Streamlit or yfinance dependencies.  Keeping
the model here makes the core calculations deterministic and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence


MIN_TERMINAL_SPREAD = 0.005
DEFAULT_FALLBACK_WACC = 0.10


def _finite(value: object) -> Optional[float]:
    """Return a finite float or ``None`` for unavailable/non-numeric values."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_number(name: str, value: object) -> Optional[str]:
    if _finite(value) is None:
        return f"{name} must be a finite number."
    return None


@dataclass(frozen=True)
class DCFResult:
    """The complete output of either supported DCF forecast method."""

    valid: bool
    error: Optional[str] = None
    projected_fcf: tuple[float, ...] = field(default_factory=tuple)
    discounted_fcf: tuple[float, ...] = field(default_factory=tuple)
    terminal_value: Optional[float] = None
    discounted_terminal_value: Optional[float] = None
    enterprise_value: Optional[float] = None
    equity_value: Optional[float] = None
    implied_share_price: Optional[float] = None
    terminal_value_pct_enterprise_value: Optional[float] = None

    @classmethod
    def invalid(cls, message: str) -> "DCFResult":
        return cls(valid=False, error=message)


def calculate_dcf_from_fcf(
    annual_fcf: Sequence[object],
    terminal_growth: object,
    wacc: object,
    shares_outstanding: object,
    total_debt: object = 0.0,
    cash: object = 0.0,
    minimum_terminal_spread: float = MIN_TERMINAL_SPREAD,
) -> DCFResult:
    """Value an explicit annual FCF sequence using the shared DCF mechanics.

    Explicit forecast years may contain positive, zero, or negative FCF.  The
    final explicit year must be positive so the Gordon-growth terminal value
    remains economically meaningful and numerically stable.
    """

    if isinstance(annual_fcf, (str, bytes)):
        return DCFResult.invalid("Annual FCF forecast must be a sequence of numbers.")
    try:
        forecast = tuple(_finite(value) for value in annual_fcf)
    except TypeError:
        return DCFResult.invalid("Annual FCF forecast must be a sequence of numbers.")
    if not forecast:
        return DCFResult.invalid("Annual FCF forecast must contain at least one year.")
    if any(value is None for value in forecast):
        return DCFResult.invalid("Annual FCF forecast must contain only finite numbers.")

    required = {
        "terminal growth": terminal_growth,
        "WACC": wacc,
        "shares outstanding": shares_outstanding,
        "total debt": total_debt,
        "cash": cash,
    }
    for name, value in required.items():
        error = _validate_number(name, value)
        if error:
            return DCFResult.invalid(error)

    terminal = float(terminal_growth)
    discount_rate = float(wacc)
    shares = float(shares_outstanding)
    debt = float(total_debt)
    cash_value = float(cash)
    projected = tuple(float(value) for value in forecast)

    if shares <= 0:
        return DCFResult.invalid("Shares outstanding must be positive to calculate an implied price.")
    if debt < 0 or cash_value < 0:
        return DCFResult.invalid("Debt and cash must be zero or positive.")
    if discount_rate <= 0:
        return DCFResult.invalid("WACC must be greater than zero.")
    if terminal <= -1:
        return DCFResult.invalid("Terminal growth must be greater than -100%.")
    if discount_rate <= terminal:
        return DCFResult.invalid("WACC must be greater than terminal growth.")
    if discount_rate - terminal < minimum_terminal_spread:
        return DCFResult.invalid(
            "WACC and terminal growth are too close; use at least a 0.5% spread."
        )
    if projected[-1] <= 0:
        return DCFResult.invalid(
            "The final forecast FCF must be positive to calculate terminal value."
        )

    try:
        terminal_value = projected[-1] * (1.0 + terminal) / (discount_rate - terminal)
        discounted = [
            value / ((1.0 + discount_rate) ** year)
            for year, value in enumerate(projected, start=1)
        ]
        discounted_terminal = terminal_value / ((1.0 + discount_rate) ** len(projected))
        enterprise_value = sum(discounted) + discounted_terminal
        equity_value = enterprise_value - debt + cash_value
        implied_price = equity_value / shares
    except (OverflowError, ValueError, ZeroDivisionError):
        return DCFResult.invalid("The DCF calculation produced an invalid result.")

    outputs = [terminal_value, discounted_terminal, enterprise_value, equity_value, implied_price]
    if not all(math.isfinite(value) for value in outputs + discounted):
        return DCFResult.invalid("The DCF produced a non-finite result.")
    if enterprise_value == 0:
        return DCFResult.invalid("The DCF produced a zero enterprise value.")

    return DCFResult(
        valid=True,
        projected_fcf=projected,
        discounted_fcf=tuple(discounted),
        terminal_value=terminal_value,
        discounted_terminal_value=discounted_terminal,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        implied_share_price=implied_price,
        terminal_value_pct_enterprise_value=discounted_terminal / enterprise_value,
    )


def project_two_stage_fcf(
    fcf: object,
    stage_1_growth: object,
    stage_2_growth: object,
    duration_1: int,
    duration_2: int,
) -> tuple[float, ...]:
    """Generate the existing two-stage annual FCF path for reuse by the UI."""

    values = (fcf, stage_1_growth, stage_2_growth)
    if any(_finite(value) is None for value in values):
        return tuple()
    if not isinstance(duration_1, int) or isinstance(duration_1, bool) or duration_1 <= 0:
        return tuple()
    if not isinstance(duration_2, int) or isinstance(duration_2, bool) or duration_2 <= 0:
        return tuple()

    current_fcf = float(fcf)
    stage_1 = float(stage_1_growth)
    stage_2 = float(stage_2_growth)
    if current_fcf <= 0 or stage_1 <= -1 or stage_2 <= -1:
        return tuple()

    projected: list[float] = []
    try:
        for year in range(1, duration_1 + duration_2 + 1):
            current_fcf *= 1.0 + (stage_1 if year <= duration_1 else stage_2)
            if not math.isfinite(current_fcf):
                return tuple()
            projected.append(current_fcf)
    except (OverflowError, ValueError, ZeroDivisionError):
        return tuple()
    return tuple(projected)


def calculate_dcf(
    fcf: object,
    stage_1_growth: object,
    stage_2_growth: object,
    terminal_growth: object,
    wacc: object,
    shares_outstanding: object,
    total_debt: object = 0.0,
    cash: object = 0.0,
    duration_1: int = 5,
    duration_2: int = 5,
    minimum_terminal_spread: float = MIN_TERMINAL_SPREAD,
) -> DCFResult:
    """Calculate a two-stage DCF and return a validated result object.

    Debt and cash may legitimately be zero, but all other required financial
    inputs must be present and finite.  A 50bp minimum spread between WACC and
    terminal growth prevents unstable terminal values.
    """

    required = {
        "free cash flow": fcf,
        "stage 1 growth": stage_1_growth,
        "stage 2 growth": stage_2_growth,
        "terminal growth": terminal_growth,
        "WACC": wacc,
        "shares outstanding": shares_outstanding,
        "total debt": total_debt,
        "cash": cash,
    }
    for name, value in required.items():
        error = _validate_number(name, value)
        if error:
            return DCFResult.invalid(error)

    if not isinstance(duration_1, int) or isinstance(duration_1, bool) or duration_1 <= 0:
        return DCFResult.invalid("Stage 1 duration must be a positive whole number of years.")
    if not isinstance(duration_2, int) or isinstance(duration_2, bool) or duration_2 <= 0:
        return DCFResult.invalid("Stage 2 duration must be a positive whole number of years.")

    fcf_value = float(fcf)
    stage_1 = float(stage_1_growth)
    stage_2 = float(stage_2_growth)
    terminal = float(terminal_growth)
    discount_rate = float(wacc)
    shares = float(shares_outstanding)
    debt = float(total_debt)
    cash_value = float(cash)

    if fcf_value <= 0:
        return DCFResult.invalid("Free cash flow must be positive to run the DCF.")
    if shares <= 0:
        return DCFResult.invalid("Shares outstanding must be positive to calculate an implied price.")
    if debt < 0 or cash_value < 0:
        return DCFResult.invalid("Debt and cash must be zero or positive.")
    if discount_rate <= 0:
        return DCFResult.invalid("WACC must be greater than zero.")
    if terminal <= -1 or stage_1 <= -1 or stage_2 <= -1:
        return DCFResult.invalid("Growth assumptions must be greater than -100%.")
    if discount_rate <= terminal:
        return DCFResult.invalid("WACC must be greater than terminal growth.")
    if discount_rate - terminal < minimum_terminal_spread:
        return DCFResult.invalid(
            "WACC and terminal growth are too close; use at least a 0.5% spread."
        )

    projected = project_two_stage_fcf(
        fcf_value,
        stage_1,
        stage_2,
        duration_1,
        duration_2,
    )
    if not projected:
        return DCFResult.invalid("The projected free cash flows are not finite.")

    return calculate_dcf_from_fcf(
        projected,
        terminal_growth,
        wacc,
        shares_outstanding,
        total_debt,
        cash,
        minimum_terminal_spread,
    )


def calculate_dcf_price(*args, **kwargs) -> Optional[float]:
    """Compatibility helper returning only the implied price for a scenario."""

    result = calculate_dcf(*args, **kwargs)
    return result.implied_share_price if result.valid else None


@dataclass(frozen=True)
class WACCResult:
    """Transparent automatic WACC calculation output."""

    valid: bool
    error: Optional[str] = None
    risk_free_rate: Optional[float] = None
    beta: Optional[float] = None
    equity_risk_premium: Optional[float] = None
    cost_of_equity: Optional[float] = None
    cost_of_debt: Optional[float] = None
    equity_weight: Optional[float] = None
    debt_weight: Optional[float] = None
    tax_rate: Optional[float] = None
    wacc: Optional[float] = None
    fallback_used: bool = False
    fallback_notes: tuple[str, ...] = field(default_factory=tuple)


def calculate_wacc(
    risk_free_rate: object,
    beta: object,
    equity_risk_premium: object,
    market_cap: object,
    total_debt: object,
    interest_expense: object,
    tax_rate: object,
    fallback_wacc: float = DEFAULT_FALLBACK_WACC,
) -> WACCResult:
    """Calculate automatic WACC and identify every fallback used.

    Cost of debt uses interest expense / total debt when both are available.
    Otherwise, or when that rate falls outside the documented 0%-30% safety
    range, it uses risk-free rate + 1.5%.  Missing market capitalization is
    material enough to use the clearly labeled fallback WACC.
    """

    values = {
        "equity risk premium": equity_risk_premium,
        "tax rate": tax_rate,
    }
    for name, value in values.items():
        error = _validate_number(name, value)
        if error:
            return WACCResult(valid=False, error=error)

    notes: list[str] = []
    fallback_used = False

    rf = _finite(risk_free_rate)
    if rf is None:
        rf = 0.04
        fallback_used = True
        notes.append("Risk-free rate unavailable; 4.0% fallback used.")
    beta_value = _finite(beta)
    if beta_value is None:
        beta_value = 1.0
        fallback_used = True
        notes.append("Beta unavailable; beta of 1.0 used.")
    erp = float(equity_risk_premium)
    market_value = _finite(market_cap)
    if market_value is None:
        market_value = 0.0
        fallback_used = True
        notes.append("Market capitalization unavailable; fallback WACC is shown.")
    debt_value = _finite(total_debt)
    if debt_value is None:
        debt_value = 0.0
        fallback_used = True
        notes.append("Total debt unavailable; zero debt used for capital weights.")
    taxes = float(tax_rate)

    if market_value < 0 or debt_value < 0:
        return WACCResult(valid=False, error="Market capitalization and debt cannot be negative.")
    if not 0 <= taxes <= 1:
        return WACCResult(valid=False, error="Tax rate must be between 0% and 100%.")

    cost_of_equity = rf + beta_value * erp
    debt_rate = _finite(interest_expense)
    if debt_value > 0 and debt_rate is not None:
        debt_rate = abs(debt_rate) / debt_value
        if not 0 <= debt_rate <= 0.30:
            debt_rate = None
            fallback_used = True
            notes.append("Implied cost of debt was outside the 0%-30% safety range.")
    if debt_rate is None:
        debt_rate = rf + 0.015
        fallback_used = True
        notes.append("Cost of debt fallback used: risk-free rate + 1.5%.")

    capital_base = market_value + debt_value
    if capital_base <= 0 or market_value <= 0:
        fallback_used = True
        return WACCResult(
            valid=True,
            risk_free_rate=rf,
            beta=beta_value,
            equity_risk_premium=erp,
            cost_of_equity=cost_of_equity,
            cost_of_debt=debt_rate,
            equity_weight=1.0,
            debt_weight=0.0,
            tax_rate=taxes,
            wacc=fallback_wacc,
            fallback_used=True,
            fallback_notes=tuple(notes),
        )

    equity_weight = market_value / capital_base
    debt_weight = debt_value / capital_base
    calculated_wacc = (
        equity_weight * cost_of_equity
        + debt_weight * debt_rate * (1.0 - taxes)
    )
    if not math.isfinite(calculated_wacc):
        return WACCResult(valid=False, error="The WACC calculation produced a non-finite result.")

    return WACCResult(
        valid=True,
        risk_free_rate=rf,
        beta=beta_value,
        equity_risk_premium=erp,
        cost_of_equity=cost_of_equity,
        cost_of_debt=debt_rate,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        tax_rate=taxes,
        wacc=calculated_wacc,
        fallback_used=fallback_used,
        fallback_notes=tuple(notes),
    )


@dataclass(frozen=True)
class ReverseDCFResult:
    """Reverse DCF output, including convergence and range status."""

    valid: bool
    converged: bool
    implied_growth: Optional[float] = None
    error: Optional[str] = None
    search_range: tuple[float, float] = (-0.50, 2.00)


def reverse_dcf(
    current_price: object,
    fcf: object,
    stage_2_growth: object,
    terminal_growth: object,
    wacc: object,
    shares_outstanding: object,
    total_debt: object = 0.0,
    cash: object = 0.0,
    duration_1: int = 5,
    duration_2: int = 5,
    growth_range: tuple[float, float] = (-0.50, 2.00),
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> ReverseDCFResult:
    """Solve for Stage 1 growth implied by the current market price."""

    price = _finite(current_price)
    if price is None or price <= 0:
        return ReverseDCFResult(False, False, error="Current price must be positive.", search_range=growth_range)

    low, high = growth_range
    if low <= -1 or high <= low:
        return ReverseDCFResult(False, False, error="The supported growth range is invalid.", search_range=growth_range)

    low_result = calculate_dcf(
        fcf, low, stage_2_growth, terminal_growth, wacc, shares_outstanding,
        total_debt, cash, duration_1, duration_2,
    )
    high_result = calculate_dcf(
        fcf, high, stage_2_growth, terminal_growth, wacc, shares_outstanding,
        total_debt, cash, duration_1, duration_2,
    )
    if not low_result.valid or not high_result.valid:
        return ReverseDCFResult(
            False,
            False,
            error=low_result.error or high_result.error or "Reverse DCF inputs are invalid.",
            search_range=growth_range,
        )

    low_price = low_result.implied_share_price
    high_price = high_result.implied_share_price
    if low_price is None or high_price is None:
        return ReverseDCFResult(False, False, error="The DCF did not produce an implied price.", search_range=growth_range)
    if price < low_price or price > high_price:
        return ReverseDCFResult(
            True,
            False,
            error=(
                f"The current price is outside the supported implied-growth range "
                f"of {low:.0%} to {high:.0%}."
            ),
            search_range=growth_range,
        )

    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        result = calculate_dcf(
            fcf, midpoint, stage_2_growth, terminal_growth, wacc, shares_outstanding,
            total_debt, cash, duration_1, duration_2,
        )
        if not result.valid or result.implied_share_price is None:
            return ReverseDCFResult(False, False, error=result.error, search_range=growth_range)
        difference = result.implied_share_price - price
        if abs(difference) <= tolerance * max(price, 1.0):
            return ReverseDCFResult(True, True, implied_growth=midpoint, search_range=growth_range)
        if result.implied_share_price < price:
            low = midpoint
        else:
            high = midpoint

    midpoint = (low + high) / 2.0
    final_result = calculate_dcf(
        fcf, midpoint, stage_2_growth, terminal_growth, wacc, shares_outstanding,
        total_debt, cash, duration_1, duration_2,
    )
    if final_result.valid and final_result.implied_share_price is not None:
        close_enough = abs(final_result.implied_share_price - price) <= tolerance * max(price, 1.0)
        return ReverseDCFResult(
            True,
            close_enough,
            implied_growth=midpoint if close_enough else None,
            error=None if close_enough else "Reverse DCF did not converge within the supported range.",
            search_range=growth_range,
        )
    return ReverseDCFResult(False, False, error="Reverse DCF did not produce a valid result.", search_range=growth_range)
