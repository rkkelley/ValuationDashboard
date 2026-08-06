import pytest

from valuation import calculate_dcf, calculate_dcf_from_fcf, calculate_wacc, reverse_dcf


BASE_CASE = {
    "fcf": 100.0,
    "stage_1_growth": 0.10,
    "stage_2_growth": 0.05,
    "terminal_growth": 0.02,
    "wacc": 0.10,
    "shares_outstanding": 10.0,
    "total_debt": 50.0,
    "cash": 20.0,
    "duration_1": 1,
    "duration_2": 1,
}


def test_known_two_stage_dcf_calculation():
    result = calculate_dcf(**BASE_CASE)

    assert result.valid
    assert result.projected_fcf == pytest.approx((110.0, 115.5))
    assert result.enterprise_value == pytest.approx(1412.5)
    assert result.equity_value == pytest.approx(1382.5)
    assert result.implied_share_price == pytest.approx(138.25)
    assert result.terminal_value_pct_enterprise_value == pytest.approx(1217.0454545 / 1412.5)


def test_increasing_wacc_lowers_implied_share_price():
    low_rate = calculate_dcf(**BASE_CASE)
    high_rate = calculate_dcf(**{**BASE_CASE, "wacc": 0.12})

    assert low_rate.valid and high_rate.valid
    assert high_rate.implied_share_price < low_rate.implied_share_price


def test_increasing_terminal_growth_raises_implied_share_price():
    low_growth = calculate_dcf(**BASE_CASE)
    high_growth = calculate_dcf(**{**BASE_CASE, "terminal_growth": 0.03})

    assert low_growth.valid and high_growth.valid
    assert high_growth.implied_share_price > low_growth.implied_share_price


def test_wacc_at_or_below_terminal_growth_is_invalid():
    equal = calculate_dcf(**{**BASE_CASE, "wacc": 0.02})
    close = calculate_dcf(**{**BASE_CASE, "wacc": 0.024})

    assert not equal.valid
    assert not close.valid
    assert "WACC" in equal.error


def test_reverse_dcf_recovers_known_stage_one_growth():
    base = calculate_dcf(**BASE_CASE)
    result = reverse_dcf(
        current_price=base.implied_share_price,
        fcf=BASE_CASE["fcf"],
        stage_2_growth=BASE_CASE["stage_2_growth"],
        terminal_growth=BASE_CASE["terminal_growth"],
        wacc=BASE_CASE["wacc"],
        shares_outstanding=BASE_CASE["shares_outstanding"],
        total_debt=BASE_CASE["total_debt"],
        cash=BASE_CASE["cash"],
        duration_1=BASE_CASE["duration_1"],
        duration_2=BASE_CASE["duration_2"],
    )

    assert result.valid
    assert result.converged
    assert result.implied_growth == pytest.approx(0.10, abs=1e-5)


@pytest.mark.parametrize(
    ("field", "value"),
    [("fcf", None), ("fcf", float("nan")), ("shares_outstanding", None), ("shares_outstanding", 0.0)],
)
def test_missing_or_invalid_required_inputs_are_rejected(field, value):
    result = calculate_dcf(**{**BASE_CASE, field: value})

    assert not result.valid
    assert result.error


def test_automatic_wacc_marks_missing_market_cap_as_fallback():
    result = calculate_wacc(
        risk_free_rate=0.04,
        beta=1.1,
        equity_risk_premium=0.055,
        market_cap=None,
        total_debt=100.0,
        interest_expense=5.0,
        tax_rate=0.21,
    )

    assert result.valid
    assert result.fallback_used
    assert result.wacc == pytest.approx(0.10)


def test_explicit_fcf_sequence_matches_shared_dcf_valuation():
    result = calculate_dcf_from_fcf(
        annual_fcf=(110.0, 115.5),
        terminal_growth=BASE_CASE["terminal_growth"],
        wacc=BASE_CASE["wacc"],
        shares_outstanding=BASE_CASE["shares_outstanding"],
        total_debt=BASE_CASE["total_debt"],
        cash=BASE_CASE["cash"],
    )

    assert result.valid
    assert result.implied_share_price == pytest.approx(138.25)


def test_explicit_forecast_allows_negative_intermediate_fcf():
    result = calculate_dcf_from_fcf(
        annual_fcf=(100.0, -20.0, 50.0),
        terminal_growth=0.02,
        wacc=0.10,
        shares_outstanding=10.0,
    )

    assert result.valid
    assert result.projected_fcf[1] == -20.0
    assert result.discounted_fcf[1] < 0


def test_sensitivity_changes_discounting_but_not_explicit_projected_fcf():
    annual_fcf = (100.0, -20.0, 50.0)
    low_wacc = calculate_dcf_from_fcf(annual_fcf, 0.02, 0.10, 10.0)
    high_wacc = calculate_dcf_from_fcf(annual_fcf, 0.02, 0.12, 10.0)

    assert low_wacc.valid and high_wacc.valid
    assert high_wacc.projected_fcf == low_wacc.projected_fcf == annual_fcf
    assert high_wacc.discounted_fcf != low_wacc.discounted_fcf
    assert high_wacc.implied_share_price < low_wacc.implied_share_price


def test_nonpositive_final_fcf_rejects_terminal_value():
    result = calculate_dcf_from_fcf((100.0, 0.0, -10.0), 0.02, 0.10, 10.0)

    assert not result.valid
    assert "final forecast FCF" in result.error
