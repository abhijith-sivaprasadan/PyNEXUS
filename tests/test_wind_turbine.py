import numpy as np
import pytest

from components.wind_turbine import OffshoreWindFarm, load_tabulated_power_curve

CONFIG = "configs/tiny_test.yaml"


def test_hub_height_entry_point_does_not_double_correct() -> None:
    """power_output_mw_from_hub_height must skip the 10m->hub shear step.

    power_output_mw treats the input as a 10m reading and first extrapolates
    it upward to hub height (hub_height_m=120 >> 10m reference), then runs
    the turbine curve. power_output_mw_from_hub_height must run the curve
    directly on the given number with no such uplift. For an identical input
    below rated speed, the 10m-based path must therefore report *more*
    power than the hub-height path — if it reported less or equal, the
    hub-height path would be silently re-applying the shear correction.
    """
    farm = OffshoreWindFarm(CONFIG)
    speed = np.array([9.0])  # below rated (13 m/s) even after uplift

    from_10m = farm.power_output_mw(speed)
    from_hub = farm.power_output_mw_from_hub_height(speed)

    assert from_10m[0] > from_hub[0]


def test_loss_factors_are_config_driven() -> None:
    farm = OffshoreWindFarm(CONFIG)
    assert farm.wake_loss_factor == pytest.approx(0.90)
    assert farm.electrical_loss_factor == pytest.approx(0.98)
    assert farm.availability == pytest.approx(0.95)


def test_tabulated_power_curve_matches_table(tmp_path) -> None:
    csv_path = tmp_path / "curve.csv"
    csv_path.write_text("wind_speed_ms,power_mw\n0,0\n3,0\n10,10\n13,15\n25,15\n25.01,0\n")

    curve = load_tabulated_power_curve(str(csv_path))
    assert curve(10.0) == pytest.approx(10.0)
    assert curve(6.5) == pytest.approx(5.0)  # midpoint of the 3->10 segment, linear
    assert curve(100.0) == pytest.approx(0.0)  # clamped to last tabulated value


def test_tabulated_power_curve_rejects_non_monotonic_speeds(tmp_path) -> None:
    csv_path = tmp_path / "bad_curve.csv"
    csv_path.write_text("wind_speed_ms,power_mw\n5,1\n3,2\n")

    with pytest.raises(ValueError):
        load_tabulated_power_curve(str(csv_path))
