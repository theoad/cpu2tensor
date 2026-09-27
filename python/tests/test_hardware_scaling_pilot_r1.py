# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from cpu2tensor.examples.hardware_scaling_pilot_r1 import local_slope


def test_local_slope_recovers_power_law_direction() -> None:
    assert local_slope([(1, 1.0), (2, 0.5), (4, 0.25)]) == pytest.approx(-1.0)
    assert local_slope([(1, 1.0), (2, 2.0), (4, 4.0)]) == pytest.approx(1.0)
