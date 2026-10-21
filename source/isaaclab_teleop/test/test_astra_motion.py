# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect per-tick wrist speed limits independently of the simulator and IK."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.motion import WristTrajectory  # noqa: E402


def neutral_action():
    action = torch.zeros(1, 28, dtype=torch.float64)
    action[0, [6, 13]] = 1
    return action


def test_diagonal_translation_and_shortest_rotation_respect_speeds():
    initial = neutral_action()
    target = initial.clone()
    target[0, :3] = torch.tensor([0.03, 0.04, 0])
    target[0, 7:10] = torch.tensor([0, -0.02, 0])
    target[0, 3:7] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    # Negative quaternion representation of +170 degrees around Z.
    target[0, 10:14] = torch.tensor([0, 0, -math.sin(math.radians(85)), -math.cos(math.radians(85))])
    trajectory = WristTrajectory(initial, linear_speed=0.025, angular_speed=0.4)
    previous = initial
    for step in range(80):
        target[0, 14:] = step / 100
        result = trajectory.advance(target, 0.1)
        wrists = result[0, :14].reshape(2, 7)
        old_wrists = previous[0, :14].reshape(2, 7)
        assert torch.all(torch.linalg.vector_norm(wrists[:, :3] - old_wrists[:, :3], dim=-1) <= 0.0025 + 1e-12)
        dots = (wrists[:, 3:] * old_wrists[:, 3:]).sum(-1).abs().clamp(max=1)
        assert torch.all(2 * torch.acos(dots) <= 0.04 + 1e-10)
        assert torch.linalg.vector_norm(wrists[:, 3:], dim=-1).tolist() == pytest.approx([1, 1])
        assert torch.equal(result[:, 14:], target[:, 14:])
        if step == 0:
            assert result[0, :3].tolist() == pytest.approx([0.0015, 0.002, 0])
            assert result[0, 12] > 0  # Short arc, despite the target's negative representation.
        previous = result
    assert result[0, :3].tolist() == pytest.approx(target[0, :3].tolist())
    assert result[0, 7:10].tolist() == pytest.approx(target[0, 7:10].tolist())
    assert result[0, 10:14].tolist() == pytest.approx((-target[0, 10:14]).tolist())


def test_retargeting_is_continuous_and_reset_reanchors_without_aliases():
    initial = neutral_action()
    trajectory = WristTrajectory(initial)
    target = initial.clone()
    target[0, 0] = 1
    first = trajectory.advance(target, 1)
    assert first[0, 0] == pytest.approx(0.025)
    first[0, 0] = 50  # The caller cannot change the trajectory by mutating an output.
    target[0, 0] = -1
    second = trajectory.advance(target, 0.2)
    assert second[0, 0] == pytest.approx(0.02)
    assert target[0, 0] == -1

    measured = neutral_action()
    measured[0, :3] = torch.tensor([0.3, 0.4, 0.5])
    measured[0, 3:7] = torch.tensor([0, math.sqrt(2), 0, math.sqrt(2)])
    trajectory.reset(measured)
    close_target = measured.clone()
    close_target[0, 0] += 0.001
    close_target[0, 3:7] *= -1
    measured.fill_(100)  # Reset also copies its input.
    result = trajectory.advance(close_target, 0.1)
    assert result[0, :3].tolist() == pytest.approx(close_target[0, :3].tolist())
    assert result[0, 3:7].tolist() == pytest.approx([0, math.sqrt(0.5), 0, math.sqrt(0.5)])


def test_invalid_motion_inputs_leave_previous_trajectory_usable():
    initial = neutral_action()
    with pytest.raises(ValueError, match="speeds"):
        WristTrajectory(initial, angular_speed=float("nan"))
    trajectory = WristTrajectory(initial)
    with pytest.raises(ValueError, match="timestep"):
        trajectory.advance(initial, 0)
    with pytest.raises(ValueError, match="shape"):
        trajectory.advance(initial[:, :-1], 0.1)
    invalid = initial.clone()
    invalid[0, 3:7] = 0
    with pytest.raises(ValueError, match="nonzero"):
        trajectory.reset(invalid)
    invalid[0, 3] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        trajectory.advance(invalid, 0.1)
    assert torch.equal(trajectory.advance(initial, 0.1), initial)
