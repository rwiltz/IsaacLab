# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for Se2SpaceMouse/Se3SpaceMouse control logic (add_callback routing, the right
button's device-intrinsic reset binding, advance() fallback).

These exercise the pure logic without constructing a real IsaacTeleopDevice, which requires
the OpenXR/CloudXR runtime -- matching the pattern established in test_keyboard_constructors.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from isaaclab_teleop.spacemouse.se2_spacemouse import Se2SpaceMouse
from isaaclab_teleop.spacemouse.se3_spacemouse import Se3SpaceMouse

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeOptionalTensorGroup:
    """Stand-in for the OptionalTensorGroup returned in a pipeline step result."""

    def __init__(self, value):
        self._value = value
        self.is_none = value is None

    def __getitem__(self, index):
        return self._value


def _buttons_bitmap(pressed: list[int]) -> np.ndarray:
    bitmap = np.zeros(8, dtype=np.uint8)
    for index in pressed:
        bitmap[index] = 1
    return bitmap


def _make_bare_se3_spacemouse(mocker, gripper_term: bool = True) -> Se3SpaceMouse:
    """Construct a Se3SpaceMouse without running __init__ (which needs a live IsaacTeleop session)."""
    spacemouse = object.__new__(Se3SpaceMouse)
    spacemouse.pos_sensitivity = 0.4
    spacemouse.rot_sensitivity = 0.8
    spacemouse.gripper_term = gripper_term
    spacemouse._sim_device = "cpu"
    spacemouse._additional_callbacks = {}
    spacemouse._prev_bitmap = None
    spacemouse._teleop_device = mocker.MagicMock()
    spacemouse._teleop_device.last_step_result = None
    return spacemouse


def _make_bare_se2_spacemouse(mocker) -> Se2SpaceMouse:
    spacemouse = object.__new__(Se2SpaceMouse)
    spacemouse.v_x_sensitivity = 0.8
    spacemouse.v_y_sensitivity = 0.4
    spacemouse.omega_z_sensitivity = 1.0
    spacemouse._sim_device = "cpu"
    spacemouse._additional_callbacks = {}
    spacemouse._prev_bitmap = None
    spacemouse._teleop_device = mocker.MagicMock()
    spacemouse._teleop_device.last_step_result = None
    return spacemouse


# ---------------------------------------------------------------------------
# add_callback routing: START/STOP/RESET/R -> teleop device, else stored locally
# ---------------------------------------------------------------------------


class TestAddCallbackRouting:
    @pytest.mark.parametrize("key", ["START", "STOP", "RESET", "R"])
    def test_control_keys_delegate_to_teleop_device(self, mocker, key):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        callback = mocker.MagicMock()

        spacemouse.add_callback(key, callback)

        spacemouse._teleop_device.add_callback.assert_called_once_with(key, callback)
        assert key not in spacemouse._additional_callbacks

    def test_arbitrary_key_stored_locally(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        callback = mocker.MagicMock()

        spacemouse.add_callback("L", callback)

        assert spacemouse._additional_callbacks["L"] is callback
        spacemouse._teleop_device.add_callback.assert_not_called()


# ---------------------------------------------------------------------------
# Physical right-button reset binding
# ---------------------------------------------------------------------------


class TestPhysicalRightButtonReset:
    def test_right_button_fires_reset_pause_true(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        spacemouse._teleop_device.last_step_result = {
            "spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))
        }

        spacemouse._poll_buttons()

        spacemouse._teleop_device.reset.assert_called_once_with(pause=True)

    def test_right_button_only_fires_on_rising_edge(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        spacemouse._teleop_device.last_step_result = {
            "spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))
        }

        spacemouse._poll_buttons()
        spacemouse._poll_buttons()  # still held -> no re-fire

        spacemouse._teleop_device.reset.assert_called_once_with(pause=True)

    def test_left_button_does_not_fire_reset(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        spacemouse._teleop_device.last_step_result = {
            "spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([0]))
        }

        spacemouse._poll_buttons()

        spacemouse._teleop_device.reset.assert_not_called()

    def test_no_step_result_is_a_noop(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker)
        spacemouse._teleop_device.last_step_result = None

        spacemouse._poll_buttons()  # should not raise

        spacemouse._teleop_device.reset.assert_not_called()

    def test_se2_right_button_fires_reset_pause_true(self, mocker):
        spacemouse = _make_bare_se2_spacemouse(mocker)
        spacemouse._teleop_device.last_step_result = {
            "spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))
        }

        spacemouse._poll_buttons()

        spacemouse._teleop_device.reset.assert_called_once_with(pause=True)


# ---------------------------------------------------------------------------
# advance() fallback when the IsaacTeleop session has not produced a step yet
# ---------------------------------------------------------------------------


class TestAdvanceDefaults:
    def test_se3_default_action_with_gripper_open(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker, gripper_term=True)
        spacemouse._teleop_device.advance.return_value = None

        action = spacemouse.advance()

        assert isinstance(action, torch.Tensor)
        assert action.shape == (7,)
        assert torch.allclose(action[:6], torch.zeros(6))
        assert action[6].item() == pytest.approx(1.0)  # gripper open by default

    def test_se3_default_action_without_gripper(self, mocker):
        spacemouse = _make_bare_se3_spacemouse(mocker, gripper_term=False)
        spacemouse._teleop_device.advance.return_value = None

        action = spacemouse.advance()

        assert action.shape == (6,)
        assert torch.allclose(action, torch.zeros(6))

    def test_se2_default_action_is_zero(self, mocker):
        spacemouse = _make_bare_se2_spacemouse(mocker)
        spacemouse._teleop_device.advance.return_value = None

        action = spacemouse.advance()

        assert action.shape == (3,)
        assert torch.allclose(action, torch.zeros(3))
