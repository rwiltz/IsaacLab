# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for Se2Keyboard/Se3Keyboard control logic (physical B/P/R bindings, START/STOP/RESET
routing, arbitrary key callbacks).

These exercise the pure edge-detection logic without constructing a real IsaacTeleopDevice,
which requires the OpenXR/CloudXR runtime -- matching the pattern established in
test_target_frame_rebase.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from isaaclab_teleop.keyboard.se2_keyboard import Se2Keyboard
from isaaclab_teleop.keyboard.se3_keyboard import Se3Keyboard

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


def _bitmap_with(codes: list[int]) -> np.ndarray:
    bitmap = np.zeros(256, dtype=np.uint8)
    for code in codes:
        bitmap[code] = 1
    return bitmap


def _make_bare_se3_keyboard(mocker, gripper_term: bool = True) -> Se3Keyboard:
    """Construct a Se3Keyboard without running __init__ (which needs a live IsaacTeleop session)."""
    keyboard = object.__new__(Se3Keyboard)
    keyboard.pos_sensitivity = 0.4
    keyboard.rot_sensitivity = 0.8
    keyboard.gripper_term = gripper_term
    keyboard._sim_device = "cpu"
    keyboard._additional_callbacks = {}
    keyboard._prev_bitmap = None
    keyboard._teleop_device = mocker.MagicMock()
    keyboard._teleop_device.last_step_result = None
    return keyboard


def _make_bare_se2_keyboard(mocker) -> Se2Keyboard:
    keyboard = object.__new__(Se2Keyboard)
    keyboard.v_x_sensitivity = 0.8
    keyboard.v_y_sensitivity = 0.4
    keyboard.omega_z_sensitivity = 1.0
    keyboard._sim_device = "cpu"
    keyboard._additional_callbacks = {}
    keyboard._prev_bitmap = None
    keyboard._teleop_device = mocker.MagicMock()
    keyboard._teleop_device.last_step_result = None
    return keyboard


# ---------------------------------------------------------------------------
# add_callback routing: START/STOP/RESET/R -> teleop device, else raw key
# ---------------------------------------------------------------------------


class TestAddCallbackRouting:
    @pytest.mark.parametrize("key", ["START", "STOP", "RESET", "R"])
    def test_control_keys_delegate_to_teleop_device(self, mocker, key):
        keyboard = _make_bare_se3_keyboard(mocker)
        callback = mocker.MagicMock()

        keyboard.add_callback(key, callback)

        keyboard._teleop_device.add_callback.assert_called_once_with(key, callback)
        assert key not in keyboard._additional_callbacks

    def test_arbitrary_key_stored_locally(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        callback = mocker.MagicMock()

        keyboard.add_callback("N", callback)

        assert keyboard._additional_callbacks["N"] is callback
        keyboard._teleop_device.add_callback.assert_not_called()


# ---------------------------------------------------------------------------
# Physical B/P/R control-surface bindings + arbitrary key edge detection
# ---------------------------------------------------------------------------


class TestSe3PhysicalControlKeys:
    def test_b_key_fires_request_start(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([48]))}

        keyboard._poll_keys()

        keyboard._teleop_device.request_start.assert_called_once()

    def test_p_key_fires_request_stop(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([25]))}

        keyboard._poll_keys()

        keyboard._teleop_device.request_stop.assert_called_once()

    def test_r_key_fires_reset_pause_true(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([19]))}

        keyboard._poll_keys()

        keyboard._teleop_device.reset.assert_called_once_with(pause=True)

    def test_arbitrary_callback_fires_on_rising_edge_only(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        callback = mocker.MagicMock()
        keyboard.add_callback("N", callback)  # evdev code 49

        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([49]))}
        keyboard._poll_keys()
        callback.assert_called_once()

        # Still held -> no re-fire.
        keyboard._poll_keys()
        callback.assert_called_once()

        # Released then pressed again -> fires again.
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([]))}
        keyboard._poll_keys()
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([49]))}
        keyboard._poll_keys()
        assert callback.call_count == 2

    def test_no_step_result_is_a_noop(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker)
        keyboard._teleop_device.last_step_result = None

        keyboard._poll_keys()  # should not raise

        keyboard._teleop_device.request_start.assert_not_called()


class TestSe2PhysicalControlKeys:
    def test_l_key_fires_reset(self, mocker):
        keyboard = _make_bare_se2_keyboard(mocker)
        keyboard._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([38]))}

        keyboard._poll_keys()

        keyboard._teleop_device.reset.assert_called_once_with()


# ---------------------------------------------------------------------------
# advance() fallback when the IsaacTeleop session has not produced a step yet
# ---------------------------------------------------------------------------


class TestAdvanceDefaults:
    def test_se3_default_action_with_gripper_open(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker, gripper_term=True)
        keyboard._teleop_device.advance.return_value = None

        action = keyboard.advance()

        assert isinstance(action, torch.Tensor)
        assert action.shape == (7,)
        assert torch.allclose(action[:6], torch.zeros(6))
        assert action[6].item() == pytest.approx(1.0)  # gripper open by default

    def test_se3_default_action_without_gripper(self, mocker):
        keyboard = _make_bare_se3_keyboard(mocker, gripper_term=False)
        keyboard._teleop_device.advance.return_value = None

        action = keyboard.advance()

        assert action.shape == (6,)
        assert torch.allclose(action, torch.zeros(6))

    def test_se2_default_action_is_zero(self, mocker):
        keyboard = _make_bare_se2_keyboard(mocker)
        keyboard._teleop_device.advance.return_value = None

        action = keyboard.advance()

        assert action.shape == (3,)
        assert torch.allclose(action, torch.zeros(3))
