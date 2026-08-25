# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for Se2Gamepad/Se3Gamepad control logic (add_callback routing, advance() fallback).

These exercise the pure logic without constructing a real IsaacTeleopDevice, which requires
the OpenXR/CloudXR runtime -- matching the pattern established in test_keyboard_constructors.py.
"""

from __future__ import annotations

import pytest
import torch
from isaaclab_teleop.gamepad.se2_gamepad import Se2Gamepad
from isaaclab_teleop.gamepad.se3_gamepad import Se3Gamepad


def _make_bare_se3_gamepad(mocker, gripper_term: bool = True) -> Se3Gamepad:
    """Construct a Se3Gamepad without running __init__ (which needs a live IsaacTeleop session)."""
    gamepad = object.__new__(Se3Gamepad)
    gamepad.pos_sensitivity = 0.4
    gamepad.rot_sensitivity = 0.8
    gamepad.dead_zone = 0.01
    gamepad.gripper_term = gripper_term
    gamepad._sim_device = "cpu"
    gamepad._additional_callbacks = {}
    gamepad._teleop_device = mocker.MagicMock()
    gamepad._teleop_device.last_step_result = None
    return gamepad


def _make_bare_se2_gamepad(mocker) -> Se2Gamepad:
    gamepad = object.__new__(Se2Gamepad)
    gamepad.v_x_sensitivity = 1.0
    gamepad.v_y_sensitivity = 1.0
    gamepad.omega_z_sensitivity = 1.0
    gamepad.dead_zone = 0.01
    gamepad._sim_device = "cpu"
    gamepad._additional_callbacks = {}
    gamepad._teleop_device = mocker.MagicMock()
    gamepad._teleop_device.last_step_result = None
    return gamepad


# ---------------------------------------------------------------------------
# add_callback routing: START/STOP/RESET/R -> teleop device, else stored locally
# ---------------------------------------------------------------------------


class TestAddCallbackRouting:
    @pytest.mark.parametrize("key", ["START", "STOP", "RESET", "R"])
    def test_control_keys_delegate_to_teleop_device(self, mocker, key):
        gamepad = _make_bare_se3_gamepad(mocker)
        callback = mocker.MagicMock()

        gamepad.add_callback(key, callback)

        gamepad._teleop_device.add_callback.assert_called_once_with(key, callback)
        assert key not in gamepad._additional_callbacks

    def test_arbitrary_key_stored_locally(self, mocker):
        gamepad = _make_bare_se3_gamepad(mocker)
        callback = mocker.MagicMock()

        gamepad.add_callback("X", callback)

        assert gamepad._additional_callbacks["X"] is callback
        gamepad._teleop_device.add_callback.assert_not_called()


# ---------------------------------------------------------------------------
# advance() fallback when the IsaacTeleop session has not produced a step yet
# ---------------------------------------------------------------------------


class TestAdvanceDefaults:
    def test_se3_default_action_with_gripper_open(self, mocker):
        gamepad = _make_bare_se3_gamepad(mocker, gripper_term=True)
        gamepad._teleop_device.advance.return_value = None

        action = gamepad.advance()

        assert isinstance(action, torch.Tensor)
        assert action.shape == (7,)
        assert torch.allclose(action[:6], torch.zeros(6))
        assert action[6].item() == pytest.approx(1.0)  # gripper open by default

    def test_se3_default_action_without_gripper(self, mocker):
        gamepad = _make_bare_se3_gamepad(mocker, gripper_term=False)
        gamepad._teleop_device.advance.return_value = None

        action = gamepad.advance()

        assert action.shape == (6,)
        assert torch.allclose(action, torch.zeros(6))

    def test_se2_default_action_is_zero(self, mocker):
        gamepad = _make_bare_se2_gamepad(mocker)
        gamepad._teleop_device.advance.return_value = None

        action = gamepad.advance()

        assert action.shape == (3,)
        assert torch.allclose(action, torch.zeros(3))
