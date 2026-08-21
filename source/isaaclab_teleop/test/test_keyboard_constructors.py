# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest
import torch
from isaaclab_teleop.keyboard import Se2Keyboard, Se2KeyboardCfg, Se3Keyboard, Se3KeyboardCfg

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_environment(mocker):
    """Set up common mock objects for tests."""
    carb_mock = mocker.MagicMock()
    omni_mock = mocker.MagicMock()
    appwindow_mock = mocker.MagicMock()
    keyboard_mock = mocker.MagicMock()
    input_mock = mocker.MagicMock()

    omni_mock.appwindow.get_default_app_window.return_value = appwindow_mock
    appwindow_mock.get_keyboard.return_value = keyboard_mock
    carb_mock.input.acquire_input_interface.return_value = input_mock

    # Mock keyboard event types
    carb_mock.input.KeyboardEventType.KEY_PRESS = 1
    carb_mock.input.KeyboardEventType.KEY_RELEASE = 2

    return {"carb": carb_mock, "omni": omni_mock, "appwindow": appwindow_mock, "keyboard": keyboard_mock}


def test_se2keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se2Keyboard."""
    config = Se2KeyboardCfg(
        v_x_sensitivity=0.9,
        v_y_sensitivity=0.5,
        omega_z_sensitivity=1.2,
    )
    device_mod = importlib.import_module("isaaclab_teleop.keyboard.se2_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se2Keyboard(config)

    assert keyboard.v_x_sensitivity == 0.9
    assert keyboard.v_y_sensitivity == 0.5
    assert keyboard.omega_z_sensitivity == 1.2

    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)  # (v_x, v_y, omega_z)


def test_se3keyboard_constructors(mock_environment, mocker):
    """Test constructor for Se3Keyboard."""
    config = Se3KeyboardCfg(
        pos_sensitivity=0.5,
        rot_sensitivity=0.9,
    )
    device_mod = importlib.import_module("isaaclab_teleop.keyboard.se3_keyboard")
    mocker.patch.dict("sys.modules", {"carb": mock_environment["carb"], "omni": mock_environment["omni"]})
    mocker.patch.object(device_mod, "carb", mock_environment["carb"])
    mocker.patch.object(device_mod, "omni", mock_environment["omni"])

    keyboard = Se3Keyboard(config)

    assert keyboard.pos_sensitivity == 0.5
    assert keyboard.rot_sensitivity == 0.9

    result = keyboard.advance()
    assert isinstance(result, torch.Tensor)
    assert result.shape == (7,)  # (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper)


def test_legacy_import_shim_resolves_to_new_module():
    """The deprecated isaaclab.devices.keyboard import path still resolves to the same classes."""
    from isaaclab.devices import Se2Keyboard as LegacySe2Keyboard
    from isaaclab.devices import Se3Keyboard as LegacySe3Keyboard

    assert LegacySe2Keyboard is Se2Keyboard
    assert LegacySe3Keyboard is Se3Keyboard
