# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :class:`~isaaclab_teleop.control_pollers.SpaceMouseResetPoller` (the right
button's device-intrinsic reset binding).

These exercise the pure logic against a mocked teleop device, without constructing a real
IsaacTeleopDevice, which requires the OpenXR/CloudXR runtime -- matching the pattern
established in test_keyboard_constructors.py.
"""

from __future__ import annotations

import sys
import types

import numpy as np
from isaaclab_teleop.control_pollers import SpaceMouseResetPoller

# ``isaacteleop.plugins`` resolves an on-disk plugin search directory, which is irrelevant to
# the pure config-shape assertions in TestSe2/Se3SpaceMouseTeleopCfg below and may not be
# present in every install of isaacteleop (e.g. minimal/CI builds). Stub it once at import time,
# matching the pattern established in test_gamepad_constructors.py.
if "isaacteleop.plugins" not in sys.modules:
    _fake_plugins = types.ModuleType("isaacteleop.plugins")
    _fake_plugins.plugin_search_path = lambda: "/dummy/plugin/path"
    sys.modules["isaacteleop.plugins"] = _fake_plugins

from isaaclab_teleop.spacemouse.se2_spacemouse import se2_spacemouse_teleop_cfg  # noqa: E402
from isaaclab_teleop.spacemouse.se3_spacemouse import se3_spacemouse_teleop_cfg  # noqa: E402

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


def _make_poller(mocker) -> SpaceMouseResetPoller:
    teleop_device = mocker.MagicMock()
    teleop_device.last_step_result = None
    return SpaceMouseResetPoller(teleop_device)


# ---------------------------------------------------------------------------
# Physical right-button reset binding
# ---------------------------------------------------------------------------


class TestOnResetCallback:
    def test_right_button_fires_on_reset_directly(self, mocker):
        teleop_device = mocker.MagicMock()
        teleop_device.last_step_result = None
        on_reset = mocker.MagicMock()
        poller = SpaceMouseResetPoller(teleop_device, on_reset=on_reset)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)
        on_reset.assert_called_once()

    def test_no_on_reset_is_optional(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}

        poller.advance()  # should not raise with the default on_reset=None


class TestPhysicalRightButtonReset:
    def test_right_button_fires_reset_pause_true(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)

    def test_right_button_only_fires_on_rising_edge(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}

        poller.advance()
        poller.advance()  # still held -> no re-fire

        poller._teleop_device.reset.assert_called_once_with(pause=True)

    def test_left_button_does_not_fire_reset(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([0]))}

        poller.advance()

        poller._teleop_device.reset.assert_not_called()

    def test_no_step_result_is_a_noop(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = None

        poller.advance()  # should not raise

        poller._teleop_device.reset.assert_not_called()

    def test_released_then_pressed_again_fires_again(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}
        poller.advance()

        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([]))}
        poller.advance()

        poller._teleop_device.last_step_result = {"spacemouse_buttons": _FakeOptionalTensorGroup(_buttons_bitmap([1]))}
        poller.advance()

        assert poller._teleop_device.reset.call_count == 2


# ---------------------------------------------------------------------------
# se2_spacemouse_teleop_cfg / se3_spacemouse_teleop_cfg: pure config-shape checks
# ---------------------------------------------------------------------------


class TestSe3SpaceMouseTeleopCfg:
    def test_defaults(self):
        cfg = se3_spacemouse_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabSpaceMouseSe3"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "spacemouse"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_spacemouse_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"


class TestSe2SpaceMouseTeleopCfg:
    def test_defaults(self):
        cfg = se2_spacemouse_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabSpaceMouseSe2"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "spacemouse"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_spacemouse_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
