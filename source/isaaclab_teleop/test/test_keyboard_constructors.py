# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller` (physical B/P/R
control-surface bindings, START/STOP/RESET routing, arbitrary key callbacks).

These exercise the pure edge-detection logic against a mocked teleop device, without
constructing a real IsaacTeleopDevice, which requires the OpenXR/CloudXR runtime -- matching
the pattern established in test_target_frame_rebase.py.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from isaaclab_teleop.control_pollers import KeyboardControlPoller

# ``isaacteleop.plugins`` resolves an on-disk plugin search directory, which is irrelevant to
# the pure config-shape assertions in TestSe2/Se3KeyboardTeleopCfg below and may not be present
# in every install of isaacteleop (e.g. minimal/CI builds). Stub it once at import time, matching
# the pattern established in test_gamepad_constructors.py.
if "isaacteleop.plugins" not in sys.modules:
    _fake_plugins = types.ModuleType("isaacteleop.plugins")
    _fake_plugins.plugin_search_path = lambda: "/dummy/plugin/path"
    sys.modules["isaacteleop.plugins"] = _fake_plugins

from isaaclab_teleop.keyboard.se2_keyboard import se2_keyboard_teleop_cfg  # noqa: E402
from isaaclab_teleop.keyboard.se3_keyboard import se3_keyboard_teleop_cfg  # noqa: E402

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


def _make_poller(mocker) -> KeyboardControlPoller:
    teleop_device = mocker.MagicMock()
    teleop_device.last_step_result = None
    return KeyboardControlPoller(teleop_device)


# ---------------------------------------------------------------------------
# add_callback: always stored locally, regardless of key name
# ---------------------------------------------------------------------------


class TestAddCallback:
    @pytest.mark.parametrize("key", ["START", "STOP", "RESET", "R", "N"])
    def test_callback_stored_locally(self, mocker, key):
        poller = _make_poller(mocker)
        callback = mocker.MagicMock()

        poller.add_callback(key, callback)

        assert poller._additional_callbacks[key] is callback


# ---------------------------------------------------------------------------
# Physical B/P/R control-surface bindings + arbitrary key edge detection
# ---------------------------------------------------------------------------


class TestPhysicalControlKeys:
    def test_b_key_fires_request_start(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([48]))}

        poller.advance()

        poller._teleop_device.request_start.assert_called_once()

    def test_p_key_fires_request_stop(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([25]))}

        poller.advance()

        poller._teleop_device.request_stop.assert_called_once()

    def test_r_key_fires_reset_pause_true(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([19]))}

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)

    def test_r_key_fires_registered_callback_directly(self, mocker):
        """A callback registered for "R" fires on the same rising edge as ``reset(pause=True)``,
        rather than depending on the teleop session's own control-event propagation.
        """
        poller = _make_poller(mocker)
        on_reset = mocker.MagicMock()
        poller.add_callback("R", on_reset)
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([19]))}

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)
        on_reset.assert_called_once()

    def test_arbitrary_callback_fires_on_rising_edge_only(self, mocker):
        poller = _make_poller(mocker)
        callback = mocker.MagicMock()
        poller.add_callback("N", callback)  # evdev code 49

        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([49]))}
        poller.advance()
        callback.assert_called_once()

        # Still held -> no re-fire.
        poller.advance()
        callback.assert_called_once()

        # Released then pressed again -> fires again.
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([]))}
        poller.advance()
        poller._teleop_device.last_step_result = {"keyboard_all_keys": _FakeOptionalTensorGroup(_bitmap_with([49]))}
        poller.advance()
        assert callback.call_count == 2

    def test_no_step_result_is_a_noop(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.last_step_result = None

        poller.advance()  # should not raise

        poller._teleop_device.request_start.assert_not_called()


# ---------------------------------------------------------------------------
# se2_keyboard_teleop_cfg / se3_keyboard_teleop_cfg: pure config-shape checks
# ---------------------------------------------------------------------------


class TestSe3KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se3_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe3"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "keyboard"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"


class TestSe2KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se2_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe2"
        assert len(cfg.plugins) == 1
        assert cfg.plugins[0].plugin_name == "keyboard"
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"
