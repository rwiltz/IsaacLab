# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller` (physical B/P/R
control-surface bindings, START/STOP/RESET routing, arbitrary key callbacks) and the
keyboard ``IsaacTeleopCfg`` builders.

The poller tests exercise the key-dispatch logic against a mocked teleop device, without
constructing a real IsaacTeleopDevice -- matching the pattern established in
test_target_frame_rebase.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop.control_pollers import KeyboardControlPoller
from isaaclab_teleop.keyboard.se2_keyboard import se2_keyboard_teleop_cfg
from isaaclab_teleop.keyboard.se3_keyboard import se3_keyboard_teleop_cfg

KEY_1, KEY_R, KEY_P, KEY_B, KEY_N, KEY_ESC = 2, 19, 25, 48, 49, 1

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDevice:
    """Stand-in for an IsaacTeleopDevice: hands out queued key presses once."""

    def __init__(self, mocker):
        self.pending: list[int] = []
        self.request_start = mocker.MagicMock()
        self.request_stop = mocker.MagicMock()
        self.reset = mocker.MagicMock()

    def press(self, *codes: int) -> None:
        self.pending.extend(codes)

    def drain_pressed_keys(self) -> list[int]:
        pressed, self.pending = self.pending, []
        return pressed


def _make_poller(mocker) -> KeyboardControlPoller:
    return KeyboardControlPoller(_FakeDevice(mocker))


# ---------------------------------------------------------------------------
# add_callback: key names and their aliases
# ---------------------------------------------------------------------------


class TestAddCallback:
    @pytest.mark.parametrize(
        ("key", "code"), [("R", KEY_R), ("KeyN", KEY_N), ("n", KEY_N), ("ESCAPE", KEY_ESC), ("Escape", KEY_ESC)]
    )
    def test_key_name_aliases_fire_on_their_key(self, mocker, key, code):
        poller = _make_poller(mocker)
        callback = mocker.MagicMock()
        poller.add_callback(key, callback)

        poller._teleop_device.press(code)
        poller.advance()

        callback.assert_called_once()

    @pytest.mark.parametrize("key", ["START", "NotAKey", ""])
    def test_unknown_key_name_is_rejected(self, mocker, key):
        with pytest.raises(ValueError):
            _make_poller(mocker).add_callback(key, mocker.MagicMock())


# ---------------------------------------------------------------------------
# Physical B/P/R control-surface bindings + arbitrary key callbacks
# ---------------------------------------------------------------------------


class TestPhysicalControlKeys:
    def test_b_key_fires_request_start(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.press(KEY_B)

        poller.advance()

        poller._teleop_device.request_start.assert_called_once()

    def test_p_key_fires_request_stop(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.press(KEY_P)

        poller.advance()

        poller._teleop_device.request_stop.assert_called_once()

    def test_r_key_fires_reset_pause_true(self, mocker):
        poller = _make_poller(mocker)
        poller._teleop_device.press(KEY_R)

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)

    def test_r_key_fires_registered_callback_directly(self, mocker):
        """A callback registered for "R" fires on the same press as ``reset(pause=True)``."""
        poller = _make_poller(mocker)
        on_reset = mocker.MagicMock()
        poller.add_callback("R", on_reset)
        poller._teleop_device.press(KEY_R)

        poller.advance()

        poller._teleop_device.reset.assert_called_once_with(pause=True)
        on_reset.assert_called_once()

    def test_each_press_fires_once(self, mocker):
        """Presses are drained, so one press fires once however many frames follow it."""
        poller = _make_poller(mocker)
        callback = mocker.MagicMock()
        poller.add_callback("N", callback)

        poller._teleop_device.press(KEY_N)
        poller.advance()
        poller.advance()
        callback.assert_called_once()

        poller._teleop_device.press(KEY_N, KEY_N)  # two taps within one frame
        poller.advance()
        assert callback.call_count == 3

    def test_presses_fire_in_order(self, mocker):
        poller = _make_poller(mocker)
        fired = []
        poller.add_callback("N", lambda: fired.append("N"))
        poller.add_callback("1", lambda: fired.append("1"))
        poller._teleop_device.press(KEY_1, KEY_N, KEY_1)

        poller.advance()

        assert fired == ["1", "N", "1"]

    @pytest.mark.parametrize(("name", "code"), [("1", KEY_1), ("ESCAPE", KEY_ESC)])
    def test_named_keys(self, mocker, name, code):
        """Digits and named keys use the evdev name without its ``KEY_`` prefix; ESC is ``ESCAPE``."""
        poller = _make_poller(mocker)
        callback = mocker.MagicMock()
        poller.add_callback(name, callback)
        poller._teleop_device.press(code)

        poller.advance()

        callback.assert_called_once()

    def test_no_presses_is_a_noop(self, mocker):
        poller = _make_poller(mocker)

        poller.advance()

        poller._teleop_device.request_start.assert_not_called()
        poller._teleop_device.reset.assert_not_called()


# ---------------------------------------------------------------------------
# se2_keyboard_teleop_cfg / se3_keyboard_teleop_cfg
# ---------------------------------------------------------------------------


def _keyboard_sources(pipeline):
    from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource, find_sources

    return find_sources(pipeline, KeyboardSource)


class TestSe3KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se3_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe3"
        # the keyboard runs in process: no plugin
        assert cfg.plugins == []
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se3_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"

    def test_pipeline_exposes_key_bitmaps(self):
        pipeline = se3_keyboard_teleop_cfg().pipeline_builder()

        assert set(pipeline.output_types()) == {"action", "keyboard_held", "keyboard_pressed"}
        assert len(_keyboard_sources(pipeline)) == 1

    @pytest.mark.parametrize(("gripper_term", "action_size"), [(True, 7), (False, 6)])
    def test_action_is_the_pose_delta_plus_an_optional_gripper(self, gripper_term, action_size):
        """The action is ``[dx, dy, dz, drx, dry, drz]``, plus the gripper command when ``gripper_term``."""
        pipeline = se3_keyboard_teleop_cfg(gripper_term=gripper_term).pipeline_builder()

        (action,) = pipeline.output_types()["action"].types
        assert action.shape == (action_size,)


class TestSe2KeyboardTeleopCfg:
    def test_defaults(self):
        cfg = se2_keyboard_teleop_cfg()

        assert cfg.sim_device == "cpu"
        assert cfg.teleoperation_active_default is True
        assert cfg.app_name == "IsaacLabKeyboardSe2"
        assert cfg.plugins == []
        assert callable(cfg.pipeline_builder)

    def test_custom_sim_device(self):
        cfg = se2_keyboard_teleop_cfg(sim_device="cuda:0")

        assert cfg.sim_device == "cuda:0"

    def test_pipeline_exposes_key_bitmaps(self):
        pipeline = se2_keyboard_teleop_cfg().pipeline_builder()

        assert set(pipeline.output_types()) == {"action", "keyboard_held", "keyboard_pressed"}
        assert len(_keyboard_sources(pipeline)) == 1
