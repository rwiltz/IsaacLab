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

import numpy as np
import pytest
from isaaclab_teleop.control_pollers import KeyboardControlPoller

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
