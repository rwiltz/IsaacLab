# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-input control pollers for plugin-backed :class:`~isaaclab_teleop.IsaacTeleopDevice` pipelines.

A plugin-backed pipeline (keyboard, spacemouse, ...) exposes its raw button state as an
extra output alongside ``"action"`` so a caller can translate physical button presses into
session-level control events (``request_start()`` / ``request_stop()`` / ``reset()``) or
arbitrary registered callbacks. These pollers do that translation; they are plain objects,
not teleop devices themselves -- construct one alongside an :class:`IsaacTeleopDevice` and
call ``.advance()`` on it once per frame, after the device's own ``.advance()``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Evdev key codes (linux/input-event-codes.h) for every QWERTY letter key plus a few
# common named keys, used to resolve raw key-state bitmap indices back to the
# carb-style names `add_callback` accepts (e.g. "R", "N", "B", "ESCAPE").
_EVDEV_CODE_TO_KEY_NAME = {
    16: "Q", 17: "W", 18: "E", 19: "R", 20: "T", 21: "Y", 22: "U", 23: "I", 24: "O", 25: "P",
    30: "A", 31: "S", 32: "D", 33: "F", 34: "G", 35: "H", 36: "J", 37: "K", 38: "L",
    44: "Z", 45: "X", 46: "C", 47: "V", 48: "B", 49: "N", 50: "M",
    1: "ESCAPE", 15: "TAB", 28: "ENTER", 57: "SPACE",
}  # fmt: skip

_START_KEY = "B"
_STOP_KEY = "P"
_RESET_KEY = "R"

# Button bit position matching the legacy Se2/Se3SpaceMouse: the right button requests a
# reset (the left button's gripper toggle is handled by SpaceMouseGripperRetargeter inside
# the pipeline itself, so it needs no polling here).
_SPACEMOUSE_BUTTON_RIGHT = 1


class KeyboardControlPoller:
    """Polls a keyboard-plugin-backed pipeline's ``keyboard_all_keys`` bitmap output.

    Fires ``teleop_device``'s own ``request_start()`` / ``request_stop()`` / ``reset(pause=True)``
    on rising edges of the physical B / P / R keys, and any callback registered via
    :meth:`add_callback` on rising edges of its key.

    Works against any :class:`~isaaclab_teleop.IsaacTeleopDevice` whose pipeline exposes a
    ``"keyboard_all_keys"`` output -- the primary teleop device itself when it is
    keyboard-plugin-backed, or a separate, auxiliary keyboard-only device layered on top of a
    non-keyboard primary device for headset-free control.
    """

    def __init__(self, teleop_device) -> None:
        self._teleop_device = teleop_device
        self._prev_bitmap: np.ndarray | None = None
        self._additional_callbacks: dict[str, Callable] = {}

    def add_callback(self, key: str, func: Callable) -> None:
        """Register a callback fired on the rising edge of ``key`` (e.g. ``"N"``, ``"L"``).

        Args:
            key: Single uppercase letter or named key (see the evdev-to-name table). ``B``,
                ``P``, and ``R`` always drive ``request_start`` / ``request_stop`` / ``reset``
                regardless of whether a callback is registered for them.
            func: The function to call. Should take no arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> None:
        """Check for rising edges since the last call and fire the matching action."""
        bitmap = self._read_bitmap()
        if bitmap is None:
            return

        prev = self._prev_bitmap
        self._prev_bitmap = bitmap
        if prev is None or prev.shape != bitmap.shape:
            prev = np.zeros_like(bitmap)
        rising_codes = np.nonzero((bitmap != 0) & (prev == 0))[0]

        for code in rising_codes:
            name = _EVDEV_CODE_TO_KEY_NAME.get(int(code))
            if name is None:
                continue
            if name == _START_KEY:
                self._teleop_device.request_start()
            elif name == _STOP_KEY:
                self._teleop_device.request_stop()
            elif name == _RESET_KEY:
                self._teleop_device.reset(pause=True)
            callback = self._additional_callbacks.get(name)
            if callback is not None:
                callback()

    def _read_bitmap(self) -> np.ndarray | None:
        result = self._teleop_device.last_step_result
        if result is None:
            return None
        all_keys = result.get("keyboard_all_keys")
        if all_keys is None or all_keys.is_none:
            return None
        return np.asarray(all_keys[0])


class SpaceMouseResetPoller:
    """Polls a spacemouse-plugin-backed pipeline's ``spacemouse_buttons`` output.

    Fires ``teleop_device.reset(pause=True)`` on the right button's rising edge, matching the
    legacy Se2/Se3SpaceMouse's device-intrinsic reset binding, plus an optional caller-supplied
    ``on_reset`` callback fired directly (not through the teleop session's own control-event
    propagation, which can lag a frame or more behind the physical button press).
    """

    def __init__(self, teleop_device, on_reset: Callable[[], None] | None = None) -> None:
        self._teleop_device = teleop_device
        self._on_reset = on_reset
        self._prev_bitmap: np.ndarray | None = None

    def advance(self) -> None:
        bitmap = self._read_bitmap()
        if bitmap is None:
            return

        prev = self._prev_bitmap
        self._prev_bitmap = bitmap
        if prev is None or prev.shape != bitmap.shape:
            prev = np.zeros_like(bitmap)

        if bitmap[_SPACEMOUSE_BUTTON_RIGHT] and not prev[_SPACEMOUSE_BUTTON_RIGHT]:
            self._teleop_device.reset(pause=True)
            if self._on_reset is not None:
                self._on_reset()

    def _read_bitmap(self) -> np.ndarray | None:
        result = self._teleop_device.last_step_result
        if result is None:
            return None
        buttons = result.get("spacemouse_buttons")
        if buttons is None or buttons.is_none:
            return None
        return np.asarray(buttons[0])
