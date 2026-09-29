# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-input control pollers for :class:`~isaaclab_teleop.IsaacTeleopDevice` sessions.

They translate operator key presses and SpaceMouse buttons into session-level control events
(``request_start()`` / ``request_stop()`` / ``reset()``) or arbitrary registered callbacks. They
are plain objects, not teleop devices themselves -- construct one alongside an
:class:`IsaacTeleopDevice` and call ``.advance()`` on it once per frame, after the device's own
``.advance()``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np


@functools.cache
def _w3c_codes_by_lowercase() -> dict[str, int]:
    """Evdev code of every W3C ``KeyboardEvent.code``, keyed by the lowercased code."""
    from isaacteleop.deviceio_trackers import EvdevKeyCode

    return {name.lower(): int(member) for name, member in EvdevKeyCode.__members__.items()}


def key_code(key: str) -> int | None:
    """Evdev code for a key name, or ``None`` when the name is not a key.

    Accepts a W3C ``KeyboardEvent.code`` (``"KeyR"``, ``"Digit1"``, ``"Escape"``), a single letter
    or digit (``"R"``, ``"1"``), or a W3C code in any case (``"ESCAPE"``, ``"space"``). Names are
    resolved through Isaac Capture's key table, so every key it knows is accepted.
    """
    if len(key) == 1 and key.isalpha():
        key = f"Key{key.upper()}"
    elif len(key) == 1 and key.isdigit():
        key = f"Digit{key}"
    return _w3c_codes_by_lowercase().get(key.lower())


# Button bit position matching the legacy Se2/Se3SpaceMouse: the right button requests a
# reset (the left button's gripper toggle is handled by SpaceMouseGripperRetargeter inside
# the pipeline itself, so it needs no polling here).
_SPACEMOUSE_BUTTON_RIGHT = 1


class KeyboardControlPoller:
    """Fires control actions and callbacks for the operator's key presses.

    Fires ``teleop_device``'s own ``request_start()`` / ``request_stop()`` / ``reset(pause=True)``
    on presses of the B / P / R keys, and any callback registered via :meth:`add_callback` on
    presses of its key. Presses come from
    :meth:`~isaaclab_teleop.IsaacTeleopDevice.drain_pressed_keys`, so each one fires exactly once
    whatever the pipeline and its retargeting mode, including a tap shorter than a frame. Use
    one poller per device: draining consumes the presses.
    """

    def __init__(self, teleop_device) -> None:
        self._teleop_device = teleop_device
        self._additional_callbacks: dict[int, Callable] = {}
        self._start_code, self._stop_code, self._reset_code = key_code("B"), key_code("P"), key_code("R")

    def add_callback(self, key: str, func: Callable) -> None:
        """Register a callback fired on the rising edge of ``key`` (e.g. ``"N"``, ``"L"``).

        Args:
            key: Key name: a W3C ``KeyboardEvent.code`` (``"KeyN"``, ``"Escape"``), a single letter or
                digit (``"N"``, ``"1"``), or a W3C code in any case (``"ESCAPE"``); see :func:`key_code`.
                ``B``, ``P``, and ``R`` always drive ``request_start`` / ``request_stop`` / ``reset``
                regardless of whether a callback is registered for them.
            func: The function to call. Should take no arguments.

        Raises:
            ValueError: If ``key`` names no key.
        """
        code = key_code(key)
        if code is None:
            raise ValueError(f"Unknown key name: {key!r}")
        self._additional_callbacks[code] = func

    def advance(self) -> None:
        """Fire the matching action for every key pressed since the last call, in order."""
        for code in self._teleop_device.drain_pressed_keys():
            if code == self._start_code:
                self._teleop_device.request_start()
            elif code == self._stop_code:
                self._teleop_device.request_stop()
            elif code == self._reset_code:
                self._teleop_device.reset(pause=True)
            callback = self._additional_callbacks.get(code)
            if callback is not None:
                callback()


class SpaceMouseResetPoller:
    """Polls a spacemouse pipeline's ``spacemouse_buttons`` output.

    Fires ``teleop_device.reset(pause=True)`` on the right button's rising edge, matching the
    legacy Se2/Se3SpaceMouse's device-intrinsic reset binding, plus an optional caller-supplied
    ``on_reset`` callback fired directly (not through the teleop session's own control-event
    propagation, which can lag a frame or more behind the physical button press).

    The button state is sampled from pipeline results, so a press released before a result reaches
    the caller (a quick tap, or a result skipped in pipelined retargeting mode) can be missed. A
    held button never fires twice.
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


def create_control_pollers(
    teleop_device, on_reset: Callable[[], None] | None = None
) -> list[KeyboardControlPoller | SpaceMouseResetPoller]:
    """Create the pollers for headset-free session control.

    The ``B`` / ``P`` / ``R`` keys start-resume / pause / reset the session, whatever the
    pipeline, and the SpaceMouse's right button resets it. Advance every poller once per frame,
    after the device.

    Args:
        teleop_device: The :class:`~isaaclab_teleop.IsaacTeleopDevice` to control.
        on_reset: Also called on every reset, directly rather than through the session's control
            event a frame later.

    Returns:
        The keyboard poller and the SpaceMouse poller.
    """
    keyboard = KeyboardControlPoller(teleop_device)
    if on_reset is not None:
        keyboard.add_callback("R", on_reset)
    return [keyboard, SpaceMouseResetPoller(teleop_device, on_reset=on_reset)]
