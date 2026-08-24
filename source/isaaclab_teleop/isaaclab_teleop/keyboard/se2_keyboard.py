# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase

if TYPE_CHECKING:
    from .se2_keyboard_cfg import Se2KeyboardCfg

# Evdev key codes (linux/input-event-codes.h) for every QWERTY letter key plus a few
# common named keys, used to resolve raw key-state bitmap indices back to the
# carb-style names `add_callback` accepts (e.g. "R", "N", "B", "ESCAPE").
_EVDEV_CODE_TO_KEY_NAME = {
    16: "Q", 17: "W", 18: "E", 19: "R", 20: "T", 21: "Y", 22: "U", 23: "I", 24: "O", 25: "P",
    30: "A", 31: "S", 32: "D", 33: "F", 34: "G", 35: "H", 36: "J", 37: "K", 38: "L",
    44: "Z", 45: "X", 46: "C", 47: "V", 48: "B", 49: "N", 50: "M",
    1: "ESCAPE", 15: "TAB", 28: "ENTER", 57: "SPACE",
}  # fmt: skip

# Physical keys wired to fire this device's own START/STOP/RESET control events,
# folding the old "control_keyboard" pattern (a second keyboard bound to another
# device's request_start/request_stop/reset) directly into the primary device.
_START_KEY = "B"
_STOP_KEY = "P"
_RESET_KEY = "R"

_CONTROL_CALLBACK_KEYS = {"START", "STOP", "RESET", "R"}

# Tensor collection ID shared by the bundled keyboard plugin and its KeyboardTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_KEYBOARD_COLLECTION_ID = "keyboard"


class Se2Keyboard(DeviceBase):
    r"""A keyboard controller for sending SE(2) commands as velocity commands.

    This class provides a keyboard controller for a mobile base (such as quadrupeds), built on
    the IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`). Raw key state is
    read through a bundled ``keyboard`` IsaacTeleop plugin (Linux evdev) and retargeted via
    :class:`~isaacteleop.retargeters.KeyboardToSe2Retargeter`.

    For a new environment config, prefer declaring ``self.isaac_teleop = IsaacTeleopCfg(
    pipeline_builder=...)`` directly rather than routing through this class -- this class
    exists for scripts and callers that want a single constructible, ``DeviceBase``-shaped
    object instead of an env-declared pipeline.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
        ====================== ========================= ========================

    Teleop commands ("B" start/resume, "P" pause, "R" reset) are bound to this device's own
    START/STOP/RESET control events -- register callbacks for them the same way as any other
    IsaacTeleop device, via ``add_callback("START"/"STOP"/"RESET", func)``. Any other key (e.g.
    "N") is a raw press-edge callback, matching the legacy ``add_callback`` contract used for
    non-teleop purposes (e.g. demo-browsing shortcuts). "L" resets this device.
    """

    def __init__(self, cfg: Se2KeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.
        """
        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self._sim_device = cfg.sim_device

        self._additional_callbacks: dict[str, Callable] = {}
        self._prev_bitmap: np.ndarray | None = None

        self._teleop_device = self._create_teleop_device()
        self._teleop_device.__enter__()

    def __del__(self):
        """Release the IsaacTeleop session."""
        teleop_device = getattr(self, "_teleop_device", None)
        if teleop_device is not None:
            teleop_device.__exit__(None, None, None)

    def __str__(self) -> str:
        """Returns: A string containing the information of the keyboard controller."""
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward   (along x-axis): Numpad 8 / Arrow Up\n"
        msg += "\tMove backward  (along x-axis): Numpad 2 / Arrow Down\n"
        msg += "\tMove right     (along y-axis): Numpad 4 / Arrow Right\n"
        msg += "\tMove left      (along y-axis): Numpad 6 / Arrow Left\n"
        msg += "\tYaw positively (along z-axis): Numpad 7 / Z\n"
        msg += "\tYaw negatively (along z-axis): Numpad 9 / X\n"
        msg += f"\tStart/resume teleoperation: {_START_KEY}\n"
        msg += f"\tPause teleoperation: {_STOP_KEY}\n"
        msg += f"\tReset: {_RESET_KEY}"
        return msg

    """
    Operations
    """

    def reset(self):
        self._teleop_device.reset()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The keyboard button to check against. ``"START"``, ``"STOP"``, ``"RESET"``,
                and ``"R"`` (aliased to ``"RESET"``) bind to this device's own teleop control
                events, fired by the ``B`` / ``P`` / ``R`` keys respectively. Any other single
                uppercase letter (e.g. ``"N"``) binds a raw press-edge callback.
            func: The function to call. Should take no arguments.
        """
        if key in _CONTROL_CALLBACK_KEYS:
            self._teleop_device.add_callback(key, func)
        else:
            self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            Tensor containing the linear (x,y) and angular velocity (z).
        """
        action = self._teleop_device.advance()
        if action is None:
            action = torch.zeros(3, dtype=torch.float32, device=self._sim_device)
        self._poll_keys()
        return action

    """
    Internal helpers.
    """

    def _create_teleop_device(self):
        """Build the IsaacTeleop session driving this keyboard's SE(2) pipeline."""
        from isaacteleop.plugins import plugin_search_path
        from isaacteleop.teleop_session_manager import PluginConfig

        from ..isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
        from ..isaac_teleop_device import create_isaac_teleop_device

        v_x_sensitivity = self.v_x_sensitivity
        v_y_sensitivity = self.v_y_sensitivity
        omega_z_sensitivity = self.omega_z_sensitivity

        def build_pipeline():
            from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig, TensorReorderer
            from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
            from isaacteleop.retargeting_engine.interface import OutputCombiner

            keyboard_source = KeyboardSource(_KEYBOARD_COLLECTION_ID)

            se2 = KeyboardToSe2Retargeter(
                KeyboardToSe2RetargeterConfig(
                    v_x_sensitivity=v_x_sensitivity,
                    v_y_sensitivity=v_y_sensitivity,
                    omega_z_sensitivity=omega_z_sensitivity,
                ),
                name="se2",
            )
            connected_se2 = se2.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})

            base_command_elements = ["v_x", "v_y", "omega_z"]
            reorderer = TensorReorderer(
                input_config={"base_command": base_command_elements},
                output_order=base_command_elements,
                name="action_reorderer",
                input_types={"base_command": "array"},
            )
            connected_reorderer = reorderer.connect({"base_command": connected_se2.output("base_command")})

            return OutputCombiner(
                {
                    "action": connected_reorderer.output("output"),
                    "keyboard_all_keys": keyboard_source.output("keyboard_all_keys"),
                }
            )

        teleop_cfg = IsaacTeleopCfg(
            pipeline_builder=build_pipeline,
            plugins=[
                PluginConfig(
                    plugin_name="keyboard",
                    plugin_root_id=_KEYBOARD_COLLECTION_ID,
                    search_paths=[plugin_search_path()],
                )
            ],
            sim_device=self._sim_device,
            teleoperation_active_default=True,
            app_name="IsaacLabKeyboardSe2",
        )
        return create_isaac_teleop_device(
            teleop_cfg,
            cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
            use_kit_xr_bridge=False,
        )

    def _poll_keys(self) -> None:
        """Fire START/STOP/RESET and raw-key callbacks on rising edges of the key-state bitmap."""
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
            elif name == "L":
                self._teleop_device.reset()
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
