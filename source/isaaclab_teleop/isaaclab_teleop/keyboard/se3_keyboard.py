# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase

if TYPE_CHECKING:
    from .se3_keyboard_cfg import Se3KeyboardCfg

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


class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class provides a keyboard controller for a robotic arm with a gripper, built on the
    IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`). Raw key state is read
    through a bundled ``keyboard`` IsaacTeleop plugin (Linux evdev) and retargeted via
    :class:`~isaacteleop.retargeters.KeyboardToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.KeyboardGripperRetargeter`.

    For a new environment config, prefer declaring ``self.isaac_teleop = IsaacTeleopCfg(
    pipeline_builder=...)`` directly (see the Franka / UR10 / Galbot / Agibot relative-mode
    stack/place configs for the pattern) rather than routing through this class -- this class
    exists for scripts and callers that want a single constructible, ``DeviceBase``-shaped
    object instead of an env-declared pipeline.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    Teleop commands ("B" start/resume, "P" pause, "R" reset) are bound to this device's own
    START/STOP/RESET control events -- register callbacks for them the same way as any other
    IsaacTeleop device, via ``add_callback("START"/"STOP"/"RESET", func)``. Any other key (e.g.
    "N") is a raw press-edge callback, matching the legacy ``add_callback`` contract used for
    non-teleop purposes (e.g. demo-browsing shortcuts).
    """

    def __init__(self, cfg: Se3KeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for keyboard settings.
        """
        # store inputs (public, matching the legacy attribute surface)
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
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
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V\n"
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
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        action = self._teleop_device.advance()
        if action is None:
            action = self._default_action()
        self._poll_keys()
        return action

    """
    Internal helpers.
    """

    def _create_teleop_device(self):
        """Build the IsaacTeleop session driving this keyboard's SE(3) pipeline."""
        from isaacteleop.plugins import plugin_search_path
        from isaacteleop.teleop_session_manager import PluginConfig

        from ..isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
        from ..isaac_teleop_device import create_isaac_teleop_device

        pos_sensitivity = self.pos_sensitivity
        rot_sensitivity = self.rot_sensitivity
        gripper_term = self.gripper_term

        def build_pipeline():
            from isaacteleop.retargeters import (
                KeyboardGripperRetargeter,
                KeyboardToSe3RelRetargeter,
                KeyboardToSe3RelRetargeterConfig,
                TensorReorderer,
            )
            from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
            from isaacteleop.retargeting_engine.interface import OutputCombiner

            keyboard_source = KeyboardSource(_KEYBOARD_COLLECTION_ID)

            se3 = KeyboardToSe3RelRetargeter(
                KeyboardToSe3RelRetargeterConfig(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity),
                name="se3",
            )
            connected_se3 = se3.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})

            ee_delta_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
            input_config = {"ee_delta": ee_delta_elements}
            input_types = {"ee_delta": "array"}
            reorder_inputs = {"ee_delta": connected_se3.output("ee_delta")}
            output_order = list(ee_delta_elements)

            if gripper_term:
                gripper = KeyboardGripperRetargeter(name="gripper")
                connected_gripper = gripper.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})
                input_config["gripper_command"] = ["gripper_value"]
                input_types["gripper_command"] = "scalar"
                reorder_inputs["gripper_command"] = connected_gripper.output("gripper_command")
                output_order.append("gripper_value")

            reorderer = TensorReorderer(
                input_config=input_config,
                output_order=output_order,
                name="action_reorderer",
                input_types=input_types,
            )
            connected_reorderer = reorderer.connect(reorder_inputs)

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
            app_name="IsaacLabKeyboardSe3",
        )
        return create_isaac_teleop_device(
            teleop_cfg,
            cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
            use_kit_xr_bridge=False,
        )

    def _default_action(self) -> torch.Tensor:
        """Zero-safe action returned before the IsaacTeleop session has produced a first step."""
        values = [0.0] * 6
        if self.gripper_term:
            values.append(1.0)  # open
        return torch.tensor(values, dtype=torch.float32, device=self._sim_device)

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
