# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(3) control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.devices.device_base import DeviceBase

if TYPE_CHECKING:
    from .se3_gamepad_cfg import Se3GamepadCfg

# START/STOP/RESET are driven by this device's teleop control state machine (e.g. an
# auxiliary control device or the session's control channel), not a physical gamepad
# button -- matching the legacy Se3Gamepad, which never wired any button to them either.
_CONTROL_CALLBACK_KEYS = {"START", "STOP", "RESET", "R"}

# Tensor collection ID shared by the bundled gamepad plugin and its GamepadTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_GAMEPAD_COLLECTION_ID = "gamepad"


class Se3Gamepad(DeviceBase):
    """A gamepad controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class provides a gamepad controller for a robotic arm with a gripper, built on the
    IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`). Raw stick/button state is
    read through a bundled ``gamepad`` IsaacTeleop plugin (Linux joystick API) and retargeted via
    :class:`~isaacteleop.retargeters.GamepadToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.GamepadGripperRetargeter`.

    For a new environment config, prefer declaring ``self.isaac_teleop = IsaacTeleopCfg(
    pipeline_builder=...)`` directly (see the Franka / UR10 / Galbot / Agibot relative-mode
    stack/place configs for the pattern) rather than routing through this class -- this class
    exists for scripts and callers that want a single constructible, ``DeviceBase``-shaped
    object instead of an env-declared pipeline.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Stick and Button bindings:
        ============================ ========================= =========================
        Description                  Stick/Button (+ve axis)   Stick/Button (-ve axis)
        ============================ ========================= =========================
        Toggle gripper(open/close)   X Button                  X Button
        Move along x-axis            Left Stick Up             Left Stick Down
        Move along y-axis            Left Stick Left           Left Stick Right
        Move along z-axis            Right Stick Up            Right Stick Down
        Rotate along x-axis          D-Pad Left                D-Pad Right
        Rotate along y-axis          D-Pad Down                D-Pad Up
        Rotate along z-axis          Right Stick Left          Right Stick Right
        ============================ ========================= =========================

    Register callbacks for teleop control events via ``add_callback("START"/"STOP"/"RESET", func)``,
    the same way as any other IsaacTeleop device -- they fire when the underlying teleop session
    transitions state, regardless of what drives that transition (e.g. an auxiliary keyboard, or
    the session's control channel).
    """

    def __init__(self, cfg: Se3GamepadCfg):
        """Initialize the gamepad layer.

        Args:
            cfg: Configuration object for gamepad settings.
        """
        # store inputs (public, matching the legacy attribute surface)
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.dead_zone = cfg.dead_zone
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device

        self._additional_callbacks: dict[str, Callable] = {}

        self._teleop_device = self._create_teleop_device()
        self._teleop_device.__enter__()

    def __del__(self):
        """Release the IsaacTeleop session."""
        teleop_device = getattr(self, "_teleop_device", None)
        if teleop_device is not None:
            teleop_device.__exit__(None, None, None)

    def __str__(self) -> str:
        """Returns: A string containing the information of the gamepad controller."""
        msg = f"Gamepad Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): X\n"
        msg += "\tMove arm along x-axis: Left Stick Up/Down\n"
        msg += "\tMove arm along y-axis: Left Stick Left/Right\n"
        msg += "\tMove arm along z-axis: Right Stick Up/Down\n"
        msg += "\tRotate arm along x-axis: D-Pad Right/Left\n"
        msg += "\tRotate arm along y-axis: D-Pad Down/Up\n"
        msg += "\tRotate arm along z-axis: Right Stick Left/Right\n"
        return msg

    """
    Operations
    """

    def reset(self):
        self._teleop_device.reset()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind gamepad.

        Args:
            key: The teleop control event to check against. ``"START"``, ``"STOP"``, ``"RESET"``,
                and ``"R"`` (aliased to ``"RESET"``) bind to this device's own teleop control
                events. Any other key is stored but never invoked, matching the legacy gamepad
                device (which only ever fired callbacks for raw carb button events, none of which
                this pipeline reproduces).
            func: The function to call. Should take no arguments.
        """
        if key in _CONTROL_CALLBACK_KEYS:
            self._teleop_device.add_callback(key, func)
        else:
            self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from gamepad event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        action = self._teleop_device.advance()
        if action is None:
            action = self._default_action()
        return action

    """
    Internal helpers.
    """

    def _create_teleop_device(self):
        """Build the IsaacTeleop session driving this gamepad's SE(3) pipeline."""
        from isaacteleop.plugins import plugin_search_path
        from isaacteleop.teleop_session_manager import PluginConfig

        from ..isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
        from ..isaac_teleop_device import create_isaac_teleop_device

        pos_sensitivity = self.pos_sensitivity
        rot_sensitivity = self.rot_sensitivity
        dead_zone = self.dead_zone
        gripper_term = self.gripper_term

        def build_pipeline():
            from isaacteleop.retargeters import (
                GamepadGripperRetargeter,
                GamepadToSe3RelRetargeter,
                GamepadToSe3RelRetargeterConfig,
                TensorReorderer,
            )
            from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource
            from isaacteleop.retargeting_engine.interface import OutputCombiner

            gamepad_source = GamepadSource(_GAMEPAD_COLLECTION_ID)

            se3 = GamepadToSe3RelRetargeter(
                GamepadToSe3RelRetargeterConfig(
                    pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity, dead_zone=dead_zone
                ),
                name="se3",
            )
            connected_se3 = se3.connect({"gamepad_axes": gamepad_source.output("gamepad_axes")})

            ee_delta_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
            input_config = {"ee_delta": ee_delta_elements}
            input_types = {"ee_delta": "array"}
            reorder_inputs = {"ee_delta": connected_se3.output("ee_delta")}
            output_order = list(ee_delta_elements)

            if gripper_term:
                gripper = GamepadGripperRetargeter(name="gripper")
                connected_gripper = gripper.connect({"gamepad_buttons": gamepad_source.output("gamepad_buttons")})
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

            return OutputCombiner({"action": connected_reorderer.output("output")})

        teleop_cfg = IsaacTeleopCfg(
            pipeline_builder=build_pipeline,
            plugins=[
                PluginConfig(
                    plugin_name="gamepad",
                    plugin_root_id=_GAMEPAD_COLLECTION_ID,
                    search_paths=[plugin_search_path()],
                )
            ],
            sim_device=self._sim_device,
            teleoperation_active_default=True,
            app_name="IsaacLabGamepadSe3",
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
