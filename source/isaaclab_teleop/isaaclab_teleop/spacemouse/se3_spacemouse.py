# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse controller for SE(3) control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase

if TYPE_CHECKING:
    from .se3_spacemouse_cfg import Se3SpaceMouseCfg

# START/STOP are driven by this device's teleop control state machine (e.g. an auxiliary
# control device or the session's control channel), matching the legacy Se3SpaceMouse,
# which never wired either to a physical button. RESET is device-intrinsic: the right
# button has always reset this device locally (see _poll_buttons below).
_CONTROL_CALLBACK_KEYS = {"START", "STOP", "RESET", "R"}

# Button bit position matching Isaac Lab's legacy Se3SpaceMouse: the right button
# requests a reset (the left button's gripper toggle is handled by
# SpaceMouseGripperRetargeter inside the pipeline itself, so it needs no polling here).
_BUTTON_RIGHT = 1

# Tensor collection ID shared by the bundled spacemouse plugin and its SpaceMouseTracker;
# must match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_SPACEMOUSE_COLLECTION_ID = "spacemouse"


class Se3SpaceMouse(DeviceBase):
    """A space-mouse controller for sending SE(3) commands as delta poses.

    This class provides a space-mouse controller for a robotic arm with a gripper, built on the
    IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`). Raw axis/button state is
    read through a bundled ``spacemouse`` IsaacTeleop plugin (3Dconnexion HID) and retargeted via
    :class:`~isaacteleop.retargeters.SpaceMouseToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.SpaceMouseGripperRetargeter`.

    For a new environment config, prefer declaring ``self.isaac_teleop = IsaacTeleopCfg(
    pipeline_builder=...)`` directly (see the Franka / UR10 / Galbot / Agibot relative-mode
    stack/place configs for the pattern) rather than routing through this class -- this class
    exists for scripts and callers that want a single constructible, ``DeviceBase``-shaped
    object instead of an env-declared pipeline.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Note:
        The interface finds and uses the first supported device connected to the computer.

    Currently tested for following devices:

    - SpaceMouse Compact: https://3dconnexion.com/de/product/spacemouse-compact/

    Register callbacks for teleop control events via ``add_callback("START"/"STOP"/"RESET", func)``,
    the same way as any other IsaacTeleop device -- they fire when the underlying teleop session
    transitions state, regardless of what drives that transition (this device's own right button,
    an auxiliary keyboard, or the session's control channel).
    """

    def __init__(self, cfg: Se3SpaceMouseCfg):
        """Initialize the space-mouse layer.

        Args:
            cfg: Configuration object for space-mouse settings.
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
        """Returns: A string containing the information of joystick."""
        msg = f"Spacemouse Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tRight button: reset command\n"
        msg += "\tLeft button: toggle gripper command (open/close)\n"
        msg += "\tMove mouse laterally: move arm horizontally in x-y plane\n"
        msg += "\tMove mouse vertically: move arm vertically\n"
        msg += "\tTwist mouse about an axis: rotate arm about a corresponding axis"
        return msg

    """
    Operations
    """

    def reset(self):
        self._teleop_device.reset()

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind spacemouse.

        Args:
            key: The teleop control event to check against. ``"START"``, ``"STOP"``, ``"RESET"``,
                and ``"R"`` (aliased to ``"RESET"``) bind to this device's own teleop control
                events -- ``RESET``/``R`` also fires on the right physical button. Any other key is
                stored but never invoked.
            func: The function to call when the event fires. Should take no arguments.
        """
        if key in _CONTROL_CALLBACK_KEYS:
            self._teleop_device.add_callback(key, func)
        else:
            self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Provides the result from spacemouse event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        action = self._teleop_device.advance()
        if action is None:
            action = self._default_action()
        self._poll_buttons()
        return action

    """
    Internal helpers.
    """

    def _create_teleop_device(self):
        """Build the IsaacTeleop session driving this spacemouse's SE(3) pipeline."""
        from isaacteleop.plugins import plugin_search_path
        from isaacteleop.teleop_session_manager import PluginConfig

        from ..isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
        from ..isaac_teleop_device import create_isaac_teleop_device

        pos_sensitivity = self.pos_sensitivity
        rot_sensitivity = self.rot_sensitivity
        gripper_term = self.gripper_term

        def build_pipeline():
            from isaacteleop.retargeters import (
                SpaceMouseGripperRetargeter,
                SpaceMouseToSe3RelRetargeter,
                SpaceMouseToSe3RelRetargeterConfig,
                TensorReorderer,
            )
            from isaacteleop.retargeting_engine.deviceio_source_nodes import SpaceMouseSource
            from isaacteleop.retargeting_engine.interface import OutputCombiner

            spacemouse_source = SpaceMouseSource(_SPACEMOUSE_COLLECTION_ID)

            se3 = SpaceMouseToSe3RelRetargeter(
                SpaceMouseToSe3RelRetargeterConfig(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity),
                name="se3",
            )
            connected_se3 = se3.connect(
                {
                    "spacemouse_translation": spacemouse_source.output("spacemouse_translation"),
                    "spacemouse_rotation": spacemouse_source.output("spacemouse_rotation"),
                }
            )

            ee_delta_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
            input_config = {"ee_delta": ee_delta_elements}
            input_types = {"ee_delta": "array"}
            reorder_inputs = {"ee_delta": connected_se3.output("ee_delta")}
            output_order = list(ee_delta_elements)

            if gripper_term:
                gripper = SpaceMouseGripperRetargeter(name="gripper")
                connected_gripper = gripper.connect(
                    {"spacemouse_buttons": spacemouse_source.output("spacemouse_buttons")}
                )
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
                    "spacemouse_buttons": spacemouse_source.output("spacemouse_buttons"),
                }
            )

        teleop_cfg = IsaacTeleopCfg(
            pipeline_builder=build_pipeline,
            plugins=[
                PluginConfig(
                    plugin_name="spacemouse",
                    plugin_root_id=_SPACEMOUSE_COLLECTION_ID,
                    search_paths=[plugin_search_path()],
                )
            ],
            sim_device=self._sim_device,
            teleoperation_active_default=True,
            app_name="IsaacLabSpaceMouseSe3",
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

    def _poll_buttons(self) -> None:
        """Reset on the right button's rising edge, matching the legacy device-intrinsic binding."""
        bitmap = self._read_bitmap()
        if bitmap is None:
            return

        prev = self._prev_bitmap
        self._prev_bitmap = bitmap
        if prev is None or prev.shape != bitmap.shape:
            prev = np.zeros_like(bitmap)

        if bitmap[_BUTTON_RIGHT] and not prev[_BUTTON_RIGHT]:
            self._teleop_device.reset(pause=True)

    def _read_bitmap(self) -> np.ndarray | None:
        result = self._teleop_device.last_step_result
        if result is None:
            return None
        buttons = result.get("spacemouse_buttons")
        if buttons is None or buttons.is_none:
            return None
        return np.asarray(buttons[0])
