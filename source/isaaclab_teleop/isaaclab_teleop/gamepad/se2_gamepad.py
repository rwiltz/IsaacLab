# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad controller for SE(2) control."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.devices.device_base import DeviceBase

if TYPE_CHECKING:
    from .se2_gamepad_cfg import Se2GamepadCfg

# START/STOP/RESET are driven by this device's teleop control state machine (e.g. an
# auxiliary control device or the session's control channel), not a physical gamepad
# button -- matching the legacy Se2Gamepad, which never wired any button to them either.
_CONTROL_CALLBACK_KEYS = {"START", "STOP", "RESET", "R"}

# Tensor collection ID shared by the bundled gamepad plugin and its GamepadTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_GAMEPAD_COLLECTION_ID = "gamepad"


class Se2Gamepad(DeviceBase):
    r"""A gamepad controller for sending SE(2) commands as velocity commands.

    This class provides a gamepad controller for a mobile base (such as quadrupeds), built on
    the IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`). Raw stick state is
    read through a bundled ``gamepad`` IsaacTeleop plugin (Linux joystick API) and retargeted via
    :class:`~isaacteleop.retargeters.GamepadToSe2Retargeter`.

    For a new environment config, prefer declaring ``self.isaac_teleop = IsaacTeleopCfg(
    pipeline_builder=...)`` directly rather than routing through this class -- this class
    exists for scripts and callers that want a single constructible, ``DeviceBase``-shaped
    object instead of an env-declared pipeline.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      left stick up             left stick down
        Move along y-axis      left stick right          left stick left
        Rotate along z-axis    right stick right         right stick left
        ====================== ========================= ========================

    Register callbacks for teleop control events via ``add_callback("START"/"STOP"/"RESET", func)``,
    the same way as any other IsaacTeleop device -- they fire when the underlying teleop session
    transitions state, regardless of what drives that transition (e.g. an auxiliary keyboard, or
    the session's control channel).
    """

    def __init__(self, cfg: Se2GamepadCfg):
        """Initialize the gamepad layer.

        Args:
            cfg: Configuration object for gamepad settings.
        """
        self.v_x_sensitivity = cfg.v_x_sensitivity
        self.v_y_sensitivity = cfg.v_y_sensitivity
        self.omega_z_sensitivity = cfg.omega_z_sensitivity
        self.dead_zone = cfg.dead_zone
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
        msg = f"Gamepad Controller for SE(2): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove in X-Y plane: left stick\n"
        msg += "\tRotate in Z-axis: right stick\n"
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
            Tensor containing the linear (x,y) and angular velocity (z).
        """
        action = self._teleop_device.advance()
        if action is None:
            action = torch.zeros(3, dtype=torch.float32, device=self._sim_device)
        return action

    """
    Internal helpers.
    """

    def _create_teleop_device(self):
        """Build the IsaacTeleop session driving this gamepad's SE(2) pipeline."""
        from isaacteleop.plugins import plugin_search_path
        from isaacteleop.teleop_session_manager import PluginConfig

        from ..isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
        from ..isaac_teleop_device import create_isaac_teleop_device

        v_x_sensitivity = self.v_x_sensitivity
        v_y_sensitivity = self.v_y_sensitivity
        omega_z_sensitivity = self.omega_z_sensitivity
        dead_zone = self.dead_zone

        def build_pipeline():
            from isaacteleop.retargeters import GamepadToSe2Retargeter, GamepadToSe2RetargeterConfig, TensorReorderer
            from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource
            from isaacteleop.retargeting_engine.interface import OutputCombiner

            gamepad_source = GamepadSource(_GAMEPAD_COLLECTION_ID)

            se2 = GamepadToSe2Retargeter(
                GamepadToSe2RetargeterConfig(
                    v_x_sensitivity=v_x_sensitivity,
                    v_y_sensitivity=v_y_sensitivity,
                    omega_z_sensitivity=omega_z_sensitivity,
                    dead_zone=dead_zone,
                ),
                name="se2",
            )
            connected_se2 = se2.connect({"gamepad_axes": gamepad_source.output("gamepad_axes")})

            base_command_elements = ["v_x", "v_y", "omega_z"]
            reorderer = TensorReorderer(
                input_config={"base_command": base_command_elements},
                output_order=base_command_elements,
                name="action_reorderer",
                input_types={"base_command": "array"},
            )
            connected_reorderer = reorderer.connect({"base_command": connected_se2.output("base_command")})

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
            app_name="IsaacLabGamepadSe2",
        )
        return create_isaac_teleop_device(
            teleop_cfg,
            cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
            use_kit_xr_bridge=False,
        )
