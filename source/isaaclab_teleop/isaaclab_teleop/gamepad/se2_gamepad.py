# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad-driven IsaacTeleopCfg builder for SE(2) control."""

from __future__ import annotations

# Tensor collection ID shared by the bundled gamepad plugin and its GamepadTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_GAMEPAD_COLLECTION_ID = "gamepad"


def se2_gamepad_teleop_cfg(
    v_x_sensitivity: float = 1.0,
    v_y_sensitivity: float = 1.0,
    omega_z_sensitivity: float = 1.0,
    dead_zone: float = 0.01,
    sim_device: str = "cpu",
):
    r"""Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for gamepad-driven SE(2) velocity control.

    Raw stick state is read through a bundled ``gamepad`` IsaacTeleop plugin (Linux joystick API)
    and retargeted via :class:`~isaacteleop.retargeters.GamepadToSe2Retargeter`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      left stick up             left stick down
        Move along y-axis      left stick right          left stick left
        Rotate along z-axis    right stick right         right stick left
        ====================== ========================= ========================

    No physical button drives session start/stop/reset -- those come from the teleop session's
    control channel or an auxiliary control device, matching the legacy ``Se2Gamepad``.

    Args:
        v_x_sensitivity: Linear x-velocity scale per step.
        v_y_sensitivity: Linear y-velocity scale per step.
        omega_z_sensitivity: Angular z-velocity scale per step.
        dead_zone: Stick deflection magnitude below which input is treated as zero.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    from ..isaac_teleop_cfg import IsaacTeleopCfg

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

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        plugins=[
            PluginConfig(
                plugin_name="gamepad",
                plugin_root_id=_GAMEPAD_COLLECTION_ID,
                search_paths=[plugin_search_path()],
            )
        ],
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabGamepadSe2",
    )
