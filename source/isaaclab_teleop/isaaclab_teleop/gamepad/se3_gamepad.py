# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad-driven IsaacTeleopCfg builder for SE(3) control."""

from __future__ import annotations

# Tensor collection ID shared by the bundled gamepad plugin and its GamepadTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_GAMEPAD_COLLECTION_ID = "gamepad"


def se3_gamepad_teleop_cfg(
    pos_sensitivity: float = 1.0,
    rot_sensitivity: float = 1.6,
    dead_zone: float = 0.01,
    gripper_term: bool = True,
    sim_device: str = "cpu",
):
    """Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for gamepad-driven SE(3) delta-pose control.

    Raw stick/button state is read through a bundled ``gamepad`` IsaacTeleop plugin (Linux
    joystick API) and retargeted via :class:`~isaacteleop.retargeters.GamepadToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.GamepadGripperRetargeter`.

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

    No physical button drives session start/stop/reset -- those come from the teleop session's
    control channel or an auxiliary control device, matching the legacy ``Se3Gamepad``.

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
        dead_zone: Stick deflection magnitude below which input is treated as zero.
        gripper_term: Whether to include a gripper open/close command as a 7th action element.
        sim_device: Torch device string for the pipeline's output tensors.

    Returns:
        IsaacTeleopCfg driving this pipeline.
    """
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    from ..isaac_teleop_cfg import IsaacTeleopCfg

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
        app_name="IsaacLabGamepadSe3",
    )
