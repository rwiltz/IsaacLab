# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse-driven IsaacTeleopCfg builder for SE(3) control."""

from __future__ import annotations

# Tensor collection ID shared by the bundled spacemouse plugin and its SpaceMouseTracker;
# must match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_SPACEMOUSE_COLLECTION_ID = "spacemouse"


def se3_spacemouse_teleop_cfg(
    pos_sensitivity: float = 0.4,
    rot_sensitivity: float = 0.8,
    gripper_term: bool = True,
    sim_device: str = "cpu",
):
    """Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for spacemouse-driven SE(3) delta-pose control.

    Raw axis/button state is read through a bundled ``spacemouse`` IsaacTeleop plugin
    (3Dconnexion HID) and retargeted via
    :class:`~isaacteleop.retargeters.SpaceMouseToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.SpaceMouseGripperRetargeter`.

    The interface finds and uses the first supported device connected to the computer.
    Currently tested for the SpaceMouse Compact
    (https://3dconnexion.com/de/product/spacemouse-compact/).

    The left button's gripper toggle is handled inside the pipeline by
    ``SpaceMouseGripperRetargeter``. The resulting pipeline's ``OutputCombiner`` also exposes a
    ``"spacemouse_buttons"`` output -- poll it with
    :class:`~isaaclab_teleop.control_pollers.SpaceMouseResetPoller` for the right button's
    device-intrinsic reset, matching the legacy ``Se3SpaceMouse``.

    Args:
        pos_sensitivity: Position delta scale per step [m].
        rot_sensitivity: Rotation delta scale per step [rad].
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
            connected_gripper = gripper.connect({"spacemouse_buttons": spacemouse_source.output("spacemouse_buttons")})
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

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        plugins=[
            PluginConfig(
                plugin_name="spacemouse",
                plugin_root_id=_SPACEMOUSE_COLLECTION_ID,
                search_paths=[plugin_search_path()],
            )
        ],
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabSpaceMouseSe3",
    )
