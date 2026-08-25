# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard-driven IsaacTeleopCfg builder for SE(3) control."""

from __future__ import annotations

# Tensor collection ID shared by the bundled keyboard plugin and its KeyboardTracker; must
# match on both sides of the wire for the SchemaPusher/SchemaTracker pairing to succeed.
_KEYBOARD_COLLECTION_ID = "keyboard"


def se3_keyboard_teleop_cfg(
    pos_sensitivity: float = 0.4,
    rot_sensitivity: float = 0.8,
    gripper_term: bool = True,
    sim_device: str = "cpu",
):
    """Build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard-driven SE(3) delta-pose control.

    Raw key state is read through a bundled ``keyboard`` IsaacTeleop plugin (Linux evdev) and
    retargeted via :class:`~isaacteleop.retargeters.KeyboardToSe3RelRetargeter` /
    :class:`~isaacteleop.retargeters.KeyboardGripperRetargeter`.

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

    The resulting pipeline's ``OutputCombiner`` also exposes a ``"keyboard_all_keys"`` output --
    poll it with :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller` for physical
    B (start/resume) / P (pause) / R (reset) control-key handling.

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

    return IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        plugins=[
            PluginConfig(
                plugin_name="keyboard",
                plugin_root_id=_KEYBOARD_COLLECTION_ID,
                search_paths=[plugin_search_path()],
            )
        ],
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabKeyboardSe3",
    )
