# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.keyboard import Se3KeyboardCfg
from isaaclab_teleop.spacemouse import Se3SpaceMouseCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from . import stack_joint_pos_env_cfg


def _build_ur10_stack_keyboard_pipeline():
    """Build an IsaacTeleop retargeting pipeline for keyboard-driven SE(3) delta-pose control.

    Creates a KeyboardSource with a KeyboardToSe3RelRetargeter (delta pose) and a
    KeyboardGripperRetargeter (suction toggle), flattened into a single action tensor via
    TensorReorderer.

    Returns:
        OutputCombiner with a single "action" output containing the flattened
        7D action tensor: [dx, dy, dz, drx, dry, drz, gripper].
    """
    from isaacteleop.retargeters import (
        KeyboardGripperRetargeter,
        KeyboardToSe3RelRetargeter,
        KeyboardToSe3RelRetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    keyboard_source = KeyboardSource(name="keyboard")

    se3_cfg = KeyboardToSe3RelRetargeterConfig(pos_sensitivity=0.02, rot_sensitivity=0.05)
    se3 = KeyboardToSe3RelRetargeter(se3_cfg, name="se3")
    connected_se3 = se3.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})

    gripper = KeyboardGripperRetargeter(name="gripper")
    connected_gripper = gripper.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})

    ee_delta_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
    reorderer = TensorReorderer(
        input_config={"ee_delta": ee_delta_elements, "gripper_command": ["gripper_value"]},
        output_order=ee_delta_elements + ["gripper_value"],
        name="action_reorderer",
        input_types={"ee_delta": "array", "gripper_command": "scalar"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_delta": connected_se3.output("ee_delta"),
            "gripper_command": connected_gripper.output("gripper_command"),
        }
    )

    return OutputCombiner({"action": connected_reorderer.output("output")})


def _ur10_keyboard_plugin_config():
    """IsaacTeleop plugin configuration for the bundled ``keyboard`` evdev plugin."""
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    return PluginConfig(plugin_name="keyboard", plugin_root_id="keyboard", search_paths=[plugin_search_path()])


@configclass
class UR10LongSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR10LongSuctionCubeStackEnvCfg):
    """Configuration for the UR10 Long Suction Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR10 LONG SUCTION)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.22]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # IsaacTeleop-based keyboard teleoperation pipeline
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_ur10_stack_keyboard_pipeline,
            plugins=[_ur10_keyboard_plugin_config()],
            sim_device=self.sim.device,
        )


@configclass
class UR10ShortSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR10ShortSuctionCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR10 SHORT SUCTION)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*_joint"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # IsaacTeleop-based keyboard teleoperation pipeline
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_ur10_stack_keyboard_pipeline,
            plugins=[_ur10_keyboard_plugin_config()],
            sim_device=self.sim.device,
        )
