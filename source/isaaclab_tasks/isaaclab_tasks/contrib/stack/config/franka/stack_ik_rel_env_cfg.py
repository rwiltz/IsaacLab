# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.keyboard import Se3KeyboardCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack.stack_env_cfg import mdp

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import (  # isort: skip
    FRANKA_PANDA_HIGH_PD_CFG,
)


def _build_franka_stack_keyboard_pipeline():
    """Build an IsaacTeleop retargeting pipeline for keyboard-driven SE(3) delta-pose control.

    Creates a KeyboardSource with a KeyboardToSe3RelRetargeter (delta pose) and a
    KeyboardGripperRetargeter (gripper toggle), flattened into a single action tensor via
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

    se3_cfg = KeyboardToSe3RelRetargeterConfig(pos_sensitivity=0.05, rot_sensitivity=0.05)
    se3 = KeyboardToSe3RelRetargeter(se3_cfg, name="se3")
    connected_se3 = se3.connect({"keyboard": keyboard_source.output("keyboard")})

    gripper = KeyboardGripperRetargeter(name="gripper")
    connected_gripper = gripper.connect({"keyboard": keyboard_source.output("keyboard")})

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


def _franka_keyboard_plugin_config():
    """IsaacTeleop plugin configuration for the bundled ``keyboard`` evdev plugin."""
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    return PluginConfig(plugin_name="keyboard", plugin_root_id="keyboard", search_paths=[plugin_search_path()])


@configclass
class FrankaCubeStackEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # Use a stiffer PD controller for better IK tracking.
        robot_init_state = self.scene.robot.init_state
        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot", init_state=robot_init_state
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
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
            pipeline_builder=_build_franka_stack_keyboard_pipeline,
            plugins=[_franka_keyboard_plugin_config()],
            sim_device=self.sim.device,
        )


@configclass
class FrankaCubeStackRedGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_2"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": None,
            },
        )


@configclass
class FrankaCubeStackRedGreenBlueEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_2"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_1"),
            },
        )


@configclass
class FrankaCubeStackBlueGreenEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": None,
            },
        )


@configclass
class FrankaCubeStackBlueGreenRedEnvCfg(FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terminations.success = DoneTerm(
            func=mdp.cubes_stacked,
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_3"),
                "cube_3_cfg": SceneEntityCfg("cube_2"),
            },
        )
