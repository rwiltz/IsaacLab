# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates an interactive demo with the H1 rough terrain environment.

.. code-block:: bash

    # Usage
    uv run python scripts/demos/h1_locomotion.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib import metadata

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # isort: skip


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx"],
    help="Physics backend.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Velocity-Rough-H1"
RL_LIBRARY = "rsl_rl"

# Evdev key codes (linux/input-event-codes.h) for the keys this demo reads.
_EVDEV_ARROW_KEY_CODES = {"UP": 103, "DOWN": 108, "LEFT": 105, "RIGHT": 106}
_EVDEV_ESCAPE_CODE = 1
_EVDEV_C_CODE = 46


def _create_keyboard_bitmap_device(sim_device: str):
    """Build an IsaacTeleop device whose ``action`` output is the raw 256-key evdev bitmap.

    Used for demos that need continuous held/released key state (not the edge-triggered
    start/stop/reset semantics of :class:`~isaaclab_teleop.Se2Keyboard` /
    :class:`~isaaclab_teleop.Se3Keyboard`).
    """
    from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
    from isaaclab_teleop.isaac_teleop_device import create_isaac_teleop_device
    from isaacteleop.plugins import plugin_search_path
    from isaacteleop.teleop_session_manager import PluginConfig

    def build_pipeline():
        from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardAllKeysType, KeyboardSource
        from isaacteleop.retargeting_engine.interface import BaseRetargeter, OutputCombiner
        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
        from isaacteleop.retargeting_engine.tensor_types import DLDataType, NDArrayType

        class _BitmapPassthrough(BaseRetargeter):
            """Converts the optional keyboard_all_keys bitmap into a definite one (zero-filled
            while the plugin has not streamed yet), so it can be used directly as ``action``."""

            def input_spec(self):
                return {"keyboard_all_keys": OptionalType(KeyboardAllKeysType())}

            def output_spec(self):
                return {
                    "bitmap": TensorGroupType(
                        "bitmap", [NDArrayType("bits", shape=(256,), dtype=DLDataType.UINT, dtype_bits=8)]
                    )
                }

            def _compute_fn(self, inputs, outputs, context):
                del context
                all_keys = inputs["keyboard_all_keys"]
                outputs["bitmap"][0] = np.zeros(256, dtype=np.uint8) if all_keys.is_none else np.asarray(all_keys[0])

        keyboard_source = KeyboardSource("keyboard")
        passthrough = _BitmapPassthrough(name="bitmap_passthrough")
        connected = passthrough.connect({"keyboard_all_keys": keyboard_source.output("keyboard_all_keys")})
        return OutputCombiner({"action": connected.output("bitmap")})

    teleop_cfg = IsaacTeleopCfg(
        pipeline_builder=build_pipeline,
        plugins=[PluginConfig(plugin_name="keyboard", plugin_root_id="keyboard", search_paths=[plugin_search_path()])],
        sim_device=sim_device,
        teleoperation_active_default=True,
        app_name="IsaacLabH1LocomotionDemo",
    )
    return create_isaac_teleop_device(teleop_cfg, cloudxr_env_file=CLOUDXR_STANDALONE_ENV, use_kit_xr_bridge=False)


def _read_keyboard_bitmap(keyboard_device) -> np.ndarray | None:
    """Advance the keyboard device and return its current 256-key bitmap, or ``None``."""
    action = keyboard_device.advance()
    if action is None:
        return None
    return np.asarray(action.cpu()) if hasattr(action, "cpu") else np.asarray(action)


class H1RoughDemo:
    """This class provides an interactive demo for the H1 rough terrain environment.
    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view"""

    def __init__(self):
        """Initializes environment config designed for the interactive model and sets up the environment,
        loads pre-trained checkpoints, and registers keyboard events."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        # create envionrment
        env_cfg, _ = resolve_task_config(TASK, "", play_mode=True, overrides=(f"physics={args_cli.physics}",))
        env_cfg.scene.num_envs = 25
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        # load the trained jit policy
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK, *backend_names)
        if checkpoint is None:
            raise FileNotFoundError("No published checkpoint is available for the H1 locomotion demo.")
        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        # load previously trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 4, device=self.device)
        self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self):
        """Creates a camera to be used for third-person view."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        """Builds the IsaacTeleop keyboard device and registers the desired keys for control."""
        self._keyboard_device = _create_keyboard_bitmap_device(self.device)
        self._keyboard_device.__enter__()
        self._prev_bitmap: np.ndarray | None = None

        T = 1
        R = 0.5
        self._key_to_control = {
            "UP": torch.tensor([T, 0.0, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, 0.0, -R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, 0.0, R], device=self.device),
        }
        self._zeros = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

    def poll_keyboard(self):
        """Reads the current keyboard state and updates the robot command / camera view.

        Movement keys (UP/DOWN/LEFT/RIGHT) and ESCAPE are level-triggered (checked every
        frame, matching the held/released semantics of the original carb key-press/release
        callbacks). C (camera toggle) fires once per press (rising edge) so holding it does
        not repeatedly flip the view.
        """
        bitmap = _read_keyboard_bitmap(self._keyboard_device)
        if bitmap is None:
            return
        prev = self._prev_bitmap
        self._prev_bitmap = bitmap
        if prev is None or prev.shape != bitmap.shape:
            prev = np.zeros_like(bitmap)

        if self._selected_id is not None:
            command = self._zeros
            for name, code in _EVDEV_ARROW_KEY_CODES.items():
                if bitmap[code]:
                    command = self._key_to_control[name]
                    break
            self.commands[self._selected_id] = command

        if bitmap[_EVDEV_ESCAPE_CODE]:
            self._prim_selection.clear_selected_prim_paths()

        c_rising = bitmap[_EVDEV_C_CODE] and not prev[_EVDEV_C_CODE]
        if c_rising and self._selected_id is not None:
            if self.viewport.get_active_camera() == self.camera_path:
                self.viewport.set_active_camera(self.perspective_path)
            else:
                self.viewport.set_active_camera(self.camera_path)

    def update_selected_object(self):
        """Determines which robot is currently selected and whether it is a valid H1 robot.
        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # a valid robot was selected, update the camera to go into third-person view
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")

        # Reset commands for previously selected robot if a new one is selected
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, 0:3] = self.env.unwrapped.command_manager.get_command("base_velocity")

    def _update_camera(self):
        """Updates the per-frame transform of the third-person view camera to follow
        the selected robot's torso transform."""

        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w.torch[
            self._selected_id, :
        ]  # - env.scene.env_origins
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w.torch[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main():
    """Main function."""
    demo_h1 = H1RoughDemo()
    obs, _ = demo_h1.env.reset()
    while simulation_app.is_running():
        # check for selected robots
        demo_h1.update_selected_object()
        demo_h1.poll_keyboard()
        with torch.inference_mode():
            action = demo_h1.policy(obs)
            obs, _, _, _ = demo_h1.env.step(action)
            # overwrite command based on keyboard input
            obs[:, 9:13] = demo_h1.commands


if __name__ == "__main__":
    main()
    simulation_app.close()
