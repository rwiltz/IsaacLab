# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-thread adapters for the environment-independent Astra test runner."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .cameras import capture_rgb_images
from .core import ActionContract

if TYPE_CHECKING:
    import argparse

    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class EnvironmentAdapter(ABC):
    """Define environment semantics without exposing simulator objects to inference workers.

    A factory is called with the unwrapped, already-reset environment. All adapter
    methods run on the simulation thread. Snapshots, descriptions, and encoded
    images must be detached JSON values; only the action methods return tensors.
    The action contract's optional validator must be pure and thread-safe.
    """

    env: ManagerBasedRLEnv

    @classmethod
    def configure_env(cls, env_cfg: ManagerBasedRLEnvCfg, args: argparse.Namespace) -> None:
        """Prepare configuration before simulator launch; preserve native task behavior."""
        args.enable_cameras = bool(args.cameras)

    @property
    @abstractmethod
    def contract(self) -> ActionContract:
        """Structured command schema and task-specific interpretation for inference."""

    @abstractmethod
    def describe(self) -> dict:
        """Return copied action, observation, and task semantics for model context."""

    def reset(self) -> None:
        """Discard adapter state after an environment reset."""

    @abstractmethod
    def snapshot(self, observations: dict, episode_id: int, step: int, sim_time: float) -> dict:
        """Copy current measurements and a schema-valid ``neutral_command``."""

    def capture_images(self, cameras: list[str], image_size: int) -> list[dict]:
        """Copy named scene RGB sensors to JPEG data URLs; an empty selection needs no renderer."""
        return capture_rgb_images(self.env, cameras, image_size)

    @abstractmethod
    def action(self, command: dict) -> torch.Tensor:
        """Validate and convert one semantic command to a finite, bounded env action."""

    @abstractmethod
    def idle_action(self) -> torch.Tensor:
        """Return an environment-appropriate fallback while inference is unavailable."""

    def task_progress(self, dt: float) -> dict | None:
        """Optionally report measured completion with a boolean ``success`` field.

        ``dt`` is executed simulation time, and is zero immediately after reset.
        Returning ``None`` leaves episode outcomes to native termination terms.
        """
        return None


class CartpoleAdapter(EnvironmentAdapter):
    """A camera-free, one-force example for the manager-based ``Isaac-Cartpole``."""

    @classmethod
    def configure_env(cls, env_cfg: ManagerBasedRLEnvCfg, args: argparse.Namespace) -> None:
        super().configure_env(env_cfg, args)
        for name in args.cameras:
            if getattr(env_cfg.scene, name, None) is None:
                raise ValueError(f"Cartpole camera {name!r} must be configured in the scene before launch.")

    def __init__(self, env: ManagerBasedRLEnv):
        from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg

        self.env = env
        manager = env.action_manager
        if env.num_envs != 1 or manager.active_terms != ["joint_effort"] or manager.action_term_dim != [1]:
            raise ValueError("CartpoleAdapter requires one environment with the single 1D joint_effort action.")
        cfg = manager.get_term("joint_effort").cfg
        if not isinstance(cfg, JointEffortActionCfg) or cfg.asset_name != "robot":
            raise ValueError("CartpoleAdapter requires a JointEffortActionCfg on the robot asset.")
        if (
            isinstance(cfg.scale, bool)
            or not isinstance(cfg.scale, (int, float))
            or not math.isfinite(cfg.scale)
            or cfg.scale <= 0
            or cfg.offset != 0.0
            or cfg.clip is not None
        ):
            raise ValueError(
                "Cartpole force mapping requires a positive scalar scale, zero offset, and no term clipping."
            )
        self.robot = env.scene["robot"]
        self._joint_names = list(self.robot.data.joint_names)
        if len(self._joint_names) != 2 or set(self._joint_names) != {"slider_to_cart", "cart_to_pole"}:
            raise ValueError("CartpoleAdapter requires the cart slider and pole hinge joints.")
        if self.robot.find_joints(cfg.joint_names)[1] != ["slider_to_cart"]:
            raise ValueError("The force action must address only slider_to_cart.")
        if env.observation_manager.active_terms.get("policy") != ["joint_pos_rel", "joint_vel_rel"]:
            raise ValueError("Cartpole policy observations must be joint_pos_rel followed by joint_vel_rel.")
        self._cart_id = self._joint_names.index("slider_to_cart")
        self._pole_id = self._joint_names.index("cart_to_pole")
        self._force_scale = float(cfg.scale)
        self._contract = ActionContract(
            command_schema={
                "type": "object",
                "properties": {"cart_force": {"type": "number", "minimum": -1.0, "maximum": 1.0}},
                "required": ["cart_force"],
                "additionalProperties": False,
            },
            instructions=(
                "Control the cart with cart_force in [-1,1], a normalized effort command. "
                "The action specification gives the Newtons-per-unit scale. Zero applies no commanded force. "
                "Use measured cart position, pole angle, and their velocities; pole angle zero is upright. "
                "Each command holds a force, not a desired cart position or velocity. Native episode termination "
                "and timeout remain enabled. Camera inputs are optional; commands contain no wrist or gripper fields."
            ),
        )

    @property
    def contract(self) -> ActionContract:
        return self._contract

    def describe(self) -> dict:
        return {
            "action_dimension": 1,
            "action_term": "joint_effort",
            "joint": "slider_to_cart",
            "command": {"cart_force": {"minimum": -1.0, "maximum": 1.0, "unit": "normalized force"}},
            "force_newtons_per_unit": self._force_scale,
            "positive_force_direction": "positive slider_to_cart joint coordinate",
            "idle_command": {"cart_force": 0.0},
            "state_source": "robot articulation joint positions and velocities",
            "policy_layout": [
                *[f"{name}.position_relative_to_default" for name in self._joint_names],
                *[f"{name}.velocity_relative_to_default" for name in self._joint_names],
            ],
            "native_episode_length_s": float(self.env.cfg.episode_length_s),
            "completion": "Native environment terminations and timeout; no adapter-defined success criterion.",
        }

    def snapshot(self, observations: dict, episode_id: int, step: int, sim_time: float) -> dict:
        policy = observations["policy"]
        if policy.shape != (1, 4) or not torch.isfinite(policy).all():
            raise ValueError("Cartpole policy observations must contain four finite values for one environment.")
        positions = self.robot.data.joint_pos.torch[0]
        velocities = self.robot.data.joint_vel.torch[0]
        state = torch.stack(
            (positions[self._cart_id], positions[self._pole_id], velocities[self._cart_id], velocities[self._pole_id])
        )
        if not torch.isfinite(state).all():
            raise ValueError("Cartpole measured state must be finite.")
        names = ("cart_position_m", "pole_angle_rad", "cart_velocity_m_s", "pole_velocity_rad_s")
        return {
            "episode_id": episode_id,
            "step": step,
            "sim_time": sim_time,
            "state": dict(zip(names, state.detach().cpu().tolist(), strict=True)),
            "policy": policy.detach().cpu()[0].tolist(),
            "neutral_command": {"cart_force": 0.0},
        }

    def action(self, command: dict) -> torch.Tensor:
        if not isinstance(command, dict) or set(command) != {"cart_force"}:
            raise ValueError("Cartpole commands must contain only cart_force.")
        force = command["cart_force"]
        if (
            isinstance(force, bool)
            or not isinstance(force, (int, float))
            or not math.isfinite(force)
            or not -1.0 <= force <= 1.0
        ):
            raise ValueError("cart_force must be a finite number in [-1,1].")
        return torch.tensor([[force]], dtype=torch.float32, device=self.env.device)

    def idle_action(self) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=torch.float32, device=self.env.device)
