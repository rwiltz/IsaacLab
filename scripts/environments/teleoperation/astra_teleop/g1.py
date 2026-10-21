# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Translate bounded fixed-base G1 commands into the existing Pink IK action term."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .cameras import capture_rgb_images

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .perception import ObjectPerception


_HAND_SUFFIXES = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
_WRIST_LINKS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# 90% of the TriHand controller's full trigger+squeeze curl, with the mirrored
# joint signs in the G1 kinematics asset; thumb opposition remains centered.
_CLOSED_LEFT_HAND = (0.0, 0.36, 0.63, -0.9, -0.9, -0.9, -0.9)
_CLOSED_RIGHT_HAND = (0.0, -0.36, -0.63, 0.9, 0.9, 0.9, 0.9)


class G1ActionAdapter:
    """Own the single-environment, fixed-base G1 action contract and simulator snapshots.

    All methods must run on the simulation thread. Only their copied JSON values
    and encoded images may be handed to inference workers.
    """

    def __init__(self, env: ManagerBasedRLEnv, perception: ObjectPerception | None = None, *, grip_speed: float = 1.6):
        from isaaclab.controllers.pink_ik import LocalFrameTaskCfg
        from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

        if isinstance(grip_speed, bool) or not math.isfinite(grip_speed) or grip_speed <= 0:
            raise ValueError("Grip speed must be positive and finite.")
        self.grip_speed = grip_speed
        self.env = env
        self.perception = perception
        manager = env.action_manager
        if env.num_envs != 1 or manager.active_terms != ["upper_body_ik"] or manager.action_term_dim != [28]:
            raise ValueError("Astra G1 requires one fixed-base environment with only the 28D upper_body_ik action.")
        upper = manager.get_term("upper_body_ik")
        if (
            not isinstance(upper.cfg, PinkInverseKinematicsActionCfg)
            or upper.cfg.asset_name != "robot"
            or upper.cfg.controller.base_link_name != "pelvis"
            or list(upper.cfg.target_eef_link_names.values()) != list(_WRIST_LINKS)
        ):
            raise ValueError("Unsupported G1 action term: expected the existing pelvis-based Pink IK controller.")
        tasks = upper.cfg.controller.variable_input_tasks[:2]
        if len(tasks) != 2 or any(
            not isinstance(task, LocalFrameTaskCfg)
            or not task.frame.endswith(link)
            or not task.base_link_frame_name.endswith("pelvis")
            for task, link in zip(tasks, _WRIST_LINKS, strict=True)
        ):
            raise ValueError("Pink IK must consume left then right wrist targets relative to the pelvis.")
        self.robot = env.scene["robot"]
        if not self.robot.is_fixed_base:
            raise ValueError("Astra G1 requires a fixed-base robot; configure scene.robot.spawn.fix_root_link=True.")
        self._wrist_ids = [self.robot.data.body_names.index(name) for name in _WRIST_LINKS]
        self._hand_link_ids = [
            self.robot.data.body_names.index(f"{side}_hand_{suffix}_link")
            for side in ("left", "right")
            for suffix in _HAND_SUFFIXES
        ]
        # Pink resolves these in articulation order, which can differ from the config's order.
        self._hand_ids, self._hand_names = self.robot.find_joints(upper.cfg.hand_joint_names)
        self._semantic_hand_names = [
            f"{side}_hand_{suffix}_joint" for side in ("left", "right") for suffix in _HAND_SUFFIXES
        ]
        if len(self._hand_names) != 14 or set(self._hand_names) != set(self._semantic_hand_names):
            raise ValueError("G1 Astra control requires the fourteen configured TriHand joints.")
        self._to_action_order = [self._semantic_hand_names.index(name) for name in self._hand_names]
        self._closed_hand_target = torch.tensor(_CLOSED_LEFT_HAND + _CLOSED_RIGHT_HAND, device=env.device)[
            self._to_action_order
        ].unsqueeze(0)
        self._left_hand_mask = torch.tensor(
            [[name.startswith("left_") for name in self._hand_names]], device=env.device
        )
        self.reset()

    def reset(self) -> None:
        """Clear trajectories and temporal comparisons at an environment episode boundary."""
        self._left_grip_closed = False
        self._right_grip_closed = False
        self._grip_position: torch.Tensor | None = None
        self._grip_target: torch.Tensor | None = None
        self._episode_object_pose = self._object_pose()
        self._motion_episode_id: int | None = None
        self._previous_motion: tuple | None = None

    def _grip_targets(self, left_closed: bool, right_closed: bool) -> torch.Tensor:
        target = torch.where(
            self._left_hand_mask,
            self._closed_hand_target * left_closed,
            self._closed_hand_target * right_closed,
        )
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self._hand_ids]
        return torch.clamp(target, min=limits[..., 0], max=limits[..., 1])

    def grippers_open(self) -> list[bool]:
        """Require both an open command and measured opening, allowing 0.1 rad joint error."""
        errors = (self.robot.data.joint_pos.torch[:, self._hand_ids] - self._grip_targets(False, False)).abs()
        return [
            not self._left_grip_closed and bool((errors[self._left_hand_mask] <= 0.1).all()),
            not self._right_grip_closed and bool((errors[~self._left_hand_mask] <= 0.1).all()),
        ]

    def describe(self) -> dict:
        """Describe coordinates, units, ordering, and enforced command limits."""
        return {
            "action_dim": 28,
            "position_frame": "environment origin with world axes, in meters; NOT pelvis-relative",
            "quaternion_convention": "[x,y,z,w], absolute orientation in world axes",
            "wrist_frame_guide": (
                "The commanded pose is the wrist_yaw_link origin, not the palm or a fingertip. "
                "In the G1 kinematics, wrist +X points toward the index/middle roots; their separation is along Z. "
                "Left index/middle fingers curl toward wrist -Y and right index/middle toward +Y. "
                "With the current open preset, the right thumb extends roughly along wrist +Y and the left "
                "along wrist -Y. Use hand_geometry.<side>.wrist_axes_environment to compare these directions "
                "with gravity (environment -Z in this task): gravity can pull an object toward the thumb base "
                "or toward its tip. An open hand can still support a hooked object; verify release with "
                "measured contacts and object motion. "
                "Use measured hand_geometry to account for these offsets when positioning or rotating the hand. "
                "A wrist rotation also moves the fingers around that wrist origin."
            ),
            "nominal_pinch_centers": {
                "frame": "wrist_yaw_link local coordinates [m], for the current 90%-closed finger preset",
                "left": {
                    "position_wrist_m": [0.12125379, -0.08139178, -0.00274833],
                    "source": "approximate capture center from a simulator-validated left-hand rim grasp and slow lift",
                },
                "right": {
                    "position_wrist_m": [0.12125379, 0.08139178, -0.00274833],
                    "source": (
                        "mirrored left-hand center, simulator-validated with a right-hand rim grasp, "
                        "slow lift, and two-second fixed hold"
                    ),
                },
                "usage": (
                    "These are robot-frame tool offsets, not scene grasp targets or measured contacts. "
                    "For wrist rotation R, center = wrist_position + R * position_wrist_m. Align the nominal "
                    "center with a selected object surface while allowing clearance, then verify actual capture "
                    "with cameras and contact support. Calibration covers this simulated hand and finger preset."
                ),
            },
            "observation_guide": {
                "hand_geometry": (
                    "Measured robot link frames: wrist_axes_environment gives local X/Y/Z directions in world axes. "
                    "links contains each finger link's origin in environment and wrist coordinates [m], "
                    "and absolute XYZW orientation. These origins are not fingertip surfaces or contact pads. "
                    "object_origin_wrist_m locates the object's root in the current wrist frame."
                ),
                "object": (
                    "observations.object_pos/object_rot are the object's root pose, not a rim grasp point. "
                    "observations.object packs [position(3), XYZW(4), object-minus-left-wrist(3), "
                    "object-minus-right-wrist(3)] in environment axes, meters."
                ),
                "contacts": (
                    "hand_contact_forces_world_n is net contact force [N] per hand_contact_body_names entry, "
                    "aggregated over all contacted bodies, not object-specific. Isaac Sim 6.2 includes normal "
                    "and friction forces. Nonzero force or a closed grip request alone does not establish a grasp."
                ),
                "object_contacts": (
                    "When present, object_contact_forces_world_n labels total normal+friction forces [N] ON the "
                    "object from each finger body, excluding table and other contacts. Opposing finger forces "
                    "can cancel, so inspect individual vectors. These measured contacts do not prove a secure grasp; "
                    "verify object motion and support. Aggregate hand forces may instead come from the table. "
                    "object_total_contact_force_world_n includes all object collisions; "
                    "object_nonfinger_contact_force_world_n is the residual after subtracting finger vectors "
                    "and can include palms, table, or bin contacts. "
                    "Residual forces can cancel. object_contact_positions_environment_m gives each finger pair's "
                    "unweighted mean normal-contact position [m], or null when unavailable."
                ),
                "object_geometry": (
                    "When present, object_geometry contains static CAD mesh bounds in the object's rigid-body "
                    "frame [m], with asset scale already included. Transform by measured object_pos/object_rot; "
                    "do not treat the root position as the rim. Authored mesh names label the parts. An outer "
                    "radial band, when available, describes source vertices, not a wrist or grasp target. "
                    "CAD geometry is not an observation from camera pixels or a guarantee of collision geometry."
                ),
                "object_motion": (
                    "Measured episode_initial_pose is captured at reset. height_change_from_episode_start_m "
                    "compares current object root Z with that baseline. since_previous_snapshot reports elapsed "
                    "simulation time and object/wrist displacement since the last inference snapshot. Object-relative "
                    "drift is measured in each wrist's local frame; small translation AND rotation drift while "
                    "object and wrist move together supports a stable grasp. Current root-link linear [m/s] and "
                    "angular [rad/s] velocities, when available, use world axes. These are sampled endpoints, "
                    "not proof that the object stayed stable throughout the intervening interval. A captured rim "
                    "can pivot under gravity, so root-pose drift alone does not establish grasp loss. Assess the "
                    "retained contact region, contact support, and fresh views. Allow settling before transfer. "
                    "contact_centroid_translation_wrist_m compares each finger's unweighted mean contact position "
                    "in its associated wrist frame, only where both samples have finite contact. These are sampled "
                    "contact-region centroids, not tracked material points or continuous contact history."
                ),
                "cameras": (
                    "Measured camera position is environment-relative [m]; quaternion_ros_xyzw has camera "
                    "+Z forward, +X image-right, +Y image-down. Intrinsics are in native-image pixels; "
                    "scale their first and second rows by delivered/native image width and height for resized JPEGs."
                ),
            },
            "hand_order": list(_HAND_SUFFIXES),
            "hand_units": "radians",
            "hand_joint_names_in_action": self._hand_names,
            "hand_limits_in_action_order": self.robot.data.soft_joint_pos_limits.torch[0, self._hand_ids]
            .cpu()
            .tolist(),
            "grip_control": "left_grip_closed/right_grip_closed: true selects close, false selects open",
            "grip_targets_in_action_order_rad": {
                "open": self._grip_targets(False, False)[0].cpu().tolist(),
                "closed": self._grip_targets(True, True)[0].cpu().tolist(),
            },
            "grip_trajectory": (
                f"Each control tick ramps joints toward soft-limit-clipped endpoints at {self.grip_speed:g} rad/s. "
                "The path obeys physical joint limits. Inference waits retain accepted finger targets. "
                "Explicit stop, plan expiry, or single-command expiry holds measured joints. "
                "Neutral grip booleans retain the previous requested states, initially open."
            ),
            "limits": {
                "wrist_translation_per_command_m": 0.05,
                "wrist_rotation_per_command_rad": 0.2,
                "hand_joint_target_speed_rad_s": self.grip_speed,
            },
            "low_level_control": "Existing Pink IK controls the wrists and hands inside env.step; the base is fixed.",
        }

    def _current_command(self) -> dict:
        poses = self.robot.data.body_link_pose_w.torch[0, self._wrist_ids].clone()
        poses[:, :3] -= self.env.scene.env_origins[0]
        return {
            "left_wrist": {"position": poses[0, :3].cpu().tolist(), "quaternion": poses[0, 3:].cpu().tolist()},
            "right_wrist": {"position": poses[1, :3].cpu().tolist(), "quaternion": poses[1, 3:].cpu().tolist()},
            "left_grip_closed": self._left_grip_closed,
            "right_grip_closed": self._right_grip_closed,
        }

    def snapshot(self, observations: dict, episode_id: int, step: int, sim_time: float) -> dict:
        """Copy observable state, with simulation time [s] and poses in environment coordinates."""
        # The dense link-state array and policy history duplicate more useful named state.
        state = {
            name: value[0].detach().cpu().tolist()
            for name, value in observations["policy"].items()
            if isinstance(value, torch.Tensor) and value.ndim < 4 and name != "robot_links_state"
        }
        contacts = {}
        contact_names = {}
        for name in ("left_hand_contact", "right_hand_contact"):
            if name in self.env.scene.sensors:
                sensor = self.env.scene.sensors[name]
                forces = sensor.data.net_forces_w
                if forces is not None:
                    contacts[name] = forces.torch[0].detach().cpu().tolist()
                    contact_names[name] = list(sensor.body_names or [])
        cameras = {}
        for name in ("robot_pov_cam", "left_wrist_cam", "right_wrist_cam"):
            if name in self.env.scene.sensors:
                data = self.env.scene.sensors[name].data
                cameras[name] = {
                    "position_environment_m": (data.pos_w.torch[0] - self.env.scene.env_origins[0]).cpu().tolist(),
                    "quaternion_ros_xyzw": data.quat_w_ros.torch[0].cpu().tolist(),
                    "intrinsic_matrix_native_pixels": data.intrinsic_matrices.torch[0].cpu().tolist(),
                    "image_size_native_wh": [data.image_shape[1], data.image_shape[0]],
                }
        snapshot = {
            "episode_id": episode_id,
            "step": step,
            "sim_time": sim_time,
            "observations": state,
            "hand_contact_forces_world_n": contacts,
            "hand_contact_body_names": contact_names,
            "hand_geometry": self._hand_geometry(state.get("object_pos")),
            "cameras": cameras,
            "joint_names": list(self.robot.data.joint_names),
            "hand_joint_positions_rad": self.robot.data.joint_pos.torch[0, self._hand_ids].cpu().tolist(),
            "grip_trajectory_active": self._grip_position is not None,
            "neutral_command": self._current_command(),
            **(self.perception.snapshot() if self.perception is not None else {}),
        }
        motion = self._object_motion(episode_id, sim_time, snapshot.get("object_contact_positions_environment_m", {}))
        if motion is not None:
            snapshot["object_motion"] = motion
        return snapshot

    def _object_pose(self) -> torch.Tensor | None:
        """Read the actor pose in environment coordinates without requiring object observations."""
        if "object" not in self.env.scene.keys():
            return None
        pose = self.env.scene["object"].data.root_link_pose_w.torch[0].detach().cpu().clone()
        pose[:3] -= self.env.scene.env_origins[0].cpu()
        return pose

    def _object_motion(self, episode_id: int, sim_time: float, contact_positions: dict) -> dict | None:
        """Compare only the reset pose and preceding inference snapshot; retain no growing history."""
        from isaaclab.utils.math import matrix_from_quat, quat_error_magnitude, subtract_frame_transforms

        pose = self._object_pose()
        if pose is None:
            return None
        if self._motion_episode_id is not None and self._motion_episode_id != episode_id:
            # Avoid comparisons across episodes even if a caller missed reset().
            self._episode_object_pose = pose.clone()
            self._previous_motion = None
        self._motion_episode_id = episode_id
        if self._episode_object_pose is None:
            self._episode_object_pose = pose.clone()
        wrists = self.robot.data.body_link_pose_w.torch[0, self._wrist_ids].detach().cpu().clone()
        wrists[:, :3] -= self.env.scene.env_origins[0].cpu()
        relative_position, relative_quaternion = subtract_frame_transforms(
            wrists[:, :3], wrists[:, 3:], pose[:3].expand(2, -1), pose[3:].expand(2, -1)
        )
        rotations = matrix_from_quat(wrists[:, 3:])
        contact_centroids = {}
        for i, side in enumerate(("left", "right")):
            for suffix in _HAND_SUFFIXES:
                name = f"{side}_hand_{suffix}_link"
                position = contact_positions.get(name)
                if position is not None:
                    point = torch.as_tensor(position, dtype=wrists.dtype)
                    if point.shape == (3,) and torch.isfinite(point).all():
                        contact_centroids[name] = (point - wrists[i, :3]) @ rotations[i]
        result = {
            "episode_initial_pose": {
                "position_environment_m": self._episode_object_pose[:3].tolist(),
                "quaternion_xyzw": self._episode_object_pose[3:].tolist(),
            },
            "height_change_from_episode_start_m": float(pose[2] - self._episode_object_pose[2]),
            "since_previous_snapshot": None,
        }
        velocity = getattr(self.env.scene["object"].data, "root_link_vel_w", None)
        if velocity is not None:
            measured_velocity = velocity.torch[0].detach().cpu()
            result["linear_velocity_world_m_s"] = measured_velocity[:3].tolist()
            result["angular_velocity_world_rad_s"] = measured_velocity[3:].tolist()
        if self._previous_motion is not None:
            old_time, old_pose, old_wrists, old_relative_position, old_relative_quaternion, old_contacts = (
                self._previous_motion
            )
            result["since_previous_snapshot"] = {
                "previous_sim_time_s": old_time,
                "elapsed_sim_seconds": sim_time - old_time,
                "object_translation_environment_m": (pose[:3] - old_pose[:3]).tolist(),
                "object_rotation_rad": float(quat_error_magnitude(pose[3:], old_pose[3:])),
                "contact_centroid_translation_wrist_m": {
                    name: (position - old_contacts[name]).tolist()
                    for name, position in contact_centroids.items()
                    if name in old_contacts
                },
                "wrists": {
                    side: {
                        "translation_environment_m": (wrists[i, :3] - old_wrists[i, :3]).tolist(),
                        "rotation_rad": float(quat_error_magnitude(wrists[i, 3:], old_wrists[i, 3:])),
                        "object_relative_translation_wrist_m": (
                            relative_position[i] - old_relative_position[i]
                        ).tolist(),
                        "object_relative_rotation_rad": float(
                            quat_error_magnitude(relative_quaternion[i], old_relative_quaternion[i])
                        ),
                    }
                    for i, side in enumerate(("left", "right"))
                },
            }
        self._previous_motion = (sim_time, pose, wrists, relative_position, relative_quaternion, contact_centroids)
        return result

    def _hand_geometry(self, object_position: list[float] | None) -> dict:
        """Copy measured finger frames and express offsets in each commanded wrist frame."""
        from isaaclab.utils.math import matrix_from_quat

        poses = self.robot.data.body_link_pose_w.torch[0, self._wrist_ids + self._hand_link_ids].detach().cpu()
        origin = self.env.scene.env_origins[0].cpu()
        geometry = {}
        for hand, side in enumerate(("left", "right")):
            wrist = poses[hand]
            rotation = matrix_from_quat(wrist[3:])
            links = poses[2 + hand * 7 : 2 + (hand + 1) * 7]
            # Row vectors times the wrist rotation express world offsets in its local frame.
            relative_positions = (links[:, :3] - wrist[:3]) @ rotation
            geometry[side] = {
                "wrist_link": _WRIST_LINKS[hand],
                "wrist_axes_environment": {axis: rotation[:, i].tolist() for i, axis in enumerate("xyz")},
                "links": {
                    suffix: {
                        "position_environment_m": (pose[:3] - origin).tolist(),
                        "quaternion_xyzw": pose[3:].tolist(),
                        "position_wrist_m": relative.tolist(),
                    }
                    for suffix, pose, relative in zip(_HAND_SUFFIXES, links, relative_positions, strict=True)
                },
            }
            if object_position is not None:
                geometry[side]["object_origin_wrist_m"] = (
                    (torch.tensor(object_position) - (wrist[:3] - origin)) @ rotation
                ).tolist()
        return geometry

    def capture_images(self, camera_names: list[str], max_size: int) -> list[dict[str, str]]:
        """Copy RGB camera buffers and encode JPEGs with longest image edge at most ``max_size`` pixels."""
        return capture_rgb_images(self.env, camera_names, max_size)

    def _vector(self, value: object, size: int, label: str) -> torch.Tensor:
        vector = torch.as_tensor(value, dtype=torch.float32, device=self.env.device)
        if vector.shape != (size,) or not torch.isfinite(vector).all():
            raise ValueError(f"{label} must contain {size} finite numbers.")
        return vector

    def action(self, command: dict) -> torch.Tensor:
        """Accept wrist goals and boolean grip goals without advancing the finger trajectory.

        Wrist positions [m] and rotations [rad] are bounded against measured state
        once per accepted command. Call :meth:`advance_grip` each control tick to
        move finger targets [rad], preserving continuity across new commands.
        """
        left_closed = command["left_grip_closed"]
        right_closed = command["right_grip_closed"]
        if type(left_closed) is not bool or type(right_closed) is not bool:
            raise ValueError("Grip commands must be booleans.")
        current = self._current_command()
        values = []
        for name in ("left_wrist", "right_wrist"):
            target = command[name]
            position = self._vector(target["position"], 3, f"{name}.position")
            measured = self._vector(current[name]["position"], 3, name)
            delta = position - measured
            distance = torch.linalg.vector_norm(delta)
            position = measured + delta * torch.clamp(0.05 / distance.clamp_min(1e-8), max=1.0)
            quaternion = self._vector(target["quaternion"], 4, f"{name}.quaternion")
            norm = torch.linalg.vector_norm(quaternion)
            if not torch.isfinite(norm) or norm < 1e-6:
                raise ValueError(f"{name}.quaternion must have a finite, nonzero norm.")
            quaternion = quaternion / norm
            measured_quat = self._vector(current[name]["quaternion"], 4, name)
            measured_norm = torch.linalg.vector_norm(measured_quat)
            if not torch.isfinite(measured_norm) or measured_norm < 1e-6:
                raise ValueError(f"Measured {name} quaternion must have a finite, nonzero norm.")
            measured_quat = measured_quat / measured_norm
            dot = torch.dot(measured_quat, quaternion)
            if dot < 0:
                quaternion = -quaternion
                dot = -dot
            half_angle = math.acos(float(dot.clamp(0, 1)))
            if half_angle > 0.1:
                fraction = 0.1 / half_angle
                quaternion = (
                    math.sin((1.0 - fraction) * half_angle) * measured_quat
                    + math.sin(fraction * half_angle) * quaternion
                ) / math.sin(half_angle)
            values.extend((position, quaternion))
        hands = (
            self.robot.data.joint_pos.torch[:, self._hand_ids].clone()
            if self._grip_position is None
            else self._grip_position.clone()
        )
        target = self._grip_targets(left_closed, right_closed)
        action = torch.cat((*values, hands[0])).unsqueeze(0)
        if not torch.isfinite(action).all() or not torch.isfinite(target).all():
            raise ValueError("The G1 state or joint limits produced a non-finite action.")
        self._left_grip_closed = left_closed
        self._right_grip_closed = right_closed
        self._grip_position = hands
        self._grip_target = target
        return action

    def advance_grip(self, action: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance finger setpoints by at most ``grip_speed * dt`` [rad], leaving wrist goals unchanged.

        Args:
            action: Previously accepted action with shape (1, 28).
            dt: Control timestep [s].

        Returns:
            A new action tensor, without changing the caller's cached action.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Grip timestep must be positive and finite.")
        if action.shape != (1, 28) or not torch.isfinite(action).all():
            raise ValueError("Expected a finite G1 action with shape (1, 28).")
        result = action.clone()
        if self._grip_position is None:
            result[:, 14:] = self.robot.data.joint_pos.torch[:, self._hand_ids]
        else:
            position = self._grip_position + (self._grip_target - self._grip_position).clamp(
                min=-self.grip_speed * dt, max=self.grip_speed * dt
            )
            # Zero-angle startup can lie outside soft limits. Clip endpoints to
            # soft limits and the path to physical limits to avoid a startup jump.
            limits = self.robot.data.joint_pos_limits.torch[:, self._hand_ids]
            position = torch.clamp(position, min=limits[..., 0], max=limits[..., 1])
            self._grip_position = position
            result[:, 14:] = position
        if not torch.isfinite(result).all():
            raise ValueError("Cannot advance a grip from non-finite robot state or joint limits.")
        return result

    def _measured_action(self) -> torch.Tensor:
        """Build a hold action from measured wrist and finger positions."""
        poses = self.robot.data.body_link_pose_w.torch[:, self._wrist_ids].clone()
        poses[..., :3] -= self.env.scene.env_origins.unsqueeze(1)
        norms = torch.linalg.vector_norm(poses[..., 3:], dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms < 1e-6).any():
            raise ValueError("Cannot hold invalid wrist orientations.")
        poses[..., 3:] /= norms
        hands = self.robot.data.joint_pos.torch[:, self._hand_ids]
        action = torch.cat((poses.flatten(start_dim=1), hands), dim=-1)
        if not torch.isfinite(action).all():
            raise ValueError("Cannot hold non-finite robot state.")
        return action

    def hold_action(self, dt: float) -> torch.Tensor:
        """Reanchor wrists to measured poses while preserving accepted finger targets."""
        return self.advance_grip(self._measured_action(), dt)

    def idle_action(self) -> torch.Tensor:
        """Stop the grip trajectory and hold measured wrists and hands, retaining desired grip booleans."""
        self._grip_position = None
        self._grip_target = None
        return self._measured_action()
