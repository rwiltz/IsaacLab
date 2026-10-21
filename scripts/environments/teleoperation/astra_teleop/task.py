# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment-grounded bin-drop criteria and simulation-time success tracking."""

from __future__ import annotations

import copy
import inspect
import math

import numpy as np


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return the local-to-parent rotation for a finite XYZW quaternion."""
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Task poses require finite, nonzero XYZW quaternions.")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _legacy_success(success_cfg) -> dict:
    """Describe the explicitly supported environment term, without parsing arbitrary task code."""
    from isaaclab_tasks.contrib.pick_place.mdp.terminations import task_done_pick_place

    if success_cfg is None or success_cfg.func is not task_done_pick_place:
        raise ValueError("Bin-drop criteria require the known task_done_pick_place environment success term.")
    arguments = inspect.signature(task_done_pick_place).bind_partial(None, **success_cfg.params)
    arguments.apply_defaults()
    values = arguments.arguments
    return {
        "source": f"{task_done_pick_place.__module__}.{task_done_pick_place.__qualname__}",
        "object_asset": values["object_cfg"].name,
        "task_link_name": values["task_link_name"],
        "resolved_parameters": {
            name: values[name]
            for name in ("right_wrist_max_x", "min_x", "max_x", "min_y", "max_y", "max_height", "min_vel")
        },
        "semantics": (
            "Original term: object-root X/Y strictly inside the configured ranges; root Z below max_height; "
            "each absolute linear-velocity component below min_vel; named wrist X below right_wrist_max_x. "
            "It does not require whole-object containment, released hands, bin contact, angular settling, "
            "or sustained success. The bin-drop criteria below replace this coarse term."
        ),
    }


class BinDropTask:
    """Track a released, settled object using authored geometry and measured contacts.

    The caller supplies all robot-body contact columns and combines commanded and
    measured opening into each ``grippers_open`` flag. Call ``update`` after each
    executed control step, and ``reset`` at episode boundaries. No simulation or
    wall-clock reads occur here.
    """

    def __init__(self, bin_geometry: dict, object_points_body_m, original_success_cfg, *, object_edges):
        legacy = _legacy_success(original_success_cfg)
        containers = bin_geometry["containers"]
        if len(containers) != 1 or "interior_bounds_container_m" not in containers[0]:
            raise ValueError("Bin-drop criteria require exactly one authored bin with a complete interior.")
        container = containers[0]
        self._points = np.array(object_points_body_m, dtype=float, copy=True)
        if (
            self._points.ndim != 2
            or self._points.shape[1] != 3
            or len(self._points) < 4
            or not np.isfinite(self._points).all()
        ):
            raise ValueError("Bin-drop criteria require finite mesh vertices in the object rigid-body frame.")
        self._edges = np.array(object_edges, copy=True)
        if (
            self._edges.ndim != 2
            or self._edges.shape[1] != 2
            or not len(self._edges)
            or not np.issubdtype(self._edges.dtype, np.integer)
            or np.any(self._edges < 0)
            or np.any(self._edges >= len(self._points))
        ):
            raise ValueError("Bin-drop criteria require nonempty integer mesh edges with valid vertex indices.")
        self._sphere_center = (self._points.min(axis=0) + self._points.max(axis=0)) * 0.5
        self._sphere_radius = float(np.linalg.norm(self._points - self._sphere_center, axis=1).max())
        self._interior = np.array(container["interior_bounds_container_m"], dtype=float, copy=True)
        self._bin_position = np.array(container["position_environment_m"], dtype=float, copy=True)
        bin_quaternion = np.array(container["quaternion_xyzw"], dtype=float, copy=True)
        if (
            self._interior.shape != (2, 3)
            or not np.isfinite(self._interior).all()
            or not (self._interior[1] > self._interior[0]).all()
            or self._bin_position.shape != (3,)
            or not np.isfinite(self._bin_position).all()
            or bin_quaternion.shape != (4,)
        ):
            raise ValueError("Bin-drop criteria require finite, ordered bin bounds and a valid bin pose.")
        self._bin_rotation = _rotation(bin_quaternion)
        sphere_region = np.stack(
            (self._interior[0, :2] + self._sphere_radius, self._interior[1, :2] - self._sphere_radius)
        )
        sphere_fits = bool(np.all(sphere_region[1] >= sphere_region[0]))
        self._contract = {
            "task": "Drop the whole object inside the bin, release it, and let it settle.",
            "legacy_environment_success": legacy,
            "target_bin": copy.deepcopy(container),
            "geometry_source": bin_geometry["source"],
            "object_extent_source": "Configured mesh vertices and face edges in the scaled rigid-body frame.",
            "success_criteria": {
                "below_wall_mesh_inside_bin": True,
                "mesh_center_xy_inside_bin": True,
                "absolute_mesh_bottom_floor_gap_max_m": 0.01,
                "both_grippers_commanded_and_measured_open": True,
                "sum_robot_body_contact_magnitudes_max_n": 0.0,
                "bin_upward_support_min_n_exclusive": 1.0,
                "linear_speed_max_m_s": 0.05,
                "angular_speed_max_rad_s": 0.25,
                "settled_sim_seconds": 1.0,
            },
            "drop_ready_criteria": {
                "full_mesh_xy_margin_min_m": 0.02,
                "mesh_bottom_at_or_above_wall_top": True,
                "mesh_bottom_above_floor_max_m": 0.35,
            },
            "rotation_clearance": {
                "sphere_center_object_m": self._sphere_center.tolist(),
                "sphere_radius_m": self._sphere_radius,
                "sphere_fits_bin_xy": sphere_fits,
                "sphere_center_region_bin_xy_m": sphere_region.tolist() if sphere_fits else None,
                "note": (
                    "Sphere encloses every scaled mesh vertex, centered at the body-frame mesh AABB center. "
                    "With this sphere center fixed in the bin XY region, the mesh footprint stays inside for "
                    "any orientation. This is conservative guidance, not a success or drop-ready condition. "
                    "It does not predict translation, bounce, or collision clearance; rotation about a wrist "
                    "or contact can move the sphere center. A null region means this sphere cannot fit."
                ),
            },
            "notes": (
                "Bounds and margins use the bin's local frame. Above-wall overhang is permitted. The mesh portion "
                "at or below wall-top height must be nonempty and inside the bin XY bounds, including edge "
                "intersections with that height plane. The transformed body-frame mesh AABB center must also "
                "lie inside the bin XY bounds. Drop-ready uses the full footprint as guidance, not success "
                "or a collision-free path guarantee. Robot contact uses the sum of per-body vector magnitudes, "
                "not the magnitude of their sum. Unknown contacts are null and cannot establish release or support. "
                "Contacts are unavailable after reset until a physics step executes. Settling counts "
                "executed simulation time only. Success requires the final outcome, not a handoff history."
            ),
        }
        self.reset()

    def describe(self) -> dict:
        """Return the task contract to include with the initial and subsequent model observations."""
        return copy.deepcopy(self._contract)

    def reset(self) -> None:
        """Clear accumulated settling time for a new episode."""
        self._settled_seconds = 0.0

    def update(
        self,
        *,
        object_pose_environment_xyzw,
        object_velocity_world,
        grippers_open,
        robot_contact_forces_world_n,
        bin_contact_force_world_n,
        dt: float,
    ) -> dict:
        """Evaluate measured state and accumulate uninterrupted settling time.

        Args:
            object_pose_environment_xyzw: Object rigid-body position and XYZW orientation, shape (7,).
            object_velocity_world: World linear and angular velocity, shape (6,), in m/s and rad/s.
            grippers_open: Two booleans, each requiring commanded AND measured opening.
            robot_contact_forces_world_n: Force from every robot body, shape (N, 3), or None if unavailable.
            bin_contact_force_world_n: Force on the object from the bin, shape (3,), or None if unavailable.
            dt: Executed simulation seconds; zero permits an initial or paused-state observation.
        """
        pose = np.asarray(object_pose_environment_xyzw, dtype=float)
        velocity = np.asarray(object_velocity_world, dtype=float)
        if (
            pose.shape != (7,)
            or velocity.shape != (6,)
            or not all(np.isfinite(value).all() for value in (pose, velocity))
        ):
            raise ValueError("Bin-drop measurements must include finite poses and velocities.")
        if len(grippers_open) != 2 or any(type(value) is not bool for value in grippers_open):
            raise ValueError("Bin-drop measurements require two commanded-and-measured open booleans.")
        if isinstance(dt, bool) or not math.isfinite(dt) or dt < 0:
            raise ValueError("Bin-drop timestep must be finite and nonnegative.")
        robot_contact = None
        if robot_contact_forces_world_n is not None:
            robot_forces = np.asarray(robot_contact_forces_world_n, dtype=float)
            if (
                robot_forces.ndim != 2
                or robot_forces.shape[1] != 3
                or not len(robot_forces)
                or not np.isfinite(robot_forces).all()
            ):
                raise ValueError("Robot-body contact forces must be finite with shape (N, 3).")
            robot_contact = float(np.linalg.norm(robot_forces, axis=1).sum())
        support = None
        if bin_contact_force_world_n is not None:
            force = np.asarray(bin_contact_force_world_n, dtype=float)
            if force.shape != (3,) or not np.isfinite(force).all():
                raise ValueError("Bin contact force must contain three finite numbers.")
            support = float(force @ self._bin_rotation[:, 2])
        object_rotation = _rotation(pose[3:])
        points_environment = self._points @ object_rotation.T + pose[:3]
        points_bin = (points_environment - self._bin_position) @ self._bin_rotation
        sphere_center_bin = (
            self._sphere_center @ object_rotation.T + pose[:3] - self._bin_position
        ) @ self._bin_rotation
        rotation_margin = float(
            np.min(
                np.concatenate(
                    (
                        sphere_center_bin[:2] - self._sphere_radius - self._interior[0, :2],
                        self._interior[1, :2] - sphere_center_bin[:2] - self._sphere_radius,
                    )
                )
            )
        )
        bounds = np.stack((points_bin.min(axis=0), points_bin.max(axis=0)))
        xy_margin = float(
            np.min(np.concatenate((bounds[0, :2] - self._interior[0, :2], self._interior[1, :2] - bounds[1, :2])))
        )
        wall_top = self._interior[1, 2]
        below = points_bin[:, 2] <= wall_top
        below_wall_margin = None
        if np.any(below):
            clipped = points_bin[below]
            if not np.all(below):
                crossing_edges = self._edges[below[self._edges[:, 0]] != below[self._edges[:, 1]]]
                if len(crossing_edges):
                    start = points_bin[crossing_edges[:, 0]]
                    delta = points_bin[crossing_edges[:, 1]] - start
                    fraction = (wall_top - start[:, 2]) / delta[:, 2]
                    clipped = np.concatenate((clipped, start + fraction[:, None] * delta))
            below_wall_margin = float(
                np.min(
                    np.concatenate(
                        (
                            clipped[:, :2].min(axis=0) - self._interior[0, :2],
                            self._interior[1, :2] - clipped[:, :2].max(axis=0),
                        )
                    )
                )
            )
        floor_gap = float(bounds[0, 2] - self._interior[0, 2])
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        criteria = self._contract["success_criteria"]
        conditions = {
            "below_wall_mesh_inside_bin": below_wall_margin is not None and below_wall_margin >= 0,
            "mesh_center_xy_inside_bin": bool(
                np.all(sphere_center_bin[:2] >= self._interior[0, :2])
                and np.all(sphere_center_bin[:2] <= self._interior[1, :2])
            ),
            "mesh_bottom_near_floor": abs(floor_gap) <= criteria["absolute_mesh_bottom_floor_gap_max_m"],
            "both_grippers_open": all(grippers_open),
            "robot_released": robot_contact is not None
            and robot_contact <= criteria["sum_robot_body_contact_magnitudes_max_n"],
            "bin_support": support is not None and support > criteria["bin_upward_support_min_n_exclusive"],
            "linear_settled": linear_speed <= criteria["linear_speed_max_m_s"],
            "angular_settled": angular_speed <= criteria["angular_speed_max_rad_s"],
        }
        self._settled_seconds = self._settled_seconds + dt if all(conditions.values()) else 0.0
        drop_criteria = self._contract["drop_ready_criteria"]
        return {
            "success": self._settled_seconds + 1e-9 >= criteria["settled_sim_seconds"],
            "settled_sim_seconds": self._settled_seconds,
            "conditions": conditions,
            "measurements": {
                "object_bounds_bin_m": bounds.tolist(),
                "floor_gap_m": floor_gap,
                "xy_margin_m": xy_margin,
                "below_wall_xy_margin_m": below_wall_margin,
                "sphere_center_bin_m": sphere_center_bin.tolist(),
                "rotation_safe_xy_margin_m": rotation_margin,
                "robot_contact_magnitude_sum_n": robot_contact,
                "bin_upward_support_n": support,
                "linear_speed_m_s": linear_speed,
                "angular_speed_rad_s": angular_speed,
            },
            "drop_ready": bool(
                xy_margin >= drop_criteria["full_mesh_xy_margin_min_m"]
                and bounds[0, 2] >= self._interior[1, 2]
                and floor_gap <= drop_criteria["mesh_bottom_above_floor_max_m"]
            ),
        }
