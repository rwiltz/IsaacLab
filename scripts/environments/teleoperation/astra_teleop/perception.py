# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact CAD geometry and object-specific contacts for grounded grasp observations."""

from __future__ import annotations

import copy
import itertools
import re

import numpy as np

from .g1 import _HAND_SUFFIXES, G1ActionAdapter

_SENSOR_NAME = "object_hand_contact"
_HAND_LINKS = tuple(f"{side}_hand_{suffix}_link" for side in ("left", "right") for suffix in _HAND_SUFFIXES)


def configure_object_contact_sensor(scene_cfg) -> None:
    """Report one object's contacts against fourteen individually labeled finger bodies."""
    from isaaclab.sensors import ContactSensorCfg

    scene_cfg.object.spawn.activate_contact_sensors = True
    # ContactSensor resolves contact-reporting descendants of this subtree. The
    # wheel's rigid body is nested below Object/Geometry, not on Object itself.
    setattr(
        scene_cfg,
        _SENSOR_NAME,
        ContactSensorCfg(
            prim_path=f"{scene_cfg.object.prim_path}/.*",
            filter_prim_paths_expr=[f"{scene_cfg.robot.prim_path}/{name}" for name in _HAND_LINKS],
            update_period=0.0,
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=128,
        ),
    )


def extract_object_geometry(stage, body_path: str, max_parts: int = 12) -> dict:
    """Cache composed mesh bounds in physical meters relative to the rigid-body pose.

    USD transforms include inherited asset scale; rigid-body state poses do not.
    Removing only the body's rigid transform retains that scale in these vertices.
    Geometry is CAD information, not a visual estimate or a collision guarantee.
    """
    from pxr import UsdGeom

    from isaaclab.sim.utils import get_all_matching_child_prims

    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        raise ValueError(f"Object rigid body does not exist: {body_path}")
    cache = UsdGeom.XformCache()
    world_to_body = cache.GetLocalToWorldTransform(body).RemoveScaleShear().GetInverse()
    parts = []
    all_bounds = []
    for prim in get_all_matching_child_prims(body_path, lambda child: child.IsA(UsdGeom.Mesh), stage=stage):
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            continue
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * world_to_body)
        points = points @ transform[:3, :3] + transform[3, :3]
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        all_bounds.append(bounds)
        part = {"name": prim.GetName(), "bounds_m": bounds.round(6).tolist()}
        # Only authored rim labels receive this optional geometric descriptor.
        # A radial gap separates this asset's center detail from its outer ring.
        if re.search(r"(?:^|[_-])rim(?:[_-]|$)", prim.GetName(), re.IGNORECASE):
            radii = np.linalg.norm(points[:, :2], axis=1)
            unique = np.unique(radii)
            band = points
            if len(unique) > 1 and unique[-1] > 0:
                split = int(np.argmax(np.diff(unique)))
                if unique[split + 1] - unique[split] > 0.2 * unique[-1]:
                    band = points[radii >= unique[split + 1]]
                band_radii = np.linalg.norm(band[:, :2], axis=1)
                if len(band) >= 3 and band_radii.min() > 0.5 * unique[-1]:
                    part["outer_radial_band"] = {
                        "axis_object": "+Z through object origin",
                        "radius_range_m": [round(float(band_radii.min()), 6), round(float(band_radii.max()), 6)],
                        "z_range_m": [round(float(band[:, 2].min()), 6), round(float(band[:, 2].max()), 6)],
                        "source": "outer vertex band of authored rim mesh; not a grasp target",
                    }
        parts.append(part)
    if not parts:
        return {"source": "USD", "available": False, "reason": "No finite mesh vertices below object rigid body"}
    parts.sort(key=lambda part: ("outer_radial_band" not in part, part["name"]))
    return {
        "source": "composed USD mesh vertices, not camera pixels or collision contact surfaces",
        "frame": "object rigid-body frame in meters, with asset scale included; transform by object_pos/object_rot",
        "bounds_m": np.stack((np.min(all_bounds, axis=(0, 1)), np.max(all_bounds, axis=(0, 1)))).round(6).tolist(),
        "mesh_count": len(parts),
        "parts": parts[:max_parts],
    }


def extract_bin_geometry(stage, background_path: str, environment_origin: np.ndarray) -> dict:
    """Read authored bin collision cubes, preserving composed scale and environment origin.

    Interior bounds are derived only for a floor and four axis-aligned walls in
    the container frame. They describe free space, not a placement success test.
    """
    from pxr import UsdGeom, UsdPhysics

    from isaaclab.sim.utils import get_all_matching_child_prims

    cache = UsdGeom.XformCache()
    containers = []
    bodies = get_all_matching_child_prims(
        background_path,
        lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
        and re.search(r"(?:^|[_-])(?:bin|container)(?:[_-]|$)", prim.GetName(), re.IGNORECASE),
        stage=stage,
    )
    for body in bodies:
        pose = cache.GetLocalToWorldTransform(body).RemoveScaleShear()
        inverse = pose.GetInverse()
        rotation = pose.ExtractRotationQuat()
        colliders = []
        local_bounds = []
        aligned = True
        for prim in get_all_matching_child_prims(
            body.GetPath(),
            lambda child: child.IsA(UsdGeom.Cube) and child.HasAPI(UsdPhysics.CollisionAPI),
            stage=stage,
        ):
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            corners = np.asarray(list(itertools.product((-size / 2, size / 2), repeat=3)))
            world_transform = cache.GetLocalToWorldTransform(prim)
            local_transform = np.asarray(world_transform * inverse)
            aligned &= np.allclose(local_transform[:3, :3], np.diag(np.diag(local_transform[:3, :3])), atol=1e-6)
            local_points = corners @ local_transform[:3, :3] + local_transform[3, :3]
            local_bounds.append(np.stack((local_points.min(axis=0), local_points.max(axis=0))))
            world_transform = np.asarray(world_transform)
            points = corners @ world_transform[:3, :3] + world_transform[3, :3] - environment_origin
            colliders.append(
                {
                    "name": prim.GetName(),
                    "bounds_environment_m": [points.min(0).round(6).tolist(), points.max(0).round(6).tolist()],
                }
            )
        if not colliders:
            continue
        container = {
            "name": body.GetName(),
            "prim_path": str(body.GetPath()),
            "position_environment_m": (np.asarray(pose.ExtractTranslation()) - environment_origin).round(6).tolist(),
            "quaternion_xyzw": np.asarray([*rotation.GetImaginary(), rotation.GetReal()]).round(6).tolist(),
            "colliders": colliders,
        }
        interior = _box_interior(np.asarray(local_bounds)) if aligned else None
        if interior is not None:
            container["interior_bounds_container_m"] = interior.round(6).tolist()
            points = np.asarray(list(itertools.product(*zip(*interior, strict=True))))
            transform = np.asarray(pose)
            points = points @ transform[:3, :3] + transform[3, :3] - environment_origin
            container["interior_aabb_environment_m"] = [
                points.min(0).round(6).tolist(),
                points.max(0).round(6).tolist(),
            ]
        containers.append(container)
    return {
        "source": "composed USD collision cubes under authored bin/container rigid bodies in the static background",
        "frame": "meters; environment bounds subtract env origin; container bounds use its supplied XYZ/XYZW pose",
        "interior_note": (
            "Interior derives from one floor and four walls; environment AABB is only an enclosure if rotated. "
            "Object containment and release require current object geometry and contacts."
        ),
        "containers": containers,
    }


def _box_interior(bounds: np.ndarray) -> np.ndarray | None:
    """Recognize the thin axes of one floor and two opposing pairs of walls."""
    thin_axes = np.argmin(bounds[:, 1] - bounds[:, 0], axis=1)
    if [int(np.count_nonzero(thin_axes == axis)) for axis in range(3)] != [2, 2, 1]:
        return None
    floor = bounds[thin_axes == 2][0]
    walls_x = sorted(bounds[thin_axes == 0], key=lambda wall: wall[0, 0])
    walls_y = sorted(bounds[thin_axes == 1], key=lambda wall: wall[0, 1])
    interior = np.asarray(
        [
            [walls_x[0][1, 0], walls_y[0][1, 1], floor[1, 2]],
            [walls_x[1][0, 0], walls_y[1][0, 1], bounds[thin_axes != 2, 1, 2].min()],
        ]
    )
    return interior if np.all(interior[1] > interior[0]) else None


class ObjectPerception:
    """Own cached USD geometry and copy current filtered forces on the simulation thread."""

    def __init__(self, env):
        from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

        self.sensor = env.scene.sensors[_SENSOR_NAME]
        self._environment_origin = env.scene.env_origins[0].detach().cpu().numpy().copy()
        if env.num_envs != 1 or self.sensor.num_sensors != 1:
            raise ValueError(
                "Astra object contact filtering requires exactly one rigid object body and one environment."
            )
        stage = get_current_stage()
        self.geometry = extract_object_geometry(stage, self.sensor.body_physx_view.prim_paths[0])
        self.bin_geometry = None
        if "packing_table" in env.scene.keys():
            paths = find_matching_prim_paths(env.scene["packing_table"].cfg.prim_path, stage=stage)
            if len(paths) != 1:
                raise ValueError("Astra bin geometry requires one configured packing-table background.")
            self.bin_geometry = extract_bin_geometry(stage, paths[0], self._environment_origin)

    def snapshot(self) -> dict:
        """Return detached JSON values; force vectors act on the object, not the fingers."""
        # Read all fields from one sensor update, without advancing simulation.
        data = self.sensor.data
        matrix = data.force_matrix_w
        if matrix is None or tuple(matrix.torch.shape) != (1, 1, len(_HAND_LINKS), 3):
            raise ValueError("Object contact sensor must report one force column per configured finger body.")
        total = data.net_forces_w
        positions = data.contact_pos_w
        if total is None or tuple(total.torch.shape) != (1, 1, 3):
            raise ValueError("Object contact sensor must report one total contact-force vector.")
        if positions is None or tuple(positions.torch.shape) != (1, 1, len(_HAND_LINKS), 3):
            raise ValueError("Object contact sensor must track one mean contact position per finger body.")
        forces = matrix.torch[0, 0].detach().cpu().numpy()
        total = total.torch[0, 0].detach().cpu().numpy()
        positions = positions.torch[0, 0].detach().cpu().numpy() - self._environment_origin
        if not np.isfinite(forces).all() or not np.isfinite(total).all():
            raise ValueError("Object contact forces must be finite.")
        return {
            "object_geometry": copy.deepcopy(self.geometry),
            "object_contact_forces_world_n": dict(zip(_HAND_LINKS, forces.tolist(), strict=True)),
            "object_total_contact_force_world_n": total.tolist(),
            "object_nonfinger_contact_force_world_n": (total - forces.sum(axis=0)).tolist(),
            "object_contact_positions_environment_m": {
                name: position.tolist() if np.isfinite(position).all() else None
                for name, position in zip(_HAND_LINKS, positions, strict=True)
            },
            "object_contact_measurement": {
                "force": (
                    "Latest physics-step contact force on object in world N; "
                    "normal + friction in PhysX 6.2, excluding gravity."
                ),
                "nonfinger": (
                    "Total minus fourteen filtered finger vectors; includes palms, other robot links, table and bin. "
                    "Opposing contacts can cancel; zero alone does not prove suspension."
                ),
                "position": (
                    "Unweighted mean of reported normal-contact points per object/finger pair in environment meters; "
                    "null means no finite contact position, not a known pad center."
                ),
            },
            **({"bin_geometry": copy.deepcopy(self.bin_geometry)} if self.bin_geometry is not None else {}),
        }


class BinDropObserver:
    """Measure bin-drop completion from composed geometry and all robot-body contacts.

    Construct after the environment's first reset, when PhysX views are available.
    Geometry is cached once; each update copies current measurements on the
    simulation thread. Resetting an episode retains the unchanged asset geometry
    and contact view while clearing the task's accumulated settling time.
    """

    def __init__(self, env, perception: ObjectPerception, adapter: G1ActionAdapter, original_success_cfg):
        from pxr import UsdGeom

        from isaaclab.sim.utils import get_all_matching_child_prims, get_current_stage

        from .task import BinDropTask

        manager = env.sim.backend_manager
        if manager.backend_name != "physx":
            raise ValueError("Bin-drop completion requires the PhysX 6.2 backend with filtered contact forces.")
        physics_view = manager.get_physics_sim_view()
        if physics_view is None or not hasattr(physics_view, "create_rigid_contact_view"):
            raise RuntimeError("Create BinDropObserver after resetting the PhysX environment.")
        if env.num_envs != 1 or perception.bin_geometry is None:
            raise ValueError("Bin-drop completion requires one environment and authored bin geometry.")

        self._env = env
        self._adapter = adapter
        self._contacts_fresh = False
        self._origin = env.scene.env_origins[0].detach().cpu().numpy().copy()
        body_path = perception.sensor.body_physx_view.prim_paths[0]
        stage = get_current_stage()
        body = stage.GetPrimAtPath(body_path)
        if not body.IsValid():
            raise ValueError(f"Object rigid body does not exist: {body_path}")
        cache = UsdGeom.XformCache()
        world_to_body = cache.GetLocalToWorldTransform(body).RemoveScaleShear().GetInverse()
        meshes = []
        mesh_edges = []
        vertex_offset = 0
        for prim in get_all_matching_child_prims(body_path, lambda child: child.IsA(UsdGeom.Mesh), stage=stage):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError(f"Bin-drop completion requires finite mesh vertices: {prim.GetPath()}")
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            if (
                counts.ndim != 1
                or indices.ndim != 1
                or not len(counts)
                or not np.issubdtype(counts.dtype, np.integer)
                or not np.issubdtype(indices.dtype, np.integer)
                or (counts < 3).any()
                or counts.sum() != len(indices)
                or (indices < 0).any()
                or (indices >= len(points)).any()
            ):
                raise ValueError(f"Bin-drop completion requires valid mesh polygon topology: {prim.GetPath()}")
            ends = np.cumsum(counts) - 1
            following = np.roll(indices, -1)
            following[ends] = indices[ends - counts + 1]
            # Include each polygon's closing edge. Shared undirected edges only
            # need one wall-height intersection per update.
            edges = np.sort(np.column_stack((indices, following)), axis=1)
            edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
            if not len(edges):
                raise ValueError(f"Bin-drop completion requires nondegenerate mesh edges: {prim.GetPath()}")
            mesh_edges.append(edges + vertex_offset)
            vertex_offset += len(points)
            # USD uses row-vector transforms. Remove only rigid pose so inherited
            # asset scale remains in the physical, body-local vertex positions.
            transform = np.asarray(cache.GetLocalToWorldTransform(prim) * world_to_body)
            meshes.append(points @ transform[:3, :3] + transform[3, :3])
        if not meshes:
            raise ValueError("Bin-drop completion requires object mesh vertices below its rigid body.")
        self.task = BinDropTask(
            perception.bin_geometry,
            np.concatenate(meshes),
            original_success_cfg,
            object_edges=np.concatenate(mesh_edges),
        )

        robot_paths = list(env.scene["robot"].root_view.link_paths[0])
        if not robot_paths or len(set(robot_paths)) != len(robot_paths):
            raise ValueError("Bin-drop completion requires a distinct contact filter for every robot body.")
        self._robot_body_names = [path.rsplit("/", 1)[-1] for path in robot_paths]
        bin_path = perception.bin_geometry["containers"][0]["prim_path"]
        self._filter_count = len(robot_paths) + 1
        self._contact_view = physics_view.create_rigid_contact_view(
            [body_path], filter_patterns=[robot_paths + [bin_path]], max_contact_data_count=0
        )
        if not all(
            hasattr(self._contact_view, method)
            for method in ("get_normal_contact_force_matrix", "get_friction_contact_force_matrix")
        ):
            raise RuntimeError("Bin-drop completion requires PhysX 6.2 normal and friction contact matrices.")
        if self._contact_view.filter_count != self._filter_count:
            raise RuntimeError("Bin-drop contact view must expose one column per robot body plus the bin.")

    def reset(self) -> None:
        """Discard settling time and stale contact buffers without rebuilding physics views."""
        self._contacts_fresh = False
        self.task.reset()

    def update(self, dt: float) -> dict:
        """Evaluate the current state after ``dt`` executed simulation seconds."""
        if isinstance(dt, bool) or not np.isfinite(dt) or dt < 0:
            raise ValueError("Bin-drop timestep must be finite and nonnegative.")
        self._contacts_fresh |= dt > 0
        forces = None
        if self._contacts_fresh:
            normal = self._contact_view.get_normal_contact_force_matrix(dt=self._env.physics_dt)
            forces = np.array(normal.numpy(), copy=True)
            friction = self._contact_view.get_friction_contact_force_matrix(dt=self._env.physics_dt)
            forces += friction.numpy()
            if forces.shape != (1, self._filter_count, 3) or not np.isfinite(forces).all():
                raise ValueError("Bin-drop contact forces must be finite with one column per robot body plus the bin.")
        data = self._env.scene["object"].data
        pose = data.root_link_pose_w.torch[0].detach().cpu().numpy().copy()
        pose[:3] -= self._origin
        velocity = data.root_link_vel_w.torch[0].detach().cpu().numpy().copy()
        progress = self.task.update(
            object_pose_environment_xyzw=pose,
            object_velocity_world=velocity,
            grippers_open=self._adapter.grippers_open(),
            robot_contact_forces_world_n=None if forces is None else forces[0, :-1],
            bin_contact_force_world_n=None if forces is None else forces[0, -1],
            dt=dt,
        )
        # The reporting threshold only bounds observation noise; success uses all
        # body forces above, including those omitted from this explanatory list.
        threshold = 0.001
        progress.update(
            robot_contacts_min_magnitude_n=threshold,
            robot_contacts=None
            if forces is None
            else [
                {"body": name, "force_world_n": force.tolist()}
                for name, force in zip(self._robot_body_names, forces[0, :-1], strict=True)
                if np.linalg.norm(force) >= threshold
            ],
            bin_contact_force_world_n=None if forces is None else forces[0, -1].tolist(),
        )
        return progress
