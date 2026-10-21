# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect bin-drop completion against unsupported, partial, or transient placement."""

import copy
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Usd, UsdGeom

from isaaclab_tasks.contrib.pick_place.mdp.terminations import task_done_pick_place

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.perception import BinDropObserver  # noqa: E402
from astra_teleop.task import BinDropTask  # noqa: E402

_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


@pytest.fixture
def task_and_state():
    quarter_turn = [0, 0, math.sqrt(0.5), math.sqrt(0.5)]
    geometry = {
        "source": "authored test bin floor and wall colliders",
        "containers": [
            {
                "name": "bin",
                "position_environment_m": [2, -1, 0.5],
                "quaternion_xyzw": quarter_turn,
                "interior_bounds_container_m": [[-0.4, -0.25, 0.02], [0.4, 0.25, 0.1]],
            }
        ],
    }
    # The object is deliberately taller than the bin walls when lying on its floor.
    points = list(itertools.product((-0.2, 0.2), (-0.1, 0.1), (-0.08, 0.08)))
    original = SimpleNamespace(
        func=task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link", "min_x": 0.41}
    )
    task = BinDropTask(geometry, points, original, object_edges=_BOX_EDGES)
    state = {
        "object_pose_environment_xyzw": [2, -1, 0.6, *quarter_turn],
        "object_velocity_world": [0] * 6,
        "grippers_open": [True, True],
        "robot_contact_forces_world_n": [[0, 0, 0], [0, 0, 0]],
        "bin_contact_force_world_n": [0, 0, 5.7],
    }
    return task, state


def test_mesh_containment_uses_rotated_bin_frame_without_wall_top_height_rule(task_and_state):
    task, state = task_and_state
    for _ in range(3):
        assert not task.update(**state, dt=0.25)["success"]
    result = task.update(**state, dt=0.25)
    assert result["success"]
    assert np.asarray(result["measurements"]["object_bounds_bin_m"]) == pytest.approx(
        np.array([[-0.2, -0.1, 0.02], [0.2, 0.1, 0.18]])
    )
    assert result["measurements"]["xy_margin_m"] == pytest.approx(0.15)
    assert not result["drop_ready"]
    # The root is still inside the bin, but the object overhangs the local -Y wall.
    state["object_pose_environment_xyzw"][0] += 0.2
    outside = task.update(**state, dt=1.0)
    assert not outside["conditions"]["below_wall_mesh_inside_bin"]
    assert outside["measurements"]["xy_margin_m"] == pytest.approx(-0.05)
    assert not outside["success"]
    assert outside["settled_sim_seconds"] == 0


@pytest.mark.parametrize(
    ("apex", "height_offset", "expected_margin", "center_inside", "success"),
    [
        pytest.param((0.5, 0, 0.2), 0, 0.4 - (0.2 + 0.3 * 0.08 / 0.28), True, True, id="overhang-above-rim"),
        pytest.param((0.5, 0, -0.02), 0, -0.1, True, False, id="overhang-below-rim"),
        pytest.param((0.5, 0, 0.02), 0, -0.04, True, False, id="crossing-edge-outside"),
        pytest.param((0.5, 0, 0.2), 0.2, None, True, False, id="entire-mesh-above-rim"),
        pytest.param((1.2, 0, 1.2), 0, 0.1375, False, False, id="mesh-center-outside"),
    ],
)
def test_containment_clips_mesh_edges_at_wall_top(
    task_and_state, apex, height_offset, expected_margin, center_inside, success
):
    original_task, state = task_and_state
    geometry = {"source": "authored fixture bin", "containers": [original_task.describe()["target_bin"]]}
    points = np.array([[-0.2, -0.1, -0.08], [0.2, -0.1, -0.08], [0, 0.1, -0.08], apex])
    points[:, 2] += height_offset
    # All six edges of a tetrahedron. For crossing-edge-outside, every vertex
    # below the rim is inside, but its edge to the exterior apex crosses outside.
    edges = list(itertools.combinations(range(4), 2))
    original = SimpleNamespace(func=task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})
    task = BinDropTask(geometry, points, original, object_edges=edges)
    result = task.update(**state, dt=1.0)
    margin = result["measurements"]["below_wall_xy_margin_m"]
    assert margin is None if expected_margin is None else margin == pytest.approx(expected_margin)
    assert result["measurements"]["xy_margin_m"] < 0  # Full-footprint guidance still reports the overhang.
    assert result["conditions"]["below_wall_mesh_inside_bin"] is (expected_margin is not None and expected_margin >= 0)
    assert result["conditions"]["mesh_center_xy_inside_bin"] is center_inside
    assert result["success"] is success
    json.dumps(result, allow_nan=False)
    if success:
        state["robot_contact_forces_world_n"] = [[0.0005, 0, 0]]
        assert not task.update(**state, dt=1.0)["success"]  # Above-rim overhang never relaxes release.


@pytest.mark.parametrize("edges", [[], [[0.0, 1.0]], [[0, 8]], [[-1, 0]]])
def test_mesh_edges_require_nonempty_integer_indices_within_vertex_range(task_and_state, edges):
    original_task, _state = task_and_state
    geometry = {"source": "authored fixture bin", "containers": [original_task.describe()["target_bin"]]}
    points = list(itertools.product((-0.2, 0.2), (-0.1, 0.1), (-0.08, 0.08)))
    original = SimpleNamespace(func=task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})
    with pytest.raises(ValueError, match="nonempty integer mesh edges"):
        BinDropTask(geometry, points, original, object_edges=edges)


def test_rotation_clearance_catches_thin_near_wall_footprint_and_reports_impossible_sphere_fit(task_and_state):
    original_task, state = task_and_state
    container = original_task.describe()["target_bin"]
    geometry = {"source": "authored fixture bin", "containers": [container]}
    # Thin upright wheel, with its geometry center 5 cm above the body origin.
    points = [(x, y, z + 0.05) for x, z in ((0.14, 0), (0, 0.14), (-0.14, 0), (0, -0.14)) for y in (-0.005, 0.005)]
    edges = [(0, 2), (2, 4), (4, 6), (6, 0), (1, 3), (3, 5), (5, 7), (7, 1), (0, 1), (2, 3), (4, 5), (6, 7)]
    original = SimpleNamespace(func=task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})
    task = BinDropTask(geometry, points, original, object_edges=edges)
    radius = math.hypot(0.14, 0.005)
    guidance = task.describe()["rotation_clearance"]
    assert guidance["sphere_center_object_m"] == pytest.approx([0, 0, 0.05])
    assert guidance["sphere_radius_m"] == pytest.approx(radius)
    assert guidance["sphere_fits_bin_xy"]
    assert np.asarray(guidance["sphere_center_region_bin_xy_m"]) == pytest.approx(
        np.array([[-0.4 + radius, -0.25 + radius], [0.4 - radius, 0.25 - radius]])
    )

    # The bin is rotated 90 degrees: environment +X is bin -Y.
    state["object_pose_environment_xyzw"][:3] = [2.22, -1, 0.8]
    near_wall = task.update(**state, dt=0)
    measured = near_wall["measurements"]
    assert measured["xy_margin_m"] == pytest.approx(0.025)
    assert measured["sphere_center_bin_m"] == pytest.approx([0, -0.22, 0.35])
    assert measured["rotation_safe_xy_margin_m"] == pytest.approx(0.03 - radius)
    assert near_wall["drop_ready"]  # Existing oriented-footprint guidance remains unchanged.

    state["object_pose_environment_xyzw"][0] = 2
    centered = task.update(**state, dt=0)
    assert centered["measurements"]["rotation_safe_xy_margin_m"] == pytest.approx(0.25 - radius)
    # Rolling around the body origin moves the offset sphere center; it is not a fixed-center rotation.
    state["object_pose_environment_xyzw"][3:] = [0.5, 0.5, 0.5, 0.5]
    rolled = task.update(**state, dt=0)
    assert rolled["measurements"]["sphere_center_bin_m"] == pytest.approx([0, -0.05, 0.3])
    assert rolled["measurements"]["rotation_safe_xy_margin_m"] == pytest.approx(0.20 - radius)

    container["interior_bounds_container_m"][0][1] = -0.1
    container["interior_bounds_container_m"][1][1] = 0.1
    narrow_task = BinDropTask(geometry, points, original, object_edges=edges)
    narrow_guidance = narrow_task.describe()["rotation_clearance"]
    assert not narrow_guidance["sphere_fits_bin_xy"]
    assert narrow_guidance["sphere_center_region_bin_xy_m"] is None
    state["object_pose_environment_xyzw"][3:] = [0, 0, math.sqrt(0.5), math.sqrt(0.5)]
    narrow = narrow_task.update(**state, dt=0)
    assert narrow["measurements"]["rotation_safe_xy_margin_m"] == pytest.approx(0.1 - radius)
    assert narrow["drop_ready"]  # Failure to fit the conservative sphere does not forbid the current pose.
    state["object_pose_environment_xyzw"][2] = 0.61
    assert narrow_task.update(**state, dt=1.0)["success"]  # A released, settled current pose still qualifies.
    json.dumps(narrow_task.describe(), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "failed_condition"),
    [
        ("grippers_open", [True, False], "both_grippers_open"),
        # Opposing body forces must not cancel into apparent robot release.
        ("robot_contact_forces_world_n", [[2, 0, 0], [-2, 0, 0]], "robot_released"),
        # A visible thumb/rim contact in the live scene passed the old 0.5 N tolerance.
        ("robot_contact_forces_world_n", [[0, 0, 0.495]], "robot_released"),
        ("bin_contact_force_world_n", None, "bin_support"),
        (
            "object_pose_environment_xyzw",
            [2, -1, 0.62, 0, 0, math.sqrt(0.5), math.sqrt(0.5)],
            "mesh_bottom_near_floor",
        ),
        ("object_velocity_world", [0.04, 0.04, 0, 0, 0, 0], "linear_settled"),
        ("object_velocity_world", [0, 0, 0, 0, 0, 0.3], "angular_settled"),
    ],
)
def test_release_support_and_speed_checks_restart_settling(task_and_state, field, value, failed_condition):
    task, state = task_and_state
    task.update(**state, dt=0.75)
    unsettled = copy.deepcopy(state)
    unsettled[field] = value
    result = task.update(**unsettled, dt=0.5)
    assert not result["conditions"][failed_condition]
    assert not result["success"]
    assert result["settled_sim_seconds"] == 0
    assert not task.update(**state, dt=0.75)["success"]
    assert task.update(**state, dt=0.25)["success"]


def test_unknown_robot_contacts_do_not_establish_release_and_nonfinite_contacts_remain_invalid(task_and_state):
    task, state = task_and_state
    state["robot_contact_forces_world_n"] = None
    result = task.update(**state, dt=1.0)
    assert result["measurements"]["robot_contact_magnitude_sum_n"] is None
    assert result["conditions"]["bin_support"]
    assert not result["conditions"]["robot_released"]
    assert not result["success"]
    json.dumps(result, allow_nan=False)
    state["robot_contact_forces_world_n"] = [[float("nan"), 0, 0]]
    with pytest.raises(ValueError, match="Robot-body contact forces must be finite"):
        task.update(**state, dt=1.0)


def test_observer_hides_stale_contacts_after_reset_until_physics_executes(task_and_state, monkeypatch):
    task, state = task_and_state
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Object")
    mesh = UsdGeom.Mesh.Define(stage, "/Object/mesh")
    mesh.CreatePointsAttr(list(itertools.product((-0.2, 0.2), (-0.1, 0.1), (-0.08, 0.08))))
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 4, 6, 2, 1, 3, 7, 5, 0, 1, 5, 4, 2, 6, 7, 3, 0, 2, 3, 1, 4, 5, 7, 6])
    monkeypatch.setattr("isaaclab.sim.utils.get_current_stage", lambda: stage)
    # PhysX leaves the previous scene's support vector in this buffer after reset.
    normal = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.7]]])
    friction = np.zeros_like(normal)
    contact_view = SimpleNamespace(
        filter_count=3,
        get_normal_contact_force_matrix=lambda dt: SimpleNamespace(numpy=lambda: normal),
        get_friction_contact_force_matrix=lambda dt: SimpleNamespace(numpy=lambda: friction),
    )

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(
        object=SimpleNamespace(
            data=SimpleNamespace(
                root_link_pose_w=SimpleNamespace(torch=torch.tensor([state["object_pose_environment_xyzw"]])),
                root_link_vel_w=SimpleNamespace(torch=torch.zeros(1, 6)),
            )
        ),
        robot=SimpleNamespace(root_view=SimpleNamespace(link_paths=[["/Robot/hand", "/Robot/other"]])),
    )
    env = SimpleNamespace(
        scene=scene,
        num_envs=1,
        physics_dt=0.005,
        sim=SimpleNamespace(
            backend_manager=SimpleNamespace(
                backend_name="physx",
                get_physics_sim_view=lambda: SimpleNamespace(create_rigid_contact_view=lambda *a, **kw: contact_view),
            )
        ),
    )
    container = task.describe()["target_bin"]
    container["prim_path"] = "/Bin"
    perception = SimpleNamespace(
        sensor=SimpleNamespace(body_physx_view=SimpleNamespace(prim_paths=["/Object"])),
        bin_geometry={"source": "authored fixture bin", "containers": [container]},
    )
    original = SimpleNamespace(func=task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})
    observer = BinDropObserver(env, perception, SimpleNamespace(grippers_open=lambda: [True, True]), original)
    for _ in range(2):
        # Covers both construction and reset, including repeated render-only observations.
        for _ in range(2):
            initial = observer.update(0.0)
            assert initial["measurements"]["bin_upward_support_n"] is None
            assert initial["measurements"]["robot_contact_magnitude_sum_n"] is None
            assert initial["robot_contacts"] is None
            assert initial["bin_contact_force_world_n"] is None
            assert not initial["conditions"]["bin_support"]
            assert not initial["conditions"]["robot_released"]
            assert initial["settled_sim_seconds"] == 0
            assert not initial["success"]
        assert not observer.update(0.5)["success"]
        paused = observer.update(0.0)
        assert paused["measurements"]["bin_upward_support_n"] == pytest.approx(5.7)
        assert paused["robot_contacts"] == []
        assert paused["bin_contact_force_world_n"] == [0, 0, 5.7]
        assert paused["settled_sim_seconds"] == 0.5
        assert observer.update(0.5)["success"]
        observer.reset()

    normal[0, 0] = [0, 0.3, 0]
    friction[0, 0] = [0, 0, 0.4]
    normal[0, 1] = [0.0005, 0, 0]
    friction[0, -1] = [0.2, 0, 0]
    contact = observer.update(0.02)
    normal.fill(0)
    friction.fill(0)
    assert contact["robot_contacts_min_magnitude_n"] == 0.001
    assert contact["robot_contacts"] == [{"body": "hand", "force_world_n": [0, 0.3, 0.4]}]
    assert contact["bin_contact_force_world_n"] == [0.2, 0, 5.7]
    assert contact["measurements"]["robot_contact_magnitude_sum_n"] == pytest.approx(0.5005)
    assert not contact["conditions"]["robot_released"]

    normal[0, 1] = [0.0005, 0, 0]
    normal[0, -1] = [0, 0, 5.7]
    below_display_threshold = observer.update(1.0)
    assert below_display_threshold["robot_contacts"] == []
    assert below_display_threshold["measurements"]["robot_contact_magnitude_sum_n"] == pytest.approx(0.0005)
    assert not below_display_threshold["conditions"]["robot_released"]
    assert not below_display_threshold["success"]


def test_drop_guidance_paused_time_reset_and_environment_contract(task_and_state):
    task, state = task_and_state
    raised = copy.deepcopy(state)
    raised["object_pose_environment_xyzw"][2] += 0.15
    raised["grippers_open"] = [False, False]
    raised["bin_contact_force_world_n"] = None
    result = task.update(**raised, dt=0)
    assert result["drop_ready"]
    assert result["measurements"]["floor_gap_m"] == pytest.approx(0.15)
    assert not result["success"]
    raised["object_pose_environment_xyzw"][2] += 0.3
    assert not task.update(**raised, dt=0)["drop_ready"]

    task.update(**state, dt=0.5)
    for _ in range(10):
        paused = task.update(**state, dt=0)
        assert paused["settled_sim_seconds"] == 0.5
        assert not paused["success"]
    task.reset()
    assert not task.update(**state, dt=0.5)["success"]
    assert task.update(**state, dt=0.5)["success"]
    contract = task.describe()
    legacy = contract["legacy_environment_success"]
    assert legacy["resolved_parameters"]["min_x"] == 0.41
    assert legacy["resolved_parameters"]["min_vel"] == 0.2
    assert legacy["task_link_name"] == "right_wrist_yaw_link"
    assert legacy["object_asset"] == "object"
    assert contract["target_bin"]["position_environment_m"] == [2, -1, 0.5]
    json.dumps(contract, allow_nan=False)
    json.dumps(result, allow_nan=False)
