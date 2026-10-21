# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check authored bin collision geometry without starting the simulator."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/environments/teleoperation"))
from astra_teleop.perception import ObjectPerception, extract_bin_geometry  # noqa: E402


def test_bin_interior_uses_collision_shapes_scale_pose_and_environment_origin():
    stage = Usd.Stage.CreateInMemory()
    background = UsdGeom.Xform.Define(stage, "/World/PackingTable")
    background.AddTranslateOp().Set((10, 20, 0.5))
    background.AddRotateZOp().Set(90)
    background.AddScaleOp().Set((2, 1, 1))
    container = UsdGeom.Xform.Define(stage, "/World/PackingTable/container_01")
    container.AddTranslateOp().Set((0.3, 0.4, 0))
    UsdPhysics.RigidBodyAPI.Apply(container.GetPrim())
    specs = {
        "bottom": ((0, 0, 0), (1, 0.6, 0.04)),
        "left": ((-0.5, 0, 0.1), (0.04, 0.6, 0.2)),
        "right": ((0.5, 0, 0.1), (0.04, 0.6, 0.2)),
        "front": ((0, -0.3, 0.1), (1, 0.04, 0.2)),
        "back": ((0, 0.3, 0.1), (1, 0.04, 0.2)),
        "disabled": ((0, 0, 0), (50, 50, 50)),
    }
    for name, (position, size) in specs.items():
        cube = UsdGeom.Cube.Define(stage, f"{container.GetPath()}/{name}")
        cube.CreateSizeAttr(1)
        cube.AddTranslateOp().Set(position)
        cube.AddScaleOp().Set(size)
        collider = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collider.CreateCollisionEnabledAttr(name != "disabled")

    geometry = extract_bin_geometry(stage, str(background.GetPath()), np.array([10, 20, 0]))
    result = geometry["containers"][0]
    assert result["name"] == "container_01"
    assert len(result["colliders"]) == 5
    assert result["position_environment_m"] == pytest.approx([-0.4, 0.6, 0.5])
    assert result["quaternion_xyzw"] == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    assert np.asarray(result["interior_bounds_container_m"]) == pytest.approx(
        np.array([[-0.96, -0.28, 0.02], [0.96, 0.28, 0.2]])
    )
    assert np.asarray(result["interior_aabb_environment_m"]) == pytest.approx(
        np.array([[-0.68, -0.36, 0.52], [-0.12, 1.56, 0.7]])
    )
    floor = next(collider for collider in result["colliders"] if collider["name"] == "bottom")
    assert np.asarray(floor["bounds_environment_m"]) == pytest.approx(np.array([[-0.7, -0.4, 0.48], [-0.1, 1.6, 0.52]]))

    # An incomplete collider layout must not invent a closed interior.
    stage.RemovePrim(f"{container.GetPath()}/back")
    incomplete = extract_bin_geometry(stage, str(background.GetPath()), np.array([10, 20, 0]))
    assert "interior_bounds_container_m" not in incomplete["containers"][0]
    assert len(incomplete["containers"][0]["colliders"]) == 4


def test_nonfinger_force_is_vector_residual_and_missing_contact_positions_are_null(monkeypatch):
    import isaaclab.sim.utils as sim_utils

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Object")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Object/mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)
    forces = torch.zeros(1, 1, 14, 3)
    forces[0, 0, 0] = torch.tensor([3, 0, 4])
    forces[0, 0, 10] = torch.tensor([-1, 2, -1])
    total = torch.tensor([[[2.0, 3.0, 8.0]]])
    positions = torch.full((1, 1, 14, 3), float("nan"))
    positions[0, 0, 0] = torch.tensor([10.2, 20.3, 1.0])
    sensor = SimpleNamespace(
        num_sensors=1,
        body_physx_view=SimpleNamespace(prim_paths=["/World/Object"]),
        data=SimpleNamespace(
            force_matrix_w=SimpleNamespace(torch=forces),
            net_forces_w=SimpleNamespace(torch=total),
            contact_pos_w=SimpleNamespace(torch=positions),
        ),
    )
    scene = SimpleNamespace(
        sensors={"object_hand_contact": sensor}, env_origins=torch.tensor([[10.0, 20.0, 0.0]]), keys=lambda: []
    )
    perception = ObjectPerception(SimpleNamespace(num_envs=1, scene=scene))
    snapshot = perception.snapshot()
    forces.fill_(99)
    total.fill_(99)
    positions.fill_(99)
    assert snapshot["object_total_contact_force_world_n"] == [2, 3, 8]
    assert snapshot["object_nonfinger_contact_force_world_n"] == [0, 1, 5]
    assert snapshot["object_contact_positions_environment_m"]["left_hand_thumb_0_link"] == pytest.approx(
        [0.2, 0.3, 1], abs=1e-6
    )
    assert snapshot["object_contact_positions_environment_m"]["right_hand_index_0_link"] is None
    json.dumps(snapshot, allow_nan=False)
    total[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="forces must be finite"):
        perception.snapshot()
