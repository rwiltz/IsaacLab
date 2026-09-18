# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 teleoperation pick-and-place on Newton, with a deformable cube.

This builds on :class:`G1RigidCubeEnvCfg` and changes only what a deformable object requires, so
the diff between the two shows the rigid-to-deformable work on its own.

MJWarp cannot simulate a deformable at all: deformable support is installed by the VBD manager,
and the MJWarp manager sets ``soft_contact_max=0``. The scene is therefore partitioned between two
solvers that share one Newton model, joined by
:class:`~isaaclab_contrib.coupling.CouplerProxyCfg`:

* a ``rigid`` entry where MJWarp owns the whole G1 articulation, and
* a ``soft`` entry where VBD owns the object's particles,

with only the hand links exposed as proxies, so VBD sees the grasping geometry and nothing else.
Upper-body control stays on Pink IK.

Requires the ``tetrahedralization`` extra for the volume mesh.
"""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from .g1_rigid_cube_env_cfg import (
    TABLE_COLLIDER_POS,
    TABLE_COLLIDER_SIZE,
    G1RigidCubeEnvCfg,
    G1RigidCubeSceneCfg,
)

##
# Object
##

CUBE_SIZE = (0.06, 0.06, 0.06)
"""Edge lengths of the deformable cube [m]."""

CUBE_POS = (-0.35, 0.45, 0.7261)
"""Start position of the deformable cube [m].

The tabletop collider's top face is at z = 0.6941, so this clears it by 2 mm. A soft body spawned
intersecting a collider is ejected, where a rigid body is merely pushed out.
"""

YOUNGS_MODULUS = 5.0e4
"""Young's modulus of the cube [Pa]. A stiffer cube barely deforms around a fingertip."""

POISSONS_RATIO = 0.3
"""Poisson's ratio of the cube."""

MATERIAL_DAMPING = 5.0e1
"""Internal damping of the cube [Pa s]. Newton defaults this to zero, a perfectly elastic solid."""

##
# Solver partition
##

HAND_PROXY_BODIES = [r"/World/envs/env_[^/]+/Robot/(left|right)_hand/.*_link"]
"""Robot bodies exposed to the deformable solver. Keep this to the grasping links only."""

GROUND_SHAPE = r"/World/GroundPlane.*"
"""Static shape owned by the rigid entry, so the robot keeps its ground contact."""

TABLE_SHAPE_RIGID = r"/World/envs/env_[^/]+/PackingTableCollider.*"
"""Tabletop owned by the rigid entry, so the robot cannot reach through the table."""

TABLE_SHAPE_SOFT = r"/World/envs/env_[^/]+/SoftPackingTableCollider.*"
"""Tabletop owned by the soft entry, so the object rests on the table.

A shape belongs to at most one entry, so the two solvers cannot share one tabletop. This is a
second collider coincident with the first. Note that ``include_static_shapes=True`` would instead
hand *every* static shape to one entry: on a walking robot that takes the ground plane away from
MJWarp and the feet sink through the floor.
"""

NEWTON_PHYSICS_CFG = NewtonCfg(
    solver_cfg=CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                shape_label_patterns=[GROUND_SHAPE, TABLE_SHAPE_RIGID],
                solver_cfg=MJWarpSolverCfg(
                    solver="newton",
                    integrator="implicitfast",
                    njmax=600,
                    nconmax=400,
                    impratio=1.0,
                    cone="pyramidal",
                    update_data_interval=2,
                    iterations=100,
                    ls_iterations=15,
                    ls_parallel=False,
                    use_mujoco_contacts=False,
                ),
                bodies=[r"/World/envs/env_[^/]+/Robot"],
            ),
            CouplerEntryCfg(
                name="soft",
                # 40, not the 10 the Franka soft-lift task uses. That gripper is slim and its beam
                # small; a G1 palm puts far more particles in contact at once and the solve does
                # not converge in 10, which shows up as stored energy firing the object out of the
                # hand. Measured on a 5 mm overlap release, raising this from 10 to 40 drops peak
                # object speed from 28.6 to 2.5 m/s and peak robot joint speed from 224 to 3.5.
                solver_cfg=VBDSolverCfg(iterations=40, rigid_body_particle_contact_buffer_size=256),
                all_particles=True,
                shape_label_patterns=[TABLE_SHAPE_SOFT],
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="soft",
                bodies=HAND_PROXY_BODIES,
                collide_interval=1,
                # Full-surface contact samples each rigid shape's signed-distance field, and the
                # G1's hand colliders are meshes without one, so contact falls back to per-vertex.
                collision_pipeline=NewtonCollisionPipelineCfg(
                    enable_rigid_soft_full_surface_contact=False,
                ),
            )
        ],
        iterations=1,
    ),
    # Newton's defaults. The Franka soft-lift preset is 8x stiffer, 1000x less damped and uses a
    # friction the documentation itself calls unphysical. A bolted-down arm absorbs the resulting
    # impulse; a free-standing humanoid is thrown by it.
    soft_contact_cfg=NewtonSoftContactCfg(
        soft_contact_ke=1.0e3,
        soft_contact_kd=1.0e1,
        soft_contact_mu=3.0,
    ),
    default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    num_substeps=2,
)
"""Two-solver profile: MJWarp for the robot, VBD for the object."""


@configclass
class G1DeformableCubeSceneCfg(G1RigidCubeSceneCfg):
    """Scene with a deformable object and a second, soft-owned tabletop."""

    object: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=CUBE_POS),
        spawn=sim_utils.MeshCuboidCfg(
            size=CUBE_SIZE,
            edge_refinement=3.0,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=1000.0,
                k_mu=YOUNGS_MODULUS / (2.0 * (1.0 + POISSONS_RATIO)),
                k_lambda=(YOUNGS_MODULUS * POISSONS_RATIO / ((1.0 + POISSONS_RATIO) * (1.0 - 2.0 * POISSONS_RATIO))),
                k_damp=MATERIAL_DAMPING,
                particle_radius=0.004,
            ),
        ),
    )

    soft_packing_table_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SoftPackingTableCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(TABLE_COLLIDER_POS)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_COLLIDER_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Coupled solvers keep contact forces in per-entry buffers and reject contact sensors outright,
    # so the per-hand sensors and the controller haptics they drive are unavailable here.
    left_hand_contact = None
    right_hand_contact = None


@configclass
class G1DeformableCubeEnvCfg(G1RigidCubeEnvCfg):
    """G1 teleoperation pick-and-place with a deformable cube, on Newton MJWarp and VBD."""

    scene: G1DeformableCubeSceneCfg = G1DeformableCubeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()

        # Replace the single-solver profile with the coupled one.
        self.sim.physics = NEWTON_PHYSICS_CFG

        # A deformable has no single rigid pose: ``root_pos_w`` is the mean of the nodal positions
        # and there is no ``root_quat_w``, so the orientation-dependent terms cannot be computed.
        self.observations.policy.object_rot = None
        self.observations.policy.object = None

        # Haptics read the per-hand contact sensors, which a coupled solver cannot provide.
        self.scene.robot.spawn.activate_contact_sensors = False
        self.haptic_feedback = None
