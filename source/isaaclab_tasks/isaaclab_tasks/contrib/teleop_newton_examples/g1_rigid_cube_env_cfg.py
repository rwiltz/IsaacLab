# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 teleoperation pick-and-place on Newton MJWarp, with a rigid cube.

This subclasses the PhysX task :class:`LocomanipulationG1EnvCfg` and changes only what the Newton
backend requires, so the diff between the two is the porting work itself. Each override below is
one thing Newton needs that PhysX did not. Upper-body control stays on Pink IK.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
)

##
# 1. Solver profile
##

NEWTON_PHYSICS_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        # The contact budget has to cover two feet on the ground plus a two-handed grasp.
        njmax=400,
        nconmax=250,
        impratio=1.0,
        cone="pyramidal",
        update_data_interval=2,
        iterations=100,
        ls_iterations=15,
        ls_parallel=False,
        use_mujoco_contacts=False,
    ),
    collision_cfg=NewtonCollisionPipelineCfg(),
    # Newton's default contact stiffness is ``ke=2.5e3``. A humanoid's weight on two feet sinks
    # visibly into that: the ankles rest about 2 cm low and the locomotion policy never settles.
    default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    num_substeps=2,
)
"""MJWarp profile for a walking humanoid. Gravity stays enabled: the policy needs ground contact."""

##
# 2. Contact material
##

ROBOT_CONTACT_MATERIAL = [
    UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
]
"""Newton resolves an omitted friction value to zero, so an unauthored hand grips nothing."""

##
# 3. Graspable object
##

CUBE_SIZE = 0.035
"""Edge length of the graspable cube [m]."""

CUBE_MASS = 0.05
"""Mass of the graspable cube [kg].

About 1200 kg/m^3 at this size, so a dense plastic. The ``Isaac-Lift-Franka`` primitive this is
adapted from carries 0.2 kg, which at 3.5 cm implies roughly titanium and takes correspondingly
more grip force to hold.
"""


def cube_spawn() -> sim_utils.CuboidCfg:
    """Build the graspable cube.

    The task's authored steering wheel cannot be used here: its rim is a torus, and MuJoCo has no
    concave mesh-mesh collision, so imported as a mesh the hand passes through the rim while a
    convex hull fills the ring in and turns the wheel into a solid disc.
    """
    return sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        # Matches the robot's hand material. At 0.5 the cube slips out of the grasp.
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
    )


##
# 4. Table collider
##

TABLE_COLLIDER_POS = (0.0, 0.55, 0.19705)
"""World position of the tabletop collider proxy [m]."""

TABLE_COLLIDER_SIZE = (2.4736, 0.762, 0.9941)
"""Bounds of the tabletop collider proxy [m], matching the asset's authored collider."""

##
# 5. Joint ordering
##

LOWER_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
"""Legs the locomotion policy drives, in the order it was trained on.

The task declares these as regexes, which resolve in articulation order. PhysX and Newton do not
agree on that order, so the policy's outputs land on the wrong joints under Newton and the robot
falls over. Listing them explicitly, with ``preserve_order=True`` in the action term, pins the
order on both backends.
"""


@configclass
class G1RigidCubeSceneCfg(LocomanipulationG1SceneCfg):
    """Scene with a graspable cube and the tabletop collider Newton needs."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[0, 0, 0, 1]),
        spawn=cube_spawn(),
    )

    # ``packing_table.usd`` authors its collider as a ``boundingCube`` ``PhysicsCollisionAPI`` on
    # an Xform rather than on mesh prims. Newton emits no shape for it, so anything resting on the
    # table falls through. This invisible box reproduces the same bounding volume.
    packing_table_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTableCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(TABLE_COLLIDER_POS)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_COLLIDER_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class G1RigidCubeEnvCfg(LocomanipulationG1EnvCfg):
    """G1 teleoperation pick-and-place with a rigid cube, on Newton MJWarp."""

    scene: G1RigidCubeSceneCfg = G1RigidCubeSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()

        # 1. Select the Newton backend and its solver profile.
        self.sim.physics = NEWTON_PHYSICS_CFG

        # 2. Author a contact material on the robot.
        robot_spawn = self.scene.robot.spawn.copy()
        robot_spawn.physics_material = ROBOT_CONTACT_MATERIAL
        self.scene.robot.spawn = robot_spawn

        # 3. Point the contact sensors at Newton's body labels, which keep the asset's
        #    intermediate grouping prim (``/Robot/left_hand/left_hand_index_0_link``). The task's
        #    PhysX pattern stops at ``/Robot/`` and ``[^/]*`` cannot cross a path separator, so it
        #    full-matches nothing under Newton and sensor initialization fails.
        #    A coupled solver cannot provide contact sensors, so the deformable variant removes
        #    them and there is nothing to repoint.
        for side in ("left", "right"):
            sensor = getattr(self.scene, f"{side}_hand_contact")
            if sensor is not None:
                sensor.prim_path = "{ENV_REGEX_NS}/Robot/" + f"{side}_hand/{side}_hand_[^/]*_link"

        # 4. Pin the locomotion policy's joint order.
        self.actions.lower_body_joint_pos.joint_names = LOWER_BODY_JOINT_NAMES
