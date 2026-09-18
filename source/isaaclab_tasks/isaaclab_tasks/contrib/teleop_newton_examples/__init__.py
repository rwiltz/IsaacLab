# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Worked examples of running Isaac Teleop against the Newton physics backend.

Two tasks, both teleoperating a Unitree G1 through Isaac Teleop with Pink IK on the upper body:

* ``IsaacContrib-TeleopNewton-G1-RigidCube-Abs`` -- a rigid cube on Newton MJWarp.
* ``IsaacContrib-TeleopNewton-G1-DeformableCube-Abs`` -- a deformable cube, adding a VBD solver
  coupled to MJWarp.

See ``TELEOP_NEWTON_EXAMPLES.md`` at the repository root for the porting guide these illustrate.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-TeleopNewton-G1-RigidCube-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_rigid_cube_env_cfg:G1RigidCubeEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="IsaacContrib-TeleopNewton-G1-DeformableCube-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_deformable_cube_env_cfg:G1DeformableCubeEnvCfg",
    },
    disable_env_checker=True,
)
