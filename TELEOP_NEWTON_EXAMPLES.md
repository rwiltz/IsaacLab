# Isaac Teleop on the Newton Backend: Worked Examples

This branch adds two teleoperation tasks that run against Newton instead of PhysX, to serve as a
reference for bringing your own embodiment and task to the same stack. It is based on
`release/3.0.0`.

Both tasks teleoperate a Unitree G1 through Isaac Teleop over CloudXR, with Pink IK on the upper
body and a pretrained locomotion policy on the lower body:

| Task | Physics | Object |
| --- | --- | --- |
| `IsaacContrib-TeleopNewton-G1-RigidCube-Abs` | Newton MJWarp | rigid cube |
| `IsaacContrib-TeleopNewton-G1-DeformableCube-Abs` | Newton MJWarp + VBD | deformable cube |

Source: `source/isaaclab_tasks/isaaclab_tasks/contrib/teleop_newton_examples/`

## Running them

Rigid:

```bash
ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES \
uv run --extra teleop isaaclab -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task IsaacContrib-TeleopNewton-G1-RigidCube-Abs \
  --device cuda:0 --xr
```

Deformable (needs the `tetrahedralization` extra for the volume mesh):

```bash
ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=0 OMNI_KIT_ACCEPT_EULA=YES \
uv run --extra teleop --extra tetrahedralization isaaclab -p \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task IsaacContrib-TeleopNewton-G1-DeformableCube-Abs \
  --device cuda:0 --xr
```

Add `--viz kit` for a desktop mirror of the scene, and `--disable_external_cameras` to drop the
robot point-of-view camera and its XR picture-in-picture if you want the frame time back.

Do not pass `--teleop_device`: these tasks define no `teleop_devices` entry, and omitting the flag
selects the IsaacTeleop OpenXR pipeline they expect.

## How the examples are structured

The point of the layout is that **the diff is the lesson**.

`g1_rigid_cube_env_cfg.py` subclasses the existing PhysX task `LocomanipulationG1EnvCfg` and
overrides only what Newton requires. Read its `__post_init__` top to bottom and you have the whole
rigid-body porting story in five steps.

`g1_deformable_cube_env_cfg.py` then subclasses *that*, so its diff isolates one further question:
what changes when the manipulated object stops being rigid.

## Porting your own task: the checklist

Ordered by how early each one bites. The first four will stop the task loading or make the robot
visibly fail; the rest are quality.

### 1. Joint ordering, everywhere ordered data crosses a boundary

**This is the single highest-value item on the list, and the least obvious.**

`find_joints()` returns ids in *articulation order*, and the two backends do not agree on what that
order is. Measured on the G1:

- **PhysX** enumerates breadth-first by tree depth:
  `left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, ...`
- **Newton** enumerates each limb chain depth-first:
  `left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, ...`

Anything that selects joints **by regex** therefore silently permutes when you switch to Newton. A
pretrained policy reads a scrambled observation and writes to the wrong joints; a teleop retargeter
drives the wrong fingers. PhysX happens to order these robots the way the configs list them, so
none of this is visible until the backend changes.

`preserve_order=True` **alone is not enough**: it reorders to *config pattern* order, which matches
neither backend. The fix is an explicit joint-name list in the trained/declared order **plus**
`preserve_order=True`.

Recover the correct order by resolving the original regex list **under PhysX** and printing the
names. That is the order the consumer expects, because PhysX is the backend it currently works on.
Writing that list back makes the resolved indices identical under PhysX, so the change is provably
a no-op there while fixing Newton.

This appeared in four separate places on this branch. Audit all of them in your own task:

- the policy's **observation** terms (`SceneEntityCfg(..., joint_names=[...])`)
- the policy's **action** term
- the **IK action term's hand joints**
- any other term carrying ordered joint data

Symptom if you miss it: the robot falls over immediately (policy), or the grip does nothing
(hands), with no error and entirely finite state.

### 2. Friction defaults to zero

Newton resolves an omitted friction value to **zero**. An unauthored hand grips nothing. Author a
material explicitly:

```python
ROBOT_CONTACT_MATERIAL = [
    UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
]
```

### 3. Collision geometry that PhysX accepted and Newton does not

Two traps in the stock assets, both of which produce a scene that loads and then behaves wrongly:

- **Colliders authored on Xforms are dropped.** `packing_table.usd` declares its tabletop collider
  as a `boundingCube` `PhysicsCollisionAPI` on an Xform rather than on mesh prims. Newton emits no
  shape at all, and objects fall through the table. The examples add an invisible box reproducing
  the same bounding volume.
- **MuJoCo has no concave mesh-mesh collision.** The task's original steering wheel has a torus
  rim, which is not graspable under MJWarp however it is approximated: imported as a mesh the hand
  passes through the rim; collapsed to a convex hull the ring fills in and it becomes a solid disc.
  The examples substitute a cube. If your task depends on a concave graspable, plan for a convex
  decomposition rather than expecting the asset to port.

### 4. Contact sensors match Newton's body labels

Newton's labels keep the asset's intermediate grouping prim
(`/Robot/left_hand/left_hand_index_0_link`). A PhysX-shaped pattern that stops at `/Robot/` full
matches nothing, because `[^/]*` cannot cross a path separator, and sensor initialization fails
outright. Always check a sensor pattern against the model's actual `body_label` values.

### 5. Solver profile

Start from the locomotion profile rather than a fixed-base manipulation one if your robot walks.
The contact budget has to cover two feet on the ground plus the grasp, and contact stiffness
matters more than it does for an arm: Newton's default `ke=2.5e3` lets a humanoid's weight sink its
ankles about 2 cm into the floor, and the locomotion policy then never settles. The examples use
`NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0)`.

Gravity stays **enabled**: the policy needs real ground contact, so the gravity-compensation trick
used for fixed-base humanoids does not apply.

Tune substeps by measurement, not intuition. Raising `num_substeps` from 2 to 4 on this task made
the robot launch (peak root height 0.965 m against 0.815 m at 2 substeps).

## What the deformable example adds

MJWarp cannot simulate a deformable at all: deformable support is installed by the VBD manager, and
the MJWarp manager sets `soft_contact_max=0`. You cannot get there by swapping the asset.

Instead, Newton partitions **one** model between named solver entries, joined by
`CouplerProxyCfg`: a `rigid` entry where MJWarp owns the robot, and a `soft` entry where VBD owns
the particles, with only the grasping links exposed as proxies.

Constraints worth knowing before you design around it:

- **Contact sensors are rejected outright** under a coupled solver, because contact forces live in
  per-entry buffers. The deformable example therefore has no haptics.
- **A shape belongs to at most one entry.** `include_static_shapes=True` hands *every* static shape
  to one entry. On a fixed-base arm that is harmless; on a walking robot it takes the ground plane
  away from MJWarp and the feet sink through the floor. Assign the ground and the support surface
  individually by shape label. The example carries two coincident tabletop colliders, one per
  entry, because the two solvers cannot share one.
- **Proxy coupling supports at most two entries**, each articulation must stay in one entry, and
  nested couplers are rejected.
- **A soft body spawned intersecting a collider is ejected**, where a rigid body is merely pushed
  out. Compute the support surface's top face and clear it rather than inheriting the rigid task's
  start height.
- **Full-surface rigid-soft contact needs a signed-distance field per rigid shape.** The G1's hand
  colliders are meshes without one, so the example falls back to per-vertex contacts.

### Tuning a deformable grasp

Do not inherit the `Isaac-Lift-Soft-Franka` contact values wholesale. That task is a slim two-finger
gripper on a bolted-down arm; a humanoid palm with a free-standing base is a different problem. Its
preset is 8x stiffer, 1000x less damped, and uses a friction the documentation itself calls
unphysical, all of which a fixed mount absorbs and a walking robot does not.

The decisive parameter turned out to be **solver convergence, not contact stiffness**. Measured by
releasing the object at a fixed 5 mm overlap with a hand link:

| VBD iterations | peak object speed | peak robot joint speed |
| --- | --- | --- |
| 10 (the Franka value) | 28.6 m/s | 224 rad/s |
| **40** | **2.5 m/s** | **3.5 rad/s** |
| 100 | 2.6 m/s | 4.4 rad/s |

A G1 palm puts far more particles in contact at once than a Franka fingertip, and the solve does
not converge in 10 iterations. The stored energy then fires the object out of the hand. If you see
an object behave "springy" or gain energy, check convergence before you touch stiffness: an
under-converged solve also responds non-monotonically to stiffness, which makes tuning it
actively misleading.

## Changes outside the tasks

Five fixes were needed in the core packages. Each is a genuine Newton-compatibility bug in
`release/3.0.0`, and each is small and self-contained:

| Package | Fix | Symptom without it |
| --- | --- | --- |
| `isaaclab` | Pink IK resolves hand joints with `preserve_order` | grip does nothing |
| `isaaclab_newton` | contact sensors request their `force` attribute | raises on first step |
| `isaaclab_newton` | actuator properties re-applied after a hard reset | replay goes non-finite |
| `isaaclab_contrib` | coupler imports the MPM manager lazily | coupled scenes fail under Kit |
| `isaaclab_tasks` | locomotion policy joint ordering, and `no_grad` | robot falls over; raises |

Two are worth expanding on, because they are the kind of thing that costs days:

**Actuator properties do not survive a hard reset.** `sim.reset(soft=False)` re-finalizes the
Newton model from the builder, and the rebuilt model carries the USD-authored joint drives, so
everything `ArticulationCfg.actuators` configured is silently discarded. Measured: stiffness
4400 to 53026, damping 40 to 2148, armature 0.1 to 0.0, effort limit to `inf`. MJWarp cannot
integrate those, so the **first commanded motion** after the reset drives the articulation
non-finite, while a zero command still looks perfectly healthy. Live teleop never hits this because
it only calls `env.reset()`; record and replay does.

**Warp kernels reject tensors that require grad.** A frozen policy run without `torch.no_grad()`
raises `Can't get __cuda_array_interface__ on Variable that requires grad` on the first step. PhysX
writes through torch and never notices.

## Known limitations

- The deformable task runs and is stable, but grasping it is not yet as good as the rigid case.
  Contact tuning there is the open work.
- The deformable task has no haptics, since coupled solvers reject contact sensors.
- Coupled solvers are flagged experimental upstream.

## Notes on working with this branch

- It is based on `release/3.0.0`, so the changelog pre-commit hook needs the matching base to
  report the right packages: `ISAACLAB_CHANGELOG_BASE_REF=release/3.0.0 uv run isaaclab -f`.
  Without it the hook diffs against `develop` and flags packages you never touched.
- Isaac Lab auto-launches and manages the CloudXR runtime on every `--xr` run, including the state
  under `~/.cloudxr/run/`. Do not hand-manage it or start a second CloudXR container: a second one
  competes for the same IPC socket and host ports. If a client cannot connect, stop the Isaac Lab
  session and relaunch it.
