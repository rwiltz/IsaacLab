Added
^^^^^

* Added ``isaaclab_tasks.contrib.teleop_newton_examples``, two worked examples of teleoperating a
  Unitree G1 through Isaac Teleop against the Newton backend:
  ``IsaacContrib-TeleopNewton-G1-RigidCube-Abs`` on MJWarp, and
  ``IsaacContrib-TeleopNewton-G1-DeformableCube-Abs`` which adds a VBD solver coupled to MJWarp
  for a deformable object. Both keep Pink IK on the upper body. Each subclasses the existing PhysX
  locomanipulation task and overrides only what the backend requires, so the diff is the porting
  work itself.

Fixed
^^^^^

* Fixed the locomanipulation lower-body policy receiving joints in articulation order. The
  observation and action terms selected joints by regex, which resolves in articulation order, and
  the backends do not agree on it: PhysX enumerates breadth-first by tree depth while Newton
  enumerates each limb chain depth-first. Under Newton the policy read a permuted observation and
  its outputs were written to the wrong joints, and the robot fell over. The joints are now listed
  explicitly in the trained order and resolved with ``preserve_order=True``, which reproduces the
  PhysX indices exactly and leaves PhysX behavior unchanged.

* Fixed the lower-body action term running its frozen locomotion policy without ``torch.no_grad()``.
  Newton writes joint targets through Warp kernels, which reject a tensor that requires grad, so
  the first ``env.step`` raised under Newton. PhysX writes through torch and was unaffected.
