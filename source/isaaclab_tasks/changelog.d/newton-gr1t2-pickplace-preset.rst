Added
^^^^^

* Added a physics preset to the ``IsaacContrib-PickPlace-GR1T2-Abs`` task so it can run on the
  Newton MJWarp backend via ``physics=newton_mjwarp``. Previously the task never assigned
  ``sim.physics`` and ``physics=newton_mjwarp`` failed with ``Unknown preset(s)``. The ``default``
  preset keeps the bare :class:`~isaaclab_physx.physics.PhysxCfg` the task used before, so PhysX
  behavior is unchanged.
* Added a Newton-only spawner for the steering-wheel object. Its collision meshes are authored as
  ``convexDecomposition`` and several decompose into degenerate slivers, which made the MJWarp
  model fail to compile with ``mesh volume is too small``. Under the ``newton_mjwarp`` preset every
  collision mesh except the wheel rim and spokes is approximated with a single convex hull; those
  two keep their decomposition so the wheel stays graspable.
