Changed
^^^^^^^

* **Breaking:** The ``contrib`` place/stack environment configs that default a spacemouse
  ``teleop_devices`` entry now import :class:`~isaaclab_teleop.Se3SpaceMouseCfg` instead of the
  deprecated :class:`~isaaclab.devices.Se3SpaceMouseCfg`. Installing ``isaaclab_tasks`` now
  requires the ``teleop`` extra (or a standalone ``isaaclab_teleop`` install) to use these
  environments.
