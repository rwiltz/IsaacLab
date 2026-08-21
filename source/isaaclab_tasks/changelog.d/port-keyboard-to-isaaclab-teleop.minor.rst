Changed
^^^^^^^

* **Breaking:** ``isaaclab_tasks`` now depends on ``isaaclab_teleop``. The ``contrib`` place/stack
  environment configs that default a keyboard ``teleop_devices`` entry now import
  :class:`~isaaclab_teleop.Se3KeyboardCfg` instead of the deprecated
  :class:`~isaaclab.devices.Se3KeyboardCfg`. Installing ``isaaclab_tasks`` now requires the
  ``teleop`` extra (or a standalone ``isaaclab_teleop`` install) to use these environments.
