Changed
^^^^^^^

* **Breaking:** ``isaaclab_tasks`` now depends on ``isaaclab_teleop``. The ``contrib`` place/stack
  environment configs that default a keyboard ``teleop_devices`` entry now import
  :class:`~isaaclab_teleop.Se3KeyboardCfg` instead of the deprecated
  :class:`~isaaclab.devices.Se3KeyboardCfg`. Installing ``isaaclab_tasks`` now requires the
  ``teleop`` extra (or a standalone ``isaaclab_teleop`` install) to use these environments.
* The Franka, UR10, Galbot, and Agibot relative-mode stack/place environment configs now declare
  ``self.isaac_teleop`` (an ``IsaacTeleopCfg`` with a keyboard-driven ``pipeline_builder``), so
  keyboard teleoperation runs through the IsaacTeleop session/pipeline by default, matching the
  G1 / GR1T2 environments, instead of only through the ``Se3KeyboardCfg`` device-factory shim.
  The ``teleop_devices`` keyboard entry is kept for scripts that pass ``--teleop_device`` to opt
  out explicitly.
