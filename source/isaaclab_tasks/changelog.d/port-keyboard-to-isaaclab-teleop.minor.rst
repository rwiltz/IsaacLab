Changed
^^^^^^^

* **Breaking:** ``isaaclab_tasks`` now depends on ``isaaclab_teleop``. Installing ``isaaclab_tasks``
  now requires the ``teleop`` extra (or a standalone ``isaaclab_teleop`` install) to use
  environments with a keyboard-driven teleoperation pipeline.
* The Franka, UR10, Galbot, and Agibot relative-mode stack/place environment configs now declare
  ``self.isaac_teleop`` (an ``IsaacTeleopCfg`` built by
  :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`), so keyboard teleoperation runs through the
  IsaacTeleop session/pipeline by default, matching the G1 / GR1T2 environments, instead of
  through the removed ``Se3KeyboardCfg`` device-factory shim.
