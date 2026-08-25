Removed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.devices.Se2Gamepad`, :class:`~isaaclab.devices.Se2GamepadCfg`,
  :class:`~isaaclab.devices.Se3Gamepad`, :class:`~isaaclab.devices.Se3GamepadCfg` (the
  :mod:`isaaclab.devices.gamepad` module), :class:`~isaaclab.devices.Se2SpaceMouse`,
  :class:`~isaaclab.devices.Se2SpaceMouseCfg`, :class:`~isaaclab.devices.Se3SpaceMouse`, and
  :class:`~isaaclab.devices.Se3SpaceMouseCfg` (the :mod:`isaaclab.devices.spacemouse` module).
  Gamepad and spacemouse teleoperation are now provided by
  :func:`~isaaclab_teleop.se2_gamepad_teleop_cfg`, :func:`~isaaclab_teleop.se3_gamepad_teleop_cfg`,
  :func:`~isaaclab_teleop.se2_spacemouse_teleop_cfg`, and
  :func:`~isaaclab_teleop.se3_spacemouse_teleop_cfg`, which build an
  :class:`~isaaclab_teleop.IsaacTeleopCfg` for a gamepad- or spacemouse-plugin-backed
  :class:`~isaaclab_teleop.IsaacTeleopDevice`. Update imports from ``isaaclab.devices`` /
  ``isaaclab.devices.gamepad`` / ``isaaclab.devices.spacemouse`` to ``isaaclab_teleop.gamepad`` /
  ``isaaclab_teleop.spacemouse`` (or ``isaaclab_teleop`` directly), and install the ``teleop``
  extra (or a standalone ``isaaclab_teleop`` install).
