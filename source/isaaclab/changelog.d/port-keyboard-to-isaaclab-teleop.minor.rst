Removed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.devices.Se2Keyboard`, :class:`~isaaclab.devices.Se2KeyboardCfg`,
  :class:`~isaaclab.devices.Se3Keyboard`, and :class:`~isaaclab.devices.Se3KeyboardCfg` (the
  :mod:`isaaclab.devices.keyboard` module). Keyboard teleoperation is now provided by
  :func:`~isaaclab_teleop.se2_keyboard_teleop_cfg` and :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`,
  which build an :class:`~isaaclab_teleop.IsaacTeleopCfg` for a keyboard-plugin-backed
  :class:`~isaaclab_teleop.IsaacTeleopDevice`. Update imports from ``isaaclab.devices`` /
  ``isaaclab.devices.keyboard`` to ``isaaclab_teleop.keyboard`` (or ``isaaclab_teleop`` directly),
  and install the ``teleop`` extra (or a standalone ``isaaclab_teleop`` install).
