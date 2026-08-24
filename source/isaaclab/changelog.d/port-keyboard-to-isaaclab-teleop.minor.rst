Removed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.devices.Se2Keyboard`, :class:`~isaaclab.devices.Se2KeyboardCfg`,
  :class:`~isaaclab.devices.Se3Keyboard`, and :class:`~isaaclab.devices.Se3KeyboardCfg` (the
  :mod:`isaaclab.devices.keyboard` module). Keyboard teleoperation is now provided by
  :class:`~isaaclab_teleop.Se2Keyboard`, :class:`~isaaclab_teleop.Se2KeyboardCfg`,
  :class:`~isaaclab_teleop.Se3Keyboard`, and :class:`~isaaclab_teleop.Se3KeyboardCfg`, built on
  the IsaacTeleop session API. Update imports from ``isaaclab.devices`` /
  ``isaaclab.devices.keyboard`` to ``isaaclab_teleop.keyboard`` (or ``isaaclab_teleop`` directly),
  and install the ``teleop`` extra (or a standalone ``isaaclab_teleop`` install).
