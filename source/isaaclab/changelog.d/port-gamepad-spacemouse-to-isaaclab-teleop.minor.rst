Removed
^^^^^^^

* **Breaking:** Removed :class:`~isaaclab.devices.Se2Gamepad`, :class:`~isaaclab.devices.Se2GamepadCfg`,
  :class:`~isaaclab.devices.Se3Gamepad`, :class:`~isaaclab.devices.Se3GamepadCfg` (the
  :mod:`isaaclab.devices.gamepad` module), :class:`~isaaclab.devices.Se2SpaceMouse`,
  :class:`~isaaclab.devices.Se2SpaceMouseCfg`, :class:`~isaaclab.devices.Se3SpaceMouse`, and
  :class:`~isaaclab.devices.Se3SpaceMouseCfg` (the :mod:`isaaclab.devices.spacemouse` module).
  Gamepad and spacemouse teleoperation are now provided by :class:`~isaaclab_teleop.Se2Gamepad`,
  :class:`~isaaclab_teleop.Se2GamepadCfg`, :class:`~isaaclab_teleop.Se3Gamepad`,
  :class:`~isaaclab_teleop.Se3GamepadCfg`, :class:`~isaaclab_teleop.Se2SpaceMouse`,
  :class:`~isaaclab_teleop.Se2SpaceMouseCfg`, :class:`~isaaclab_teleop.Se3SpaceMouse`, and
  :class:`~isaaclab_teleop.Se3SpaceMouseCfg`, built on the IsaacTeleop session API. Update imports
  from ``isaaclab.devices`` / ``isaaclab.devices.gamepad`` / ``isaaclab.devices.spacemouse`` to
  ``isaaclab_teleop.gamepad`` / ``isaaclab_teleop.spacemouse`` (or ``isaaclab_teleop`` directly),
  and install the ``teleop`` extra (or a standalone ``isaaclab_teleop`` install).
