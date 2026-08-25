Added
^^^^^

* Added :class:`~isaaclab_teleop.Se2Gamepad`, :class:`~isaaclab_teleop.Se2GamepadCfg`,
  :class:`~isaaclab_teleop.Se3Gamepad`, :class:`~isaaclab_teleop.Se3GamepadCfg`,
  :class:`~isaaclab_teleop.Se2SpaceMouse`, :class:`~isaaclab_teleop.Se2SpaceMouseCfg`,
  :class:`~isaaclab_teleop.Se3SpaceMouse`, and :class:`~isaaclab_teleop.Se3SpaceMouseCfg` as the
  supported home for gamepad and spacemouse teleoperation, moved from
  :mod:`isaaclab.devices.gamepad` / :mod:`isaaclab.devices.spacemouse`.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_teleop.Se2Gamepad`, :class:`~isaaclab_teleop.Se3Gamepad`,
  :class:`~isaaclab_teleop.Se2SpaceMouse`, and :class:`~isaaclab_teleop.Se3SpaceMouse` are now
  built on the IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`) instead of
  raw Omniverse ``carb``/``omni`` gamepad events or the ``hid`` USB HID library. Raw stick/button
  state is read through bundled ``gamepad`` / ``spacemouse`` IsaacTeleop plugins and retargeted
  through the retargeting engine, requiring a standalone CloudXR/OpenXR session at construction
  time. The public ``Se2GamepadCfg`` / ``Se3GamepadCfg`` / ``Se2SpaceMouseCfg`` /
  ``Se3SpaceMouseCfg`` fields, ``reset()``, ``add_callback(key, func)``, and ``advance()``
  contracts are unchanged. ``add_callback("START"/"STOP"/"RESET"/"R", func)`` now binds to each
  device's own teleop control events, matching the same callback contract used by keyboard and
  XR-based IsaacTeleop devices. The spacemouse's right physical button now fires ``RESET``/``R``
  callbacks through this same mechanism (previously it only called the device's local ``reset()``
  without firing any registered callback, due to a missing call in the legacy ``Se2SpaceMouse``
  and a device-only reset in ``Se3SpaceMouse``).
