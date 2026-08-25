Added
^^^^^

* Added :func:`~isaaclab_teleop.se2_keyboard_teleop_cfg` and
  :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`, functions that build an
  :class:`~isaaclab_teleop.IsaacTeleopCfg` for keyboard-driven SE(2)/SE(3) teleoperation, moved
  from :mod:`isaaclab.devices.keyboard`. Raw key state is read through a bundled ``keyboard``
  IsaacTeleop plugin (Linux evdev) and retargeted through the retargeting engine.
* Added :class:`~isaaclab_teleop.control_pollers.KeyboardControlPoller`, a standalone utility
  that polls a keyboard-plugin-backed :class:`~isaaclab_teleop.IsaacTeleopDevice`'s
  ``keyboard_all_keys`` output and fires ``request_start()`` / ``request_stop()`` /
  ``reset(pause=True)`` on the physical ``B`` / ``P`` / ``R`` keys, plus any callback registered
  via ``add_callback(key, func)``, on rising edges. Construct one alongside the teleop device and
  call ``.advance()`` on it once per frame.
* Added :attr:`~isaaclab_teleop.IsaacTeleopDevice.last_step_result`, a read-only passthrough to
  the most recent full pipeline output (not just the flattened ``"action"`` tensor), for devices
  whose ``pipeline_builder`` declares extra outputs.
