Added
^^^^^

* Added :class:`~isaaclab_teleop.Se2Keyboard`, :class:`~isaaclab_teleop.Se2KeyboardCfg`,
  :class:`~isaaclab_teleop.Se3Keyboard`, and :class:`~isaaclab_teleop.Se3KeyboardCfg` as the
  supported home for keyboard teleoperation, moved from :mod:`isaaclab.devices.keyboard`.
* Added :attr:`~isaaclab_teleop.IsaacTeleopDevice.last_step_result`, a read-only passthrough to
  the most recent full pipeline output (not just the flattened ``"action"`` tensor), for devices
  whose ``pipeline_builder`` declares extra outputs.

Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_teleop.Se2Keyboard` and :class:`~isaaclab_teleop.Se3Keyboard`
  are now built on the IsaacTeleop session API (:class:`~isaaclab_teleop.IsaacTeleopDevice`)
  instead of raw Omniverse ``carb``/``omni`` keyboard events. Raw key state is read through a
  bundled ``keyboard`` IsaacTeleop plugin (Linux evdev) and retargeted through the retargeting
  engine, requiring a standalone CloudXR/OpenXR session at construction time. The public
  ``Se2KeyboardCfg`` / ``Se3KeyboardCfg`` fields, ``reset()``, ``add_callback(key, func)``, and
  ``advance()`` contracts are unchanged. ``add_callback("START"/"STOP"/"RESET"/"R", func)`` now
  binds to this device's own teleop control events (fired by the ``B`` / ``P`` / ``R`` keys
  respectively), matching the same callback contract used by XR-based IsaacTeleop devices.
  Callers must now call ``advance()`` on every constructed keyboard device each frame (even ones
  used only for callback side effects, not for their returned action) so that its physical-key
  polling runs; this matches the ``advance()``-driven polling model already used by every other
  IsaacTeleop device.
