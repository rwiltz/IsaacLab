Deprecated
^^^^^^^^^^

* Deprecated :class:`~isaaclab.devices.Se2Keyboard`, :class:`~isaaclab.devices.Se2KeyboardCfg`,
  :class:`~isaaclab.devices.Se3Keyboard`, and :class:`~isaaclab.devices.Se3KeyboardCfg` in favor of
  :class:`~isaaclab_teleop.Se2Keyboard`, :class:`~isaaclab_teleop.Se2KeyboardCfg`,
  :class:`~isaaclab_teleop.Se3Keyboard`, and :class:`~isaaclab_teleop.Se3KeyboardCfg`. Existing
  imports from :mod:`isaaclab.devices.keyboard` will continue to work as long as ``isaaclab_teleop``
  is installed (see the ``teleop`` extra).
