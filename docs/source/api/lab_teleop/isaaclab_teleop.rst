.. _isaaclab_teleop-api:

isaaclab_teleop
===============

.. automodule:: isaaclab_teleop

  .. rubric:: Classes

  .. autosummary::

    IsaacTeleopCfg
    IsaacTeleopDevice
    Se2Gamepad
    Se2GamepadCfg
    Se2Keyboard
    Se2KeyboardCfg
    Se2SpaceMouse
    Se2SpaceMouseCfg
    Se3Gamepad
    Se3GamepadCfg
    Se3Keyboard
    Se3KeyboardCfg
    Se3SpaceMouse
    Se3SpaceMouseCfg
    XrCameraFeedCfg
    XrCameraFeedLayoutCfg
    XrCameraFeedSession
    HapticFeedbackCfg
    ControllerHapticFeedbackCfg
    GloveHapticFeedbackCfg
    HapticFeedbackReceiver
    HapticFeedbackDriver
    XrCfg
    XrAnchorRotationMode
    XrAnchorSynchronizer

  .. rubric:: Functions

  .. autosummary::

    create_isaac_teleop_device
    create_haptic_feedback_driver
    remove_camera_configs

Configuration
-------------

.. autoclass:: IsaacTeleopCfg
    :members:

.. autoclass:: XrCfg
    :members:

.. autoclass:: XrAnchorRotationMode
    :members:

XR Camera Feedback
------------------

.. autoclass:: XrCameraFeedCfg
    :members:

.. autoclass:: XrCameraFeedLayoutCfg
    :members:

.. autoclass:: XrCameraFeedSession
    :members:

Device
------

.. autoclass:: IsaacTeleopDevice
    :members:
    :show-inheritance:

.. autofunction:: create_isaac_teleop_device

Keyboard
--------

.. autoclass:: Se2Keyboard
    :members:
    :show-inheritance:

.. autoclass:: Se2KeyboardCfg
    :members:

.. autoclass:: Se3Keyboard
    :members:
    :show-inheritance:

.. autoclass:: Se3KeyboardCfg
    :members:

Gamepad
-------

.. autoclass:: Se2Gamepad
    :members:
    :show-inheritance:

.. autoclass:: Se2GamepadCfg
    :members:

.. autoclass:: Se3Gamepad
    :members:
    :show-inheritance:

.. autoclass:: Se3GamepadCfg
    :members:

SpaceMouse
----------

.. autoclass:: Se2SpaceMouse
    :members:
    :show-inheritance:

.. autoclass:: Se2SpaceMouseCfg
    :members:

.. autoclass:: Se3SpaceMouse
    :members:
    :show-inheritance:

.. autoclass:: Se3SpaceMouseCfg
    :members:

Haptic Feedback
---------------

.. autoclass:: HapticFeedbackCfg
    :members:

.. autoclass:: ControllerHapticFeedbackCfg
    :members:
    :show-inheritance:

.. autoclass:: GloveHapticFeedbackCfg
    :members:
    :show-inheritance:

.. autoclass:: HapticFeedbackReceiver
    :members:

.. autoclass:: HapticFeedbackDriver
    :members:

.. autofunction:: create_haptic_feedback_driver

XR Anchor
---------

.. autoclass:: XrAnchorSynchronizer
    :members:

.. autofunction:: remove_camera_configs

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab_teleop` API.

.. currentmodule:: isaaclab_teleop

.. autosummary::
   :nosignatures:

   ControlEvents
   SupportsControlEvents
   SystemCheckItem
   SystemCheckResult

.. autoclass:: ControlEvents
   :show-inheritance:

.. autoclass:: SupportsControlEvents
   :show-inheritance:

.. autoclass:: SystemCheckItem
   :show-inheritance:

.. autoclass:: SystemCheckResult
   :show-inheritance:
