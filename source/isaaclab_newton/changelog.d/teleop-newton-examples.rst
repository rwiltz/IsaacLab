Fixed
^^^^^

* Fixed contact sensors under Newton never having their force attribute allocated. The request
  registered by :meth:`~isaaclab_newton.physics.NewtonManager.request_extended_contact_attribute`
  was cleared immediately after being forwarded, so a later model rebuild allocated a ``Contacts``
  object without it, and :meth:`add_contact_sensor` never registered its own ``force`` request at
  all. Either alone raised ``SensorContact requires a Contacts object with force allocated`` on the
  first step of any scene with a contact sensor and ``use_mujoco_contacts=False``.

* Fixed a hard reset discarding every actuator property a task configures. ``sim.reset(soft=False)``
  re-finalizes the Newton model from the builder, and the rebuilt model carries the USD-authored
  joint drives instead: measured on a humanoid, stiffness 4400 to 53026, damping 40 to 2148,
  armature 0.1 to 0.0 and the effort limit to ``inf``. MJWarp cannot integrate those, so the first
  commanded motion after a reset drove the articulation non-finite while a zero command still
  looked healthy. The actuator configs are now re-applied when physics becomes ready. Live
  teleoperation was unaffected, since it only calls ``env.reset()``; record and replay was not.
