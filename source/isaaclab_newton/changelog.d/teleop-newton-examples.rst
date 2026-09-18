Fixed
^^^^^

* Fixed contact sensors under Newton never having their force attribute allocated. The request
  registered by :meth:`~isaaclab_newton.physics.NewtonManager.request_extended_contact_attribute`
  was cleared immediately after being forwarded, so a later model rebuild allocated a ``Contacts``
  object without it, and :meth:`add_contact_sensor` never registered its own ``force`` request at
  all. Either alone raised ``SensorContact requires a Contacts object with force allocated`` on the
  first step of any scene with a contact sensor and ``use_mujoco_contacts=False``.
