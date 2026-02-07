Installing airspeed velocity
============================

**airspeed velocity** is known to work on Linux, MacOS, and Windows, for Python
3.9 and higher. PyPy 3.10 is also supported.

**airspeed velocity** is a standard Python package, and the latest released
version may be `installed from PyPI
<https://packaging.python.org/tutorials/installing-packages/>`__:

.. code-block:: sh

    pip install asv

The development version can be installed from GitHub:

.. code-block:: sh

   git clone git@github.com:airspeed-velocity/asv
   cd asv
   pip install .
   # Or in one shot
   pip install git+https://github.com/airspeed-velocity/asv

The basic requirements should be automatically installed.  If they aren't
installed automatically, for example due to networking restrictions, the
``python`` requirements are as noted in the ``pyproject.toml``.

Running the self-tests
----------------------

The testsuite is based on `pytest <https://docs.pytest.org/>`__.

To run **airspeed velocity**'s testsuite::

    pytest
