.. _installation:

Installation
============

To install with ``pip`` through `PyPI`_, it is recommend to use
`virtualenv`_ (or module `venv`_ for modern pythons).
The use of virtualenv prevents pollution of any system directories and allows
you to wipe out the full installation simply by removing the virtualenv
created directory ("WORK" in this example)::

  $ virtualenv WORK
  $ source WORK/bin/activate
  (WORK) $ python -m pip install scikit-quant
  (WORK) $

Note that ``SQNomad`` is optional, as it may take a long time to build on
machines with few cores.
You can either install it directly, using ``pip``, or enable the option::

  (WORK) $ python -m pip install scikit-quant[NOMAD]

If you use ``pip`` directly on the command line, instead of through
``python``, make sure that the ``PATH`` envar points to the bin directory
that will contain the installed entry points during the installation, as the
build process needs them.
You may also need to install ``wheel`` first if you have an older version of
``pip`` and/or do not use virtualenv (which installs wheel by default).
Older versions of ``pip`` may also require the ``--user`` option, if you do
not have write access to the installation directory.
Example::

  $ python -m pip install wheel --user
  $ PATH=$HOME/.local/bin:$PATH python -m pip install scikit-quant --user

Alternatively, you can use ``pip`` from a ``conda`` environment::

  $ conda create -n WORK
  $ conda activate WORK
  (WORK) $ python -m pip install scikit-quant
  (WORK) $

.. _`PyPI`: https://pypi.org/project/scikit-quant/
.. _`virtualenv`: https://pypi.python.org/pypi/virtualenv
.. _`venv`: https://docs.python.org/3/library/venv.html
