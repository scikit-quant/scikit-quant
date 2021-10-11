Prerequisites
=============

Different parts of the tutorial require some or all of the following packages:

    - BQSKit
    - Scikit-Quant
    - pyDOE
    - Qiskit
    - OpenFermion
    - SciPy
    - Jupyter

These install easily on most platforms using Python's `pip` and `venv`, with
the exception of Macbooks with an M1 chip, where Anaconda
(https://anaconda.org/) should be used to be able to install Qiskit.
Since the download of Anaconda is quite large, and may trigger the subsequent
download and install of Apple's Rosetta if that was not already enabled, it is
highly recommended to pre-install these packages before attending the tutorial.

The following lists the platform-specific instructions to setup Python and a
virtual environment, as well as to install all packages.

Note: the NOMAD optimizer is a C++ library that requires a local build and
thus a local C++ compiler to install. It is therefore not provided by default
and since it is not currently used in the tutorial, its installation is not
required. However, if you want to try it out, request it explicitly by
providing the `[NOMAD]` option to `scikit-quant` on the `pip` command (after
setting up the environment for your platform as instructed below), like so::

    (TUTORIAL) $ python -m pip install 'scikit-quant[NOMAD]'


Macs with an M1 chip
--------------------

Install Anaconda (https://www.anaconda.com/products/individual) for Intel 64b
(ignore any warnings about the M1 not being a 64b platform). Follow the steps
to setup the conda environment for your shell (run in the Terminal app). Then
create a new conda project named "TUTORIAL"::

    (base) $ conda create -n TUTORIAL python=3.9
    (TUTORIAL) $ conda activate TUTORIAL
    (TUTORIAL) $ python -m pip install bqskit scikit-quant pyDOE qiskit openfermion scipy jupyter

If the system asks to install Apple's Rosetta, accept the install.

The reason for requiring an Intel install is that Qiskit uses inline assembly
for Intel x86_64 chips. This will, obviously, fail when trying to install
natively for M1, but works fine through Rosetta. By choosing a full Intel 64b
install through Anaconda, it is guaranteed that all tools are Intel only. This
way, no platform mixing occurs, preventing spurious clashes.

There are also still outstanding problems with installing SciPy and NumPy from
PyPI on M1. Although these can easily be resolved by installing through a Mac
packager (such as MacPorts or Fink) instead, use of Anaconda will side-step
these installation issues as well.

Linux and Macs with an Intel chip
---------------------------------

Create and activate a virtual environment for Python, then install::

    $ python3 -m venv TUTORIAL
    $ source TUTORIAL/bin/activate
    (TUTORIAL) $ python -m pip install bqskit scikit-quant pyDOE qiskit openfermion scipy jupyter


Windows
-------

Install Python (https://www.python.org/downloads/windows/) if not already
available on your system. On a command prompt, create and setup a virtual
environment, then install::

    $ python3 -m venv TUTORIAL
    $ TUTORIAL\Scripts\activate
    (TUTORIAL) $ python -m pip install bqskit scikit-quant pyDOE qiskit openfermion scipy jupyter
