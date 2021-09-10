Prerequisites
=============

Different parts of the tutorial require some or all of the following packages:

    - BQSKit
    - Scikit-Quant
    - Qiskit
    - OpenFermion
    - SciPy
    - Jupyter

These install easily on most platforms using Python's `pip` and `venv`, with
the exception of Macbooks with an M1 chip, where Anaconda
(https://anaconda.org/) should be used to be able to install Qiskit.
Since the download of Anaconda is quite large, and may trigger the subsequent
download and install of Apple's Rosetta, it is highly recommended to
pre-install these packages before attending the tutorial.

The following lists the platform-specific instructions to setup Python and a
virtual environment, as well as to install all packages.


Mac M1
------

Install Anaconda (https://www.anaconda.com/products/individual) for Intel 64b
(ignore any warnings about the M1 not being a 64b platform). Follow the steps
to setup the conda environment for your shell (run in the Terminal app). Then
create a new conda project named "TUTORIAL"::

    (base) $ conda create -n TUTORIAL python=3.9
    (TUTORIAL) $ conda activate TUTORIAL
    (TUTORIAL) $ python -m pip install bqskit scikit-quant qiskit openfermion scipy jupyter

If the system asks to install Apple's Rosetta, accept the install.


Linux and Mac Intel
-------------------

Create and activate a virtual environment for Python, then install::

    $ python3 -m venv TUTORIAL
    $ source TUTORIAL/bin/activate
    (TUTORIAL) $ python -m pip install bqskit scikit-quant qiskit openfermion scipy jupyter


Windows
-------

Install Python (https://www.python.org/downloads/windows/) if not already
available on your system. On the command prompt, create and setup a virtual
environment::

    $ python3 -m venv TUTORIAL
    $ TUTORIAL\Scripts\activate
    (TUTORIAL) $ python -m pip install bqskit scikit-quant qiskit openfermion scipy jupyter
