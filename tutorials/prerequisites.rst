Prerequisites
=============

The the following packages should be installed to follow this tutorial:

    o BQSKit
    o Scikit-Quant
    o Qiskit
    o OpenFermion
    o SciPy
    o Jupyter

The easiest way to install them is to use `pip`, howver on Macbook's with
an M1 chip, Qiskit can not be installed natively due to inline assembly code.
To install Qiskit on that platform, install Anaconda: https://anaconda.org/
(ignore any warnings about the M1 not being a 64b platform). Follow the steps
to setup the environment and then, in the Terminal app, install Qiksit in a
new conda project::

    $ conda create -n TUTORIAL python=3.8
    $ conda activate TUTORIAL
    $ python -m pip install bqskit scikit-quant qiskit openfermion scipy jupyter

If the system asks to install Apple's Rosetta (either when running conda or
later when running Python), accept the install.