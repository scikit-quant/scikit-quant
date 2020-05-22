.. _testing:


Test suite
==========

All packages have a ``test`` subdirectory that contains tests runnable by
``pytest``.
In addition, the top-level ``test`` has a ``test_all.sh`` bash script to
walk the directories and run all tests.

To install ``pytest``::

   $ python -m pip install pytest

and to run any test, simply enter the ``test`` subdirectory and run::

   $ pytest

Some commonly used pytest parameters::

   -h : print help

   -x : stop on the first failing test
   -v : verbose
   -s : show captured output

   <file name> : run only tests from <file name>
   -k <expr>   : run only tests containing <expr> in their name
