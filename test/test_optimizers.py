import py, os, sys
from pytest import raises
import numpy as np

sys.path = [os.pardir] + sys.path


class TestOPTIMIZERS:
    def setup_class(cls):
        pass

    def test01_availability(self):
        """Quick API check on available optimizers"""

        def f_easy_simple(x):
            from math import sin

            fv = np.inner(x, x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

            return fv

        from skquant.opt import minimize

        bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
        budget = 40
        x0 = np.array([0.5, 0.5])

        # interface with incorrect input
        assert raises(RuntimeError, minimize, f_easy_simple, x0, bounds, budget, method='does not exist')

        # ImFil
        result, history = \
             minimize(f_easy_simple, x0, bounds, budget, method='imfil')
        assert type(result.optpar) == np.ndarray

        # SnobFit
        from SQSnobFit import optset
        options = optset(maxmp=len(x0)+6)
        result, history = \
             minimize(f_easy_simple, x0, bounds, budget, method='snobfit', options=options)
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar)-sum((-0.00112, -0.00078)), 8) == 0

        # Py-BOBYQA
        result, history = \
             minimize(f_easy_simple, x0, bounds, budget, method='bobyqa')
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar), 5) == 0

        # ORBIT
        if sys.version_info[0] == 3:
            randstate = 1
            np.random.seed(randstate)
            result, history = \
                 minimize(f_easy_simple, x0, bounds, budget, method='orbit')
            assert type(result.optpar) == np.ndarray
            assert np.round(sum(result.optpar)-sum((0.00076624, 0.00060909)), 7) == 0
