import py, os, sys
from pytest import raises
import numpy as np

sys.path = [os.path.join(os.pardir, 'python')] + sys.path


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
        res, histout, complete_history = \
             minimize(f_easy_simple, x0, bounds, budget, method='imfil')
        assert type(res.optpar) == np.ndarray
        assert np.round(histout[9,5:7].sum()-sum((-6.81757191e-03, 8.80742809e-03)), 8) == 0

        assert raises(RuntimeError, minimize, f_easy_simple, x0, bounds, budget, method='does not exist')
