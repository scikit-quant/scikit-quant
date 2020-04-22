import py, os, sys
from pytest import raises
import numpy as np

sys.path = [os.pardir] + sys.path


class TestOPTIMIZERS:
    def setup_class(cls):
        pass

    def test_issue2(self):
        """Errors with imfil for univariate functions"""

        import numpy as np
        from skquant.opt import minimize

        def f(a):
            return a**2 - a

        bounds = np.array([[0,2]], dtype=np.float)
        init = np.array([1.])
        res, hist = minimize(f, init, bounds, method='imfil')
