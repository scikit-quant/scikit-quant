import py, os, sys
from pytest import raises
import numpy as np

sys.path = [os.pardir] + sys.path


class TestOPTIMIZERS:
    def setup_class(cls):
        pass

    def reset(self):
        import SQSnobFit

     # reset the random state for each method to get predictable results
        SQSnobFit._gen_utils._randstate = np.random.RandomState(6)

    def setup_method(self, method):
        self.reset()

    def test_issue2(self):
        """Errors with imfil for univariate functions"""

        from skquant.opt import minimize

        def f(a):
            return a**2 - a

        bounds = np.array([[0,2]], dtype=np.float)
        init = np.array([1.])
        res, hist = minimize(f, init, bounds, method='imfil')

    def test_issue3(self):
        """error with snobfit for univariate function"""

        from skquant.opt import minimize

        def f(a):
            return a[0]**2 - a[0]

        bounds = np.array([[0,2]], dtype=np.float)
        init = np.array([1.])
        res, hist = minimize(f, init, bounds, method='snobfit')

    def test_issue4(self):
        """error in imfil with multivariate function"""

        from skquant.opt import minimize

        def g(a):
            return a[0]**2 - a[0] +a[1]**3 -4*a[1]

        bounds = np.array([[0,2],[-2,2]], dtype=np.float)
        init = np.array([1.,0.])
        res, hist = minimize(g, init, bounds, method='imfil')
