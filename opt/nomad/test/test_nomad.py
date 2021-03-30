import py, os, sys
import numpy as np

sys.path = [os.path.join(os.pardir, 'python')] + sys.path


class TestNOMAD:
    def setup_class(cls):
        import SQNomad, logging
        SQNomad.log.setLevel(logging.DEBUG)

    def test01_simple_example(self):
        """Read access to instance public data and verify values"""

        import SQNomad

        def f_easy_simple(x):
            from math import sin

            fv = np.inner(x, x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

            return fv

        bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
        budget = 40
        x0 = np.array([0.5, 0.5])

        result, history = SQNomad.minimize(f_easy_simple, x0, bounds, budget, SEED=1)

      # problem is symmetric, so values may have switched: just check the sum
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar)-sum([2.77555756e-17, -0.]), 8) == 0
