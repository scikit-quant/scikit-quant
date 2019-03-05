import py, os, sys
import numpy as np

sys.path = [os.path.join(os.pardir, 'python')] + sys.path


class TestIMFIL:
    def setup_class(cls):
        import SQImFil, logging
        SQImFil.log.setLevel(logging.DEBUG)

    def test01_simple_example(self):
        """Read access to instance public data and verify values"""

        import SQImFil

        def f_easy_simple(x):
            from math import sin

            fv = np.inner(x, x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

            return fv

        def f_easy(x):
            from math import sin

            fv = np.inner(x, x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

            return (fv, 0, 1)

        def f_easy_parallel(ax):
            from math import sin

            if len(ax.shape) == 1:
                return f_easy(ax)

            res = list()
            for i in range(ax.shape[0]):
                res.append(f_easy(ax[i])[0])
            return (res, [0]*len(res), [1]*len(res))

        bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
        budget = 40
        x0 = np.array([0.5, 0.5])

        opt_common = {'scale_depth' : 7, 'complete_history' : True}
        for func in [f_easy_simple, f_easy]:#, f_easy_parallel]:
            if func == f_easy:
                optset = SQImFil.optset(simple_function=False, **opt_common)
            elif func == f_easy_parallel:
                optset = SQImFil.optset(parallel=True, **opt_common)
            else:
                optset = SQImFil.optset(**opt_common)
            res, histout, complete_history = SQImFil.minimize(func, x0, bounds, budget, optset)
            assert type(res.optpar) == np.ndarray
            # this problem is symmetric, so values may have switched; for
            # simplicity, just check the sum
            assert np.round(sum(res.optpar)-sum((-0.00681757, 0.00880743)), 8) == 0
            assert len(histout) == 10
            assert np.round(histout[9,5:7].sum()-sum((-6.81757191e-03, 8.80742809e-03)), 8) == 0

        # TODO: figure out why the parallel version performs a bit better
        optset = SQImFil.optset(parallel=True, **opt_common)
        res, histout, complete_history = \
             SQImFil.minimize(f_easy_parallel, x0, bounds, budget, optset)
        assert type(res.optpar) == np.ndarray
        assert np.round(sum(res.optpar)-sum((0.00281554, 0.00281554)), 5) == 0

    def test02_simple_example_multi_local(self):
        """Verify with several local minima (ie. start point dependent"""

        import SQImFil

        def f_easy(x):
            from math import sin, cos, pi

            fv = x.T.dot(x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1])) - cos(pi*x[2]/2.)

            return (fv, 0, 1)

        import numpy as np
        bounds = np.array([[-1, 1], [-1, 1], [0, 1]], dtype=float)
        budget = 40
        optset = SQImFil.optset(scale_depth=7, complete_history=0)

        x0 = np.array([0., 0., 0.])
        res, histout = SQImFil.minimize(f_easy, x0, bounds, budget, optset)
        assert np.round(res.optval, 5) == -0.12422

        x0 = np.array([-1.0, -1.0, 0.011198])
        res, histout = SQImFil.minimize(f_easy, x0, bounds, budget, optset)
        assert np.round(res.optval, 5) == -0.18258

        x0 = np.array([-1.0, 0.222183, 0.037152])
        res, histout = SQImFil.minimize(f_easy, x0, bounds, budget, optset)
        assert np.round(res.optval, 5) == -0.10459

        x0 = np.array([-0.8305, 0., 0.314])
        res, histout = SQImFil.minimize(f_easy, x0, bounds, budget, optset)
        assert np.round(res.optval, 5) == -0.18259

        xmin = np.array([-1.0, -1.0, 0.0])
        assert type(res.optpar) == np.ndarray
        assert f_easy(res.optpar)[0] <= f_easy(x0)[0]
        assert res.optval            <= f_easy(x0)[0]
        assert np.round(sum(res.optpar-xmin), 8) == 0
        assert f_easy(res.optpar)[0] == f_easy(xmin)[0]
        assert round(res.optval-f_easy(res.optpar)[0], 8) == 0
