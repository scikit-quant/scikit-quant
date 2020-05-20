import py, os, sys
from pytest import raises
import numpy as np

sys.path = [os.pardir] + sys.path


def f_easy_simple(x):
    from math import sin

    fv = np.inner(x, x)
    fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

    return fv

ref_results = {
    f_easy_simple : {
        'imfil'   : (-0.00681757, 0.00880743),
        'snobfit' : (-0.00112, -0.00078),
        'bobyqa'  : (-2.3883e-10, -1.3548e-10),
    }
}


class TestOPTIMIZERS:
    def setup_class(cls):
        pass

    def reset(self):
        import SQSnobFit

     # reset the random state for each method to get predictable results
        SQSnobFit._gen_utils._randstate = np.random.RandomState(6)

    def setup_method(self, method):
        self.reset()

    def test01_availability(self):
        """Quick API check on available optimizers"""

        import skquant.opt as skqopt

        bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
        budget = 40
        x0 = np.array([0.5, 0.5])

        # interface with incorrect input
        assert raises(RuntimeError, skqopt.minimize, f_easy_simple, x0, bounds, budget, method='does not exist')

        # ImFil
        result, history = \
             skqopt.minimize(f_easy_simple, x0, bounds, budget, method='imfil')
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar)-sum(ref_results[f_easy_simple]['imfil']), 8) == 0.0

        # SnobFit
        from SQSnobFit import optset
        options = optset(maxmp=len(x0)+6)
        result, history = \
             skqopt.minimize(f_easy_simple, [], bounds, budget, method='snobfit', options=options)
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar)-sum(ref_results[f_easy_simple]['snobfit']), 8) == 0.0

        # Py-BOBYQA
        result, history = \
             skqopt.minimize(f_easy_simple, x0, bounds, budget, method='bobyqa')
        assert type(result.optpar) == np.ndarray
        assert np.round(sum(result.optpar), 5) == 0

        # ORBIT
        if skqopt._check_orbit_prerequisites():
            randstate = 1
            np.random.seed(randstate)
            result, history = \
                 skqopt.minimize(f_easy_simple, x0, bounds, budget, method='orbit')
            assert type(result.optpar) == np.ndarray
            assert np.round(sum(result.optpar)-sum((0.00076624, 0.00060909)), 7) == 0

    def test02_qiskit_interface(self):
        """Verify usability of the Qiskit Aqua interface"""

        try:
            import qiskit.aqua
        except ImportError:
            py.test.skip("qiskit.aqua not availab; not testing qiskit interface")

        bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
        budget = 40
        x0 = np.array([0.5, 0.5])

      # ImFil
        from skquant.interop.qiskit import ImFil

        optimizer = ImFil(maxfun=budget)

        ret = optimizer.optimize(num_vars=len(x0), objective_function=f_easy_simple, \
                                 variable_bounds=bounds, initial_point=x0)
        assert np.round(sum(ret[0])-sum(ref_results[f_easy_simple]['imfil']), 8) == 0.0

      # SnobFit
        from skquant.interop.qiskit import SnobFit

        optimizer = SnobFit(maxfun=budget, maxmp=len(x0)+6)

        ret = optimizer.optimize(num_vars=len(x0), objective_function=f_easy_simple, \
                                 variable_bounds=bounds)
        assert np.round(sum(ret[0])-sum(ref_results[f_easy_simple]['snobfit']), 8) == 0.0

      # PyBobyqa
        from skquant.interop.qiskit import PyBobyqa
        optimizer = PyBobyqa(maxfun=budget)

        ret = optimizer.optimize(num_vars=len(x0), objective_function=f_easy_simple, \
                                 variable_bounds=bounds, initial_point=x0)
        assert np.round(sum(ret[0])-sum(ref_results[f_easy_simple]['bobyqa']), 8) == 0.0
