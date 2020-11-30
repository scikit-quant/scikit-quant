from __future__ import print_function

import logging
import skquant.opt as skqopt

from qiskit.aqua.components.optimizers import Optimizer

__all__ = [
    'ImFil',
    'SnobFit',
    'PyBobyqa',
    'Bobyqa',
    ]

log = logging.getLogger('SKQ')


#
### ImFil Qiskit interoperable interface
#
class ImFil(Optimizer):
    """
    Implicit Filtering

    Algorithm designed for problems with local minima caused by high-frequency,
    low-amplitude noise, with an underlying large scale structure. This uses
    the SQImFil Python rewrite.

    Reference:
      C.T. Kelley, "Implicit Filtering", 2011, ISBN: 978-1-61197-189-7

    Original MATLAB code available at ctk.math.ncsu.edu/imfil.html
    """

    _OPTIONS = []

    def __init__(self,
                 maxfun: int = 500) -> None:

        """
        Args:
            maxfun: Maximum number of function evaluations.
        """

        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

        self.maxfun = maxfun

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.required,
            'initial_point': Optimizer.SupportLevel.supported,
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res, history = \
             skqopt.minimize(objective_function, initial_point, variable_bounds,
                             self.maxfun, method='imfil', options=self._options)

        return res.optpar, res.optval, len(history)


#
### SnobFit Qiskit interoperable interface
#
class SnobFit(Optimizer):
    """
    Stable Noisy Optimization by Branch and FIT

    SnobFit is specifically developed for optimization problems with noisy and
    expensive to compute objective functions. This implementation uses the
    SQSnobFit Python rewrite.

    Reference:
      W. Huyer and A. Neumaier, “Snobfit - Stable Noisy Optimization by Branch
      and Fit”, ACM Trans. Math. Software 35 (2008), Article 9.

    Original MATLAB code available at www.mat.univie.ac.at/~neum/software/snobfit
    """

    _OPTIONS = ['maxmp']

    def __init__(self,
                 maxfun: int = 500,
                 maxmp: int = -1) -> None:

        """
        Args:
            maxfun: Maximum number of function evaluations.
            maxmp:  Maximum number of function evaluations per iteration.
        """

        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

        self.maxfun = maxfun

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.required,
            'initial_point': Optimizer.SupportLevel.supported,
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res, history = \
             skqopt.minimize(objective_function, initial_point, variable_bounds,
                             self.maxfun, method='snobfit', options=self._options)

        return res.optpar, res.optval, len(history)


#
### NOMAD Qiskit interoperable interface
#
class NOMAD(Optimizer):
    """
    Nonlinear Optimization by Mesh Adaptive Direct Search

    NOMAD is designed for time-consuming blackbox simulations, with a small
    number of variables, that may fail. It samples the parameter space using a
    mesh that is adaptively adjusted based on the progress of the search.

    Reference:
      C. Audet, S. Le Digabel, C. Tribes and V. Rochon Montplaisir. "The NOMAD
      project." Software available at https://www.gerad.ca/nomad .

      S. Le Digabel. "NOMAD: Nonlinear Optimization with the MADS algorithm."
      ACM Trans. on Mathematical Software, 37(4):44:1–44:15, 2011.

    Original C++ code available at www.gerad.ca/nomad
    """

    _OPTIONS = []

    def __init__(self,
                 maxfun: int = 500) -> None:

        """
        Args:
            maxfun: Maximum number of function evaluations.
        """

        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

        self.maxfun = maxfun

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.required,
            'initial_point': Optimizer.SupportLevel.supported,
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res, history = \
             skqopt.minimize(objective_function, initial_point, variable_bounds,
                             self.maxfun, method='nomad', options=self._options)

        return res.optpar, res.optval, len(history)

Nomad = NOMAD


#
### PyBobyqa Qiskit interoperable interface
#
class PyBobyqa(Optimizer):
    """
    Bound Optimization BY Quadratic Approximation

    Trust region method that builds a quadratic approximation in each iteration
    based on a set of automatically chosen and adjusted interpolation points. 

    Reference:
      Coralia Cartis, et. al., “Improving the Flexibility and Robustness of
      Model-Based Derivative-Free Optimization Solvers”, technical report,
      University of Oxford, (2018).

    Code available at github.com/numericalalgorithmsgroup/pybobyqa/
    """

    _OPTIONS = []

    def __init__(self,
                 maxfun: int = 500) -> None:

        """
        Args:
            maxfun: Maximum number of function evaluations.
        """

        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v

        self.maxfun = maxfun

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.required,
            'initial_point': Optimizer.SupportLevel.required,
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res, history = \
             skqopt.minimize(objective_function, initial_point, variable_bounds,
                             self.maxfun, method='bobyqa', options=self._options)

        return res.optpar, res.optval, len(history)

Bobyqa = PyBobyqa
