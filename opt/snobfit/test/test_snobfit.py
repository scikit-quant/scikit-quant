import py, os, sys
import numpy as np

sys.path = [os.path.join(os.pardir, 'python')] + sys.path


class TestSNOBFIT:
    def setup_class(cls):
        import SQSnobFit, logging, numpy
        SQSnobFit.log.setLevel(logging.DEBUG)
        logging.getLogger('SKQ.SnobFit.minq').setLevel(logging.INFO)

    def reset(self):
        import SQSnobFit

     # reset the random state for each method to get predictable results
        SQSnobFit._gen_utils._randstate = np.random.RandomState(6)

    def setup_method(self, method):
        self.reset()

    def test01_simple_example(self):
        """Read access to instance public data and verify values"""

        import SQSnobFit

        def f_easy(x):
            from math import sin

            fv = np.inner(x, x)
            fv *= 1 + 0.1*sin(10*(x[0]+x[1]))

            return fv

        def run_easy(self, initial, expected):
            self.reset()

            bounds = np.array([[-1, 1], [-1, 1]], dtype=float)
            budget = 40
            x0 = np.array(initial)

            from SQSnobFit import optset
            options = optset(maxmp=2+6)

            result, history = SQSnobFit.minimize(f_easy, x0, bounds, budget, options)

          # problem is symmetric, so values may have switched: just check the sum
            assert np.round(sum(result.optpar)-sum(expected), 8) == 0
            #assert len(histout) == 10

        run_easy(self, [],         (-0.00112, -0.00078))
        run_easy(self, [0.5, 0.5], ( 0.00134, -0.00042))

    def test02_bra(self):
        """Minimize Branin's function"""

        import SQSnobFit

        def bra(x):
            from math import cos, pi

            a = 1
            b = 5.1/(4*pi*pi)
            c = 5/pi
            d = 6
            h = 10
            ff = 1/(8*pi)

            return a*(x[1]-b*x[0]**2+c*x[0]-d)**2+h*(1-ff)*cos(x[0])+h

        bounds = np.array([[-5, 5], [-5, 5]], dtype=float)
        budget = 80      # larger budget needed for full convergence
        x0 = np.array([0.5, 0.5])

        from SQSnobFit import optset
        options = optset(maxmp=len(x0)+6)

        result, history = SQSnobFit.minimize(bra, x0, bounds, budget, options)
        # LIMIT:
        # fglob = 0.397887357729739
        # xglob = [3.14159265, 2.27500000]
        assert np.round(sum(result.optpar)-sum((3.1416, 2.275)), 8) == 0

    def test03_Hartman6(self):
        """Minimize Hartman6 function"""

        import SQSnobFit

        def Hartman6(x):
            import numpy, math

            a = numpy.array(
                [[10.00,  0.05,  3.00, 17.00],
                 [ 3.00, 10.00,  3.50,  8.00],
                 [17.00, 17.00,  1.70,  0.05],
                 [ 3.50,  0.10, 10.00, 10.00],
                 [ 1.70,  8.00, 17.00,  0.10],
                 [ 8.00, 14.00,  8.00, 14.00]])

            p = numpy.array(
                [[0.1312, 0.2329, 0.2348, 0.4047],
                 [0.1696, 0.4135, 0.1451, 0.8828],
                 [0.5569, 0.8307, 0.3522, 0.8732],
                 [0.0124, 0.3736, 0.2883, 0.5743],
                 [0.8283, 0.1004, 0.3047, 0.1091],
                 [0.5886, 0.9991, 0.6650, 0.0381]])

            c = numpy.array([1.0, 1.2, 3.0, 3.2])

            d = numpy.zeros((4,))
            for i in range(4):
                d[i] = sum(a[:,i]*(x - p[:,i])**2)

            return -(c.dot(numpy.exp(-d)))

        def run_Hartman6(self, initial, expected, budget=250, options=None):
            self.reset()

            bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=float)
            x0 = np.array(initial)

            from SQSnobFit import optset
            if options is None:
                options = optset(maxmp=6+6)
            result, history = SQSnobFit.minimize(Hartman6, x0, bounds, budget, options)

            assert np.round(sum(result.optpar)-sum(expected), 8) == 0

      # note: results are still subtly different from the reference, but the errors seem to
      # accumulate over many iterations, not produced by a single step going wrong and the
      # results here actually slightly outperform (?)
        run_Hartman6(self, [],      (0.20687, 0.14968, 0.48076, 0.27357, 0.3145,  0.66129))
        run_Hartman6(self, [0.5]*6, (0.21015, 0.12032, 0.46593, 0.27053, 0.30835, 0.66478))

      # regression: the following used to fail
        run_Hartman6(self, [],      (0.02423, 0.12107, 0.98254, 0.05482, 0.07433, 0.86491),
                     options={'minfcall' : 10, 'maxmp' : 1})
        run_Hartman6(self, [],      (0.24101, 0.16523, 0.43633, 0.28035, 0.31964, 0.64909),
                     options={'maxmp' : 2, 'maxfail' : 10})

      # the following is super-slow and not common, so not currently run; uncomment to test
        #run_Hartman6(self, [],  (0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65730),
        #             budget=2000, options={'maxmp' : 12, 'maxfail' : 500})

    def test04_direct_call(self):
        """Direct call of a single iteration"""

        import SQSnobFit
        import numpy as np

        x = np.array([[23, 23], [ 50, 50],  [50, 70],  [70, 70]])
        f = np.array([[ 0, -1], [-34, -1], [-83, -1], [-85, -1]])

        bounds = np.array([[0, 100], [0, 100]])
        config = {'bounds': bounds, 'p': .5, 'nreq': 2*2+6}
        dx = (bounds[:,1]-bounds[:,0])*1E-2

        request, xbest, fbest = SQSnobFit.snobfit(x,f,config,dx)

        assert xbest[0] == 70 and xbest[1] == 70
        assert fbest == -85
