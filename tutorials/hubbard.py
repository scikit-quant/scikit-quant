import logging_setup
import logging
import numpy as np
import openfermion as of
import os
import qiskit as qk
import qiskit.opflow as qk_opflow
import qiskit.quantum_info as qk_qi
import uccsd_evolution
import scipy.linalg as spla
import warnings

try:
    from hubbard_bqskit import BQSKit_Hubbard_Optimizer
except ImportError:
    pass

__all__ = [
    'EnergyObjective',
    'hamiltonian_matrix',
    'hamiltonian_qiskit',
    'small_model',
    'medium_model',
    'clear_circuit_cache',
    'get_cached_circuit',
]

logger = logging.getLogger('hubbard')


# Allow caching of the trotterized opflow, post-BQSKit, for noise studies
_cached_circuit = None

def clear_circuit_cache():
    global _cached_circuit
    _cached_circuit = None

def get_cached_circuit():
   return _cached_circuit


class EnergyObjective:
    def __init__(self, hamiltonian, n_electrons_up, n_electrons_down,
                 trotter_steps=2, noise_model=None, shots=-1,
                 run_bqskit=False, save_evals=None):
        """\
        Create an energy estimater for the given Hamiltonian

        Args:
            hamiltonian(opflow): Hamiltonian operator
            n_electrons_up(int): number of spin-up electrons in the physical system
            n_electrons_down(int): number of spin-down electrons in the physical system
            trotter_steps(int): number of Trotter time slices for the evolution
            noise_model(NoiseModel): Qiskit noise model to apply
            shots(int): number of shots to sample and average over
            run_bqskit(bool): whether to run the bqskit stack on the evolution operator
            save_evals(str): file name to store evaluations or None
        """

        self._hamiltonian = hamiltonian
        self._n_qubits         = hamiltonian.num_qubits
        self._n_electrons      = n_electrons_up + n_electrons_down

        try:
            self._fermion_transform = hamiltonian._fermion_transform
        except AttributeError as a:
            self._fermion_transform = 'jordan-wigner'

      # Create initial state and add electrons by setting qubits to |1> (i.e., occupied)
        reg = qk.QuantumCircuit(self._n_qubits)
        if self._fermion_transform == 'bravyi-kitaev':
            if self._n_electrons:
              # fill out the mapping of electrons
                m = [0]*self._n_qubits
                for i in range(n_electrons_up):
                    m[i*2] = 1
                for i in range(n_electrons_down):
                    m[i*2+1] = 1

                for i in range(self._n_qubits):
                    if i % 2:                              # odd
                        if sum(m[:i+1]) % 2: reg.x(i)
                    elif m[i]:                             # even
                        reg.x(i)

        elif self._fermion_transform == 'jordan-wigner':
            for i in range(n_electrons_up):
                reg.x(i*2)
            for i in range(n_electrons_down):
                reg.x(i*2+1)

        self._state_in = qk_opflow.CircuitStateFn(reg)

      # Create an observable from the Hamiltonian
        self._meas_op = qk_opflow.StateFn(self._hamiltonian, is_measurement=True)

      # Number of Trotter steps to use in the evolution operator (see __call__)
        self._trotter_steps = trotter_steps

      # Create the simulator
        if shots <= 0 and noise_model is None:
            self._simulator = None
            self._meas_components = None
            self._expectation = qk_opflow.MatrixExpectation()
        else:
            self._expectation = qk_opflow.PauliExpectation()
            if noise_model is None:
                backend = qk.Aer.get_backend('qasm_simulator')
            else:
              # nominally, options should pass through kwargs of get_backend, however,
              # this does not appear to work for Aer, so set the noise_model option
              # explicitly on the retrieved backend
                if type(noise_model) == str:
                  # use an existing, named, IBM backend from the qiskit test suite to
                  # create a realistic noise model; if 'realistic' default to Montreal
                    if noise_model.lower() == 'realistic':
                        noise_model = 'Montreal'

                    import qiskit.test.mock as qk_mock
                    import qiskit.providers.aer as qk_aer_provides

                    fake_backend = 'Fake'+noise_model[0].upper()+noise_model[1:]
                    fake_device  = getattr(qk_mock, fake_backend)()

                    backend = qk_aer_provides.AerSimulator.from_backend(fake_device)
                else:
                    backend = qk.Aer.get_backend('aer_simulator', noise_model=noise_model)
                    backend.set_options(noise_model=noise_model)

                if shots <= 0:
                  # if not simulating sampling, use the AerPauliExpectation, which computes
                  # the expectation value given the noise (effectively "infinite" sampling);
                  # it also passes a special "instruction" to the sampler to ignore shots
                     self._expectation = qk_opflow.AerPauliExpectation()
                     shots = 2**20          # i.e. large enough not to contribute

            self._simulator = qk_opflow.CircuitSampler(backend=backend)
            self._simulator.quantum_instance.run_config.shots = shots

          # split measurement components to prevent fake coherent errors
            primitive = self._meas_op.primitive
            try:
                while 1: primitive = primitive.primitive
            except AttributeError:
                pass

            self._meas_components = list()
            for ops, coeff in primitive.to_list():
                self._meas_components.append(qk_opflow.StateFn(
                    qk_opflow.PauliOp(qk.quantum_info.Pauli(ops), coeff), is_measurement=True))

      # Flag to toggle running the BQSKit optimizer on the evolution operator (see __call__)
        self.bqskit_opt = BQSKit_Hubbard_Optimizer(run_bqskit=='full') if run_bqskit else None

      # File name to store evaluations, if requested
        if save_evals:
            self._save_evals = type(save_evals) == str and save_evals or 'pointlog.txt'
            try:
                os.remove(self._save_evals)
            except Exception:
                pass
        else:
            self._save_evals = None

    def npar(self):
        """\
        Number of independent parameters for the optimizer

        Returns:
           npar(int): number of parameters for the optimizer
        """

        return uccsd_evolution.singlet_paramsize(self._n_qubits, self._n_electrons)

    def generate_evolution_op(self, packed_amplitudes):
        """\
        Construct the evolution operator

        Returns:
            trotterized_ev_op (opflow): (trotterized, optimized) evolution operator
        """

      # Build the state preparation evolution operator
        if self._fermion_transform == 'bravyi-kitaev':
            def bk_with_qubits(fop):
                return of.transforms.bravyi_kitaev(fop, self._n_qubits)
            fermion_transform = bk_with_qubits
        elif self._fermion_transform == 'jordan-wigner':
            fermion_transform = of.transforms.jordan_wigner

        evolution_op = uccsd_evolution.singlet_evolution(
                           packed_amplitudes, self._n_qubits, self._n_electrons,
                           fermion_transform=fermion_transform)

      # Trotterize the evolution operator flow to be able to construct a circuit (the
      # choice of 2 slices was empirically determined; it may not fit all cases)
        if 0 < self._trotter_steps:
            num_time_slices = self._trotter_steps
            trotterized_ev_op = qk_opflow.PauliTrotterEvolution(
                trotter_mode='trotter', reps=num_time_slices).convert(evolution_op)

          # Run bqskit circuit optimizers as requested (only works on the trotterized
          # evolution operator as the normal time evolution is not unitary)
            if self.bqskit_opt is not None:
                trotterized_ev_op = self.bqskit_opt.optimize_evolution(trotterized_ev_op)

        else:
            trotterized_ev_op = evolution_op

        return trotterized_ev_op

    def generate_circuit(self, packed_amplitudes):
        """\
        Construct the circuit for the current parameters

        For the given packed_amplitudes, return the circuit to execute if this
        was a step in a VQE algorithm. The measurements are left out, because
        calculating a single Hamilitonian requires measurement Pauli-strings,
        many of which are independent.

        Args:
            packed_amplitudes(ndarray): compact array storing the unique single
                and double excitation amplitudes for a singlet UCCSD opflow.
                The ordering lists unique single excitations before double
                excitations

        Returns:
            circuit (QuantumCircuit): circuit for the current parameters
        """

      # Build the state preparation evolution operator
        trotterized_ev_op = self.generate_evolution_op(packed_amplitudes)

      # Combine with initializer and evolution
        expect_op = self._expectation.convert(
                            trotterized_ev_op @ self._state_in
                    )

      # Convert to QuantumCircuit
        circuit = (trotterized_ev_op @ self._state_in).to_circuit()

        return circuit

    def __call__(self, packed_amplitudes, use_cached_circuit=None):
        """\
        Calculate the energy expectation for the given parameters

        Args:
            packed_amplitudes(ndarray): compact array storing the unique single
                and double excitation amplitudes for a singlet UCCSD opflow.
                The ordering lists unique single excitations before double
                excitations
            use_cached_circuit(bool): use an existing cached circuit, or cache
                the currently calculated circuit

        Returns:
            energy(float): energy estimate
        """

      # Build the state preparation evolution operator
        global _cached_circuit
        if use_cached_circuit and _cached_circuit is not None:
            trotterized_ev_op = _cached_circuit
        elif isinstance(use_cached_circuit, qk_opflow.operator_base.OperatorBase):
            trotterized_ev_op = use_cached_circuit
        else:
            trotterized_ev_op = self.generate_evolution_op(packed_amplitudes)
            if use_cached_circuit:
                _cached_circuit = trotterized_ev_op

      # Run full simulation. If there are no errors, take a short cut and evaluate
      # the hamiltonian directly. Otherwise, to prevent unrealistic coherent errors,
      # calculate the energy from its components
        if self._simulator is None:
          # exact calculation
            ev_mat = trotterized_ev_op.to_matrix()
            expect_op = self._expectation.convert(
                            self._meas_op @  qk_opflow.MatrixOp(ev_mat) @ self._state_in
                        )
            energy = np.real(expect_op.eval())

        else:
          # sampled calculation from components

          # Note, sampling of the full hamiltonian would look like:
          #     sampled_op = self._simulator.convert(
          #                      self._expectation.convert(
          #                          self._meas_op @ trotterized_ev_op @ self._state_in
          #                      )
          #                  )
          #     energy = np.real(sampled_op.eval())

          # use measurement components to prevent fake coherent errors
            energy = 0.
            for meas_op in self._meas_components:
                sampled_op = self._simulator.convert(
                                 self._expectation.convert(
                                     meas_op @ trotterized_ev_op @ self._state_in
                                 )
                             )
                energy += np.real(sampled_op.eval())

        logger.info('objective: %.5f @ %s', energy, packed_amplitudes)

        if self._save_evals:
          # store parameter values and energy in log file
            f = open(self._save_evals, "a+")
            for ii in range(len(packed_amplitudes)):
                f.write("%f  " % (packed_amplitudes[ii]))
            f.write("%f \n" % (energy))
            f.close()

        return energy


def _to_qiskit(of_qop, n_qubits):
    """Convert OpenFermion QubitOperators to Qiskit equivalent"""

    opflow = list()
    for paulis, coeff in sorted(of_qop.terms.items()):
        ops = ['I']*n_qubits
        for term in paulis:
            ops[term[0]] = term[1]

        ops.reverse()

        opflow1 = coeff*getattr(qk_opflow, ops[0])
        for i in range(1, n_qubits):
            opflow1 ^= getattr(qk_opflow, ops[i])
        opflow.append(opflow1)

    return sum(opflow)


def _hubbard_qubit(x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential = 0.00, magnetic_field = 0.0, periodic = True, spinless = False,
        fermion_transform='bravyi-kitaev'):
    """Create Fermi-Hubbard model with OpenFermion"""

    _fermion_transform = fermion_transform.lower()
    known_transforms = ['jordan-wigner', 'bravyi-kitaev']
    if not _fermion_transform in known_transforms:
        raise ValueError("unknown transform '%s'" % fermion_transform)

    # Hubbard Hamiltonian expressed in FermionOperators, i.e. creation and annihilation
    # operators. Each FermionOperator consists of the site it operates on (expressed as
    # an "index") and whether it raises (1) or lowers (0; expressed asn an "action"),
    # multiplied by a coefficient.
    hubbard_fermion = of.fermi_hubbard(
        x_dimension        = x_dimension,
        y_dimension        = y_dimension,
        tunneling          = tunneling,
        coulomb            = coulomb,
        chemical_potential = chemical_potential,
        magnetic_field     = magnetic_field,
        periodic           = periodic,
        spinless           = spinless)

    # Hubbard Hamiltonian expressed in QubitOperators, i.e. Pauli's (X, Y, Z) operators.
    # Each QubitOperator consists of the qubit it operates on (expressed as an "index")
    # and which Pauli is applied (expressed as an "action") multiplied by a coefficient.
    if _fermion_transform == 'bravyi-kitaev':
        n_qubits = x_dimension * y_dimension * (spinless and 1 or 2)
        hubbard_qubit = of.transforms.bravyi_kitaev(hubbard_fermion, n_qubits)
    elif _fermion_transform == 'jordan-wigner':
        hubbard_qubit = of.transforms.jordan_wigner(hubbard_fermion)

    # Remove terms below floating point epsilon
    hubbard_qubit.compress()

    return hubbard_qubit


def hamiltonian_matrix(x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential = 0.00, magnetic_field = 0.0, periodic = True, spinless = False):
    """Create Fermi-Hubbard model Hamiltonian represented in matrix form"""

    hubbard_qubit = _hubbard_qubit(
        x_dimension        = x_dimension,
        y_dimension        = y_dimension,
        tunneling          = tunneling,
        coulomb            = coulomb,
        chemical_potential = chemical_potential,
        magnetic_field     = magnetic_field,
        periodic           = periodic,
        spinless           = spinless)

    return of.linalg.get_sparse_operator(hubbard_qubit).todense()


def hamiltonian_qiskit(x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential = 0.00, magnetic_field = 0.0, periodic = True, spinless = False,
        fermion_transform='bravyi-kitaev'):
    """Create Fermi-Hubbard model Hamiltonian represented in matrix form"""

    hubbard_qubit = _hubbard_qubit(
        x_dimension        = x_dimension,
        y_dimension        = y_dimension,
        tunneling          = tunneling,
        coulomb            = coulomb,
        chemical_potential = chemical_potential,
        magnetic_field     = magnetic_field,
        periodic           = periodic,
        spinless           = spinless,
        fermion_transform  = fermion_transform)

    n_qubits = x_dimension * y_dimension * (spinless and 1 or 2)

    hubbard_qiskit = _to_qiskit(hubbard_qubit, n_qubits)

  # store the used fermion transform with the hamiltonian to ensure that
  # the objective function later uses the same transform
    hubbard_qiskit._fermion_transform = fermion_transform

    return hubbard_qiskit


class Model(object):
    """Convenience class to capture module parameters"""

    def __init__(self, xdim, ydim, t, U, chem=0.0, mag=0.0, periodic=True, spinless=False, precalc={}):
        self.x_dimension = xdim
        self.y_dimension = ydim
        self.tunneling   = t
        self.coulomb     = U
        self.chemical_potential = chem
        self.magnetic_field     = mag
        self.periodic = periodic
        self.spinless = spinless
        self._precalc = precalc

    def __call__(self):
        """Generate the model"""

        return self.x_dimension, self.y_dimension, self.tunneling, self.coulomb, \
               self.chemical_potential, self.magnetic_field, \
               self.periodic, self.spinless

    def initial(self, n_electrons_up, n_electrons_down, npar, transform='bravyi-kitaev', good=False):
        """\
        Provide a (good) initial and tight bounds for a given configuration

        Args:
            n_electrons_up(int): number of electrons with spin-up
            n_electrons_down(int): number of electrons with spin-down
            transform(str): for which fermion transform the initial applies
            good(bool): whether to return an initial close to the optimal

        Returns:
            initial(tuple): array of (good) initial parameters and an array of bounds
        """

        if good:
            at_opt = self.optimal(n_electrons_up, n_electrons_down, transform)
            if at_opt is not None:
                close = np.round(at_opt, npar <= 4 and 1 or 2)
                bounds = np.zeros((len(close), 2))
                bounds[:,0] = np.subtract(close, 0.1)
                bounds[:,1] = np.add(     close, 0.1)
                return close, bounds

        if npar <= 0:
            raise RuntimeError("not an optimizable configuration (%d parameters)" % npar)

        rng = np.random.default_rng(42)     # for reproducibility while debugging
        initial_amplitudes = np.array(-0.05+0.1*rng.random(size=npar))
        bounds = np.array([(-1.0, 1.0)]*npar)

        return initial_amplitudes, bounds

    def optimal(self, n_electrons_up, n_electrons_down, transform='bravyi-kitaev'):
        """\
        Lookup the pre-calculated optimal paramters

        Args:
            n_electrons_up(int): number of electrons with spin-up
            n_electrons_down(int): number of electrons with spin-down
            transform(str): for which fermion transform the initial applies

        Returns:
            optimum(tuple): array of parameters for the global minimum or None
        """

        try:
            return self._precalc[(n_electrons_up, n_electrons_down)]
        except KeyError:
            pass

        warnings.warn("No pre-calculated initial for configuration (%d, %d)" %\
                      (n_electrons_up, n_electrons_down))

        return None

small_model  = Model(2, 1, t=1.0, U=2.0,
    precalc={
        (1, 0) : np.array([-0.78536064,  0.89994575]),
        (0, 1) : np.array([-0.78536609, -0.25647772]),
        (1, 1) : np.array([-0.86866234,  0.18526051]),
    })

medium_model = Model(2, 2, t=1.0, U=2.0,
    precalc={
        (1, 1) : np.array([ 0.22048886, 0.22048479,  0.27563475,
                            0.22178354, 0.22177972,  0.24547588,
                            0.6276739,  0.60108877,  0.60108406]),
        (2, 2) : np.array([-0.81965099,  0.4858986 , -0.4858995,  0.76993761,
                            0.10298091, -0.03832318, -0.03832113, 0.64542339,
                            0.00399792, -0.00399722,  0.11716964, 0.32792626,
                           -0.06136483,  0.06136485]),
        (3, 3) : np.array([ 0.64501004, -0.6305074 , -0.63050858,
                            0.08473441,  0.06774534,  0.06774663,
                           -0.0411103 , -0.0411079 , -0.01508739])
    })


def exact(hubbard_hamiltonian, n_electrons_up, n_electrons_down):
    """Return the exact solution for the given Hubbard Model Hamiltonian"""

    n_qubits = hubbard_hamiltonian.num_qubits

  # JW has one-to-one mapping and thus the fermion operator's matrix can be used
  # directly; not so for other transformations
    spin_up_op   = sum([of.FermionOperator(((i, 1), (i, 0))) for i in range(0, n_qubits, 2)])
    spin_down_op = sum([of.FermionOperator(((i, 1), (i, 0))) for i in range(1, n_qubits, 2)])

    if hubbard_hamiltonian._fermion_transform == 'bravyi-kitaev':
        spin_up_op   = of.transforms.bravyi_kitaev(spin_up_op, n_qubits)
        spin_down_op = of.transforms.bravyi_kitaev(spin_down_op, n_qubits)
    elif hubbard_hamiltonian._fermion_transform == 'jordan-wigner':
        spin_up_op   = of.transforms.jordan_wigner(spin_up_op)
        spin_down_op = of.transforms.jordan_wigner(spin_down_op)

    spin_up_op   = _to_qiskit(spin_up_op, n_qubits)
    spin_down_op = _to_qiskit (spin_down_op, n_qubits)

  # get the matrix representations; note that the up_matrix will be 2x smaller if
  # n_qubits is even; same for down_matrix if odd
    hm_matrix   = hubbard_hamiltonian.to_matrix()
    up_matrix   = spin_up_op.to_matrix()
    down_matrix = spin_down_op.to_matrix()

    eigenvalues, eigenvectors = spla.eigh(hm_matrix)
    for i in range(hm_matrix.shape[0]):
        v = eigenvectors[:,i]
        n_up = float(np.real(v.T.dot(up_matrix).dot(v)))
        if not round(n_up, 3).is_integer() or not round(n_up) == n_electrons_up:
            continue

        n_down = float(np.real(v.T.dot(down_matrix).dot(v)))
        if not round(n_down, 3).is_integer() or not round(n_down) == n_electrons_down:
            continue

        return float(np.real(v.T.dot(hm_matrix).dot(v)))

    raise RuntimeError('configuration %d up, %d down not found' % (n_electrons_up, n_electrons_down))
