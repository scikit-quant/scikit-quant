import hubbard
import logging
import math
import numpy as np
import scipy.linalg as spla

from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA
try:
    from qiskit.algorithms.optimizers import IMFIL
except ImportError:
    print("install scikit-quant to use IMFIL")
from qiskit.circuit.library import RealAmplitudes
from qiskit_nature.circuit.library import UCCSD
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter

logging.getLogger('hubbard').setLevel(logging.INFO)


# optimizer to use
optimizer = SPSA

# Select a model appropriate for the machine used:
#    laptop -> use small model
#    server -> use medium model

MODEL = hubbard.small_model
#MODEL = hubbard.medium_model

# Hubbard model for fermions (Fermi-Hubbard) required parameters
xdim, ydim, t, U, chem, magf, periodic, spinless = MODEL()

# Number of electrons to add to the system
n_electrons_up   = 1
n_electrons_down = 1
n_electrons = n_electrons_up + n_electrons_down

# total number of "sites", with each qubit representing occupied or not
spinfactor = spinless and 1 or 2
n_qubits = n_sites =  xdim * ydim * spinfactor

hubbard_op = hubbard.hamiltonian_qiskit(
    x_dimension        = xdim,
    y_dimension        = ydim,
    tunneling          = t,
    coulomb            = U,
    chemical_potential = chem,
    magnetic_field     = magf,
    periodic           = periodic,
    spinless           = spinless)

# calculate the exact solution
print('expected energy:  %.5f' % hubbard.exact(hubbard_op, n_electrons_up, n_electrons_down))

# create the Ansatz
ansatz = UCCSD(num_particles = (n_electrons_up, n_electrons_down), num_spin_orbitals = n_sites,
               qubit_converter = QubitConverter(JordanWignerMapper()))

# all parameters are angles; note that it may not be a good idea to fully
# restrict the parameters, but this can be diagnosed after the fact by
# checking whether any of the results sits against a boundary
ansatz.parameter_bounds = [(-np.pi, np.pi)]*len(ansatz.parameters)

initial = np.array(-0.05+0.1*np.random.random(size=len(ansatz.parameters)))
vqe = VQE(optimizer        = optimizer(maxiter=1000),
          initial_point    = initial,
          ansatz           = ansatz,
          quantum_instance = Aer.get_backend("qasm_simulator"))
result = vqe.compute_minimum_eigenvalue(hubbard_op)

print("estimated energy: %.5f" % result.optimal_value)
print("parameters:      ", result.optimal_point)
print("# of iters:      ", result.cost_function_evals)
