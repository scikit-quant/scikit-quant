import numpy as np
import qiskit.providers.aer.noise as noise
import qiskit.providers.aer.noise.errors as errors

__all__ = [
    'create',
]


def create(p_gate_1q=None, p_gate_2q=None, p_meas=None):
    """\
    Create a noise Qiskit model.

    Args:
        p_gate_1q(float): probability of 1-qubit gate error
        p_gate_2q(float): probability of 2-qubit gate error
        p_meas(float):    probability of measurement error

    Returns:
        noise_model(aer.noise.NoiseModel): Noise model for Aer
    """

    noise_model = noise.NoiseModel()

    if p_meas is not None:
        if p_meas < 0. or 1. < p_meas:
            raise ValueError('measurement error probability should be between 0. and 1.')

      # measurement error is applied to measurements as a confusing matrix,
      # with same readout error on each qubit
        error_meas = errors.readout_error.ReadoutError(
            np.array( ((1.-p_meas, p_meas),
                       (p_meas, 1.-p_meas)) ))
        noise_model.add_all_qubit_readout_error(error_meas)

    if p_gate_1q is not None:
        if p_gate_1q < 0. or 1. < p_gate_1q:
            raise ValueError('1-qubit gate error probability should be between 0. and 1.')

      # single qubit gate error is applied to all non-CNOT gates
        p_gate_1q *= 4./3.        # scale to max-channel (depolarizing_error will undo)
        error_gate1 = errors.depolarizing_error(p_gate_1q, 1)
        basis_gates = list(noise_model.basis_gates)
        try:
            basis_gates.remove('cx')
        except ValueError:
            pass
        noise_model.add_all_qubit_quantum_error(error_gate1, basis_gates)

    if p_gate_2q is not None:
        if p_gate_2q < 0. or 1. < p_gate_2q:
            raise ValueError('2-qubit gate error probability should be between 0. and 1.')

      # two qubit gate error is applied to both qubits of CNOT gates
        p_gate_2q *= 16./15.      # scale to max-channel (depolarizing_error will undo)
        error_gate2 = errors.depolarizing_error(p_gate_2q, 2)
        noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])

    return noise_model
