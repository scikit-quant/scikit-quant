import bqskit
import numpy as np
import qiskit as qk
import logging
import time

from bqskit.compiler import CompilationTask, Compiler
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost import HilbertSchmidtCost
from bqskit.ir.gates import CNOTGate
from bqskit.passes.control import WhileLoopPass, ForEachBlockPass
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.processing import ScanningGateRemovalPass
from bqskit.passes.synthesis import LEAPSynthesisPass, QFASTDecompositionPass, SynthesisPass
from bqskit.passes.synthesis.qpredict import QPredictDecompositionPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.util import SetRandomSeedPass, UnfoldPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

__all__ = [
    'BQSKit_Hubbard_Optimizer',
]

#logging.getLogger("bqskit").setLevel(logging.DEBUG)


def shorter_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return circuit.count(bqskit.ir.gates.CNOTGate()) < op.gate._circuit.count(bqskit.ir.gates.CNOTGate())

class PrintCNOTsPass(BasePass):
    def run(self, circuit, data) -> None:
        print("BQSKit step, current CNOT count:", circuit.count(bqskit.ir.gates.CNOTGate()))


class BQSKit_Hubbard_Optimizer:
    def __init__(self, full=False) -> None:
        self._prev_template = None
        self._full = full

    def optimize_evolution(self, evolution_op):
        """\
        Optimize evolution_op with bqskit.

        Run a representative set of BQSKit optimizer passes on the evolution_op.
        If 'full' was requested in the constructor, run until converged. Otherwise,
        run passes a single time, then break.

        Args:
            evolution_op(opflow): input unitary

        Returns:
            qiskit_circuit(opflow): optimized unitary as opflow circuit
        """

        qiskit_circuit = evolution_op.to_circuit()

        t = time.time()
        qiskit_circuit = qk.transpile(qiskit_circuit, basis_gates=["u3", "cx"])
        bqskit_circuit = OPENQASM2Language().decode(qiskit_circuit.qasm())

        if self._prev_template is not None:
            target_unitary = bqskit_circuit.get_unitary()
            self._prev_template.instantiate(target_unitary)
            if self._prev_template.get_unitary().get_distance_from(target_unitary, 1) < 1e-8:
                print("BQSKit completed from template in %.2fs" % (time.time() - t))
                qiskit_circuit = qk.QuantumCircuit.from_qasm_str(OPENQASM2Language().encode(self._prev_template))
                return qk.opflow.primitive_ops.CircuitOp(qiskit_circuit)

        if not self._full:
            # run only a single pass
            bqskit_task = CompilationTask(bqskit_circuit, [PrintCNOTsPass(),
                ClusteringPartitioner(3, 8),
                ForEachBlockPass([QSearchSynthesisPass(), ScanningGateRemovalPass()], replace_filter=shorter_replace_filter),
                UnfoldPass(),
                PrintCNOTsPass()
            ])
        else:
            # run until converged
            bqskit_task = CompilationTask(bqskit_circuit, [PrintCNOTsPass(),
            WhileLoopPass(
                 GateCountPredicate(bqskit.ir.gates.CNOTGate()),
                 [
                 ClusteringPartitioner(3, 8),
                 ForEachBlockPass([QSearchSynthesisPass(), ScanningGateRemovalPass()], replace_filter=shorter_replace_filter),
                 UnfoldPass(),
                 PrintCNOTsPass()
                 ],
            )])
        with Compiler() as compiler:
            new_circuit = compiler.compile(bqskit_task)

        self._prev_template = new_circuit

        print("BQSKit completed in %.2fs" % (time.time() - t))
        qiskit_circuit = qk.QuantumCircuit.from_qasm_str(OPENQASM2Language().encode(new_circuit))
        return qk.opflow.primitive_ops.CircuitOp(qiskit_circuit)
