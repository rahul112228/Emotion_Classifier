#------------------------------------------------#
#Normal Torch Vector To Quantum Bits From a Multinomial Distribuiton
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, execute
from qiskit.visualization import plot_histogram
from scipy.linalg import svd

def StateEncoding(idx,qubits=20):
    idx= idx.squeeze(0)
    U, S, Vt = svd(idx, full_matrices=False)
    principal_components = U[:, :qubits]
    qreg = QuantumRegister(qubits)
    creg = ClassicalRegister(qubits)
    circuit = QuantumCircuit(qreg, creg)
    for i in range(qubits):
        for j in range(40):
            circuit.ry(2 * np.arcsin(principal_components[j, i]), qreg[i])
    return circuit

vector = np.random.rand(1, 40, 768)
circuit = StateEncoding(vector)
simulator = Aer.get_backend('qasm_simulator')
job = execute(circuit, simulator, shots=8192) 
result = job.result()
print(result)
counts = result.get_counts(circuit)
print(counts)
plot_histogram(counts)
