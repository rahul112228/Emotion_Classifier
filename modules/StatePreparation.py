import pennylane as qml
import torch
import numpy as np

def quantum_embedding(data):
    data_np = data.numpy()
    batch_size, variable_dim, feature_dim = data_np.shape
    dev = qml.device("lightning.qubit", wires=variable_dim)
    @qml.qnode(dev)
    def circuit(data_flat):
        for i in range(variable_dim):
            qml.AmplitudeEmbedding(data_flat[i], wires=i)
        return qml.state()

    quantum_states = []

    for i in range(batch_size):
        data_flat = (data_np[i] - np.min(data_np[i])) / (np.max(data_np[i]) - np.min(data_np[i]))
        state = circuit(data_flat)
        quantum_states.append(state)

    return quantum_states

# Assuming you have your data tensor defined
data = torch.randn(1, 10, 768)  # You can change the size of the second dimension as needed

# Get quantum states for each data point in the batch
quantum_states = quantum_embedding(data)

# 'quantum_states' is a list containing quantum states for each data point in the batch
print(len(quantum_states))