import matplotlib.pyplot as plt
# from IPython.display import display, clear_output
import os
import numpy as np
import pandas as pd
import torch

from src.Quantum_circuits import *
from src.QSVAE_model import *
from src.POVM_dataset import *
# from torchvision import transforms, utils
import matplotlib.gridspec as gridspec
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
 dev = "cuda:0"
#  dev = "cpu"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)
torch.__version__

n = 3 # Amount of qubits
shots = 100_000 # amount of shots taken by the quantum simulator
first_run = True
# Support for "Starmon-5" and "AerSimulator" 
backend_type = "AerSimulator"
backend = select_backend(backend_type)
train = True
test = False
val = True

# Define hyperparameters
beta = 0.819
num_steps = 200
num_epochs = 1
learning_rate = 1e-3
batch_train, batch_test, batch_val = (10000, 200, 200)
num_workers = 0
shuffle = False
split = [0.6, 0.2, 4**n *500]

# Reproduction of paper
parameters = [
    (3, 100, 20_000),
    (4, 200, 100_000),
    (5, 500, 4**5 * 500),  # 4^5 * 500
    (6, 600, 4**6 * 500),  # 4^6 * 500
    (7, 800, 4**7 * 500),  # 4^7 * 500
    (8, 1000, 4**8 * 500)  # 4^8 * 500
]

# Run the model for each parameter setting and calculate fidelity:
fidelities = []
for param in parameters:
    n, batch_train, split[2] = param  # Unpack parameters
    result, circuits = None, None
    quantum_exp = QuantumExperiment(backend, n, shots)
    result, circuits = quantum_exp.run_experiment()

    train_loader, test_loader, val_loader, POVM_dataset = load_data(result,
                                                                    circuits,
                                                                    first_run,
                                                                    backend,
                                                                    n,
                                                                    shots,
                                                                    split,
                                                                    [batch_train, batch_test, batch_val],
                                                                    shuffle, num_workers)

    # Instantiate the model for the given parameters
    model = SQVAE(n=n, batch_size=[batch_train, batch_train, batch_val],
                beta=beta, num_steps=num_steps, learning_rate=learning_rate,
                    shots=shots, samples=split[2], device=device, dataset=POVM_dataset, s_vectors = s_vectors)

    # Run the model and get the fidelity for the current parameter setting
    fidelity_score = model.run(train=train, test=test, val=val,
                            train_loader=train_loader, test_loader=test_loader,
                            val_loader=val_loader, num_epochs=num_epochs)
    # Append the fidelity score to the list
    fidelities.append(fidelity_score)

# Plot the histogram of fidelities
plot_histogram(fidelities, parameters)
