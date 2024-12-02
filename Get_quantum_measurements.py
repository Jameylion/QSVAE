import torch
from src.Quantum_circuits import *
import numpy as np
from src.QSVAE_model import *
from src.POVM_dataset import *
# from src.SNN_brainscales import *


if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)

class QSVAE_Params:
    def __init__(self, 
                 I, sigma_x, sigma_y, sigma_z, s_vectors,
                 n, shots, first_run, backend_type, train, test, val,
                 beta, num_steps, num_epochs, learning_rate,
                 batch_train, batch_test, batch_val, num_workers,
                 shuffle, split, device, input_size, hidden_size, 
                 output_size, mock, result=None, circuits=None, backend=None, probabilities=None):
        self.I = I
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.s_vectors = s_vectors
        self.n = n
        self.shots = shots
        self.first_run = first_run
        self.backend_type = backend_type
        self.train = train
        self.test = test
        self.val = val
        self.beta = beta
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_train = batch_train
        self.batch_test = batch_test
        self.batch_val = batch_val
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.split = split
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mock = mock
        self.load_model = load_model
        self.result = result
        self.circuits = circuits
        self.backend = backend
        self.probabilities = probabilities
    

# Define the Pauli matrices
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Define the s^(alpha) vectors for the single-qubit POVM
s_vectors = [
    np.array([0, 0, 1]),  # s^(0)
    np.array([2 * np.sqrt(2) / 3, 0, -1 / 3]),  # s^(1)
    np.array([-np.sqrt(2) / 3, np.sqrt(2) / 3, -1 / 3]),  # s^(2)
    np.array([-np.sqrt(2) / 3, -np.sqrt(2) / 3, -1 / 3])  # s^(3)
]

# Parameters
n = 2
shots = 100_000
first_run = True
load_model = False
backend_type = "AerSimulator"
train = True
test = False
val = True
beta = 0.819
num_steps = 100
num_epochs = 3
learning_rate = 1e-3
batch_train, batch_test, batch_val = (100, 200, 100)
num_workers = 0
shuffle = False
split = [0.01, 0.2, 100]
input_size = 4 * n
hidden_size = 32 * n
output_size = 2 * 2**n
mock = False

result, circuits, backend = None, None, select_backend(backend_type)


# Create an instance of QSVAE_Params
params = QSVAE_Params(
    I, sigma_x, sigma_y, sigma_z, s_vectors,
    n, shots, first_run, backend_type, train, test,
    val, beta, num_steps, num_epochs, learning_rate, batch_train,
    batch_test, batch_val, num_workers, shuffle, split, device,
    input_size, hidden_size, output_size, mock, load_model
)


for n in range(2, 4):
    params.n = n
    quantum_exp = QuantumExperiment(backend, params.n, params.shots)
    params.result, params.circuits = quantum_exp.run_experiment()
    params.probabilities = quantum_exp.probabilities

    POVM_dataset = load_data(params)

    model = SQVAE(params, POVM_dataset)


