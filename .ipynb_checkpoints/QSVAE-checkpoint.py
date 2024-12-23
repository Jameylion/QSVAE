from src._static.common.helpers import setup_hardware_client, save_nightly_calibration
# from src._static.tutorial.snn_yinyang_helpers import plot_data, plot_input_encoding, plot_training
setup_hardware_client()
from src.QSVAE_model import *
from src.POVM_dataset import *
from src.SNN_brainscales import *
import gc
import torch
import hxtorch
# %matplotlib inline
from src.Quantum_circuits import *
import numpy as np

log = hxtorch.logger.get("grenade.backend")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)
# !python --version

class QSVAE_Params:
    def __init__(self, 
                 I, sigma_x, sigma_y, sigma_z, s_vectors,
                 n, shots, first_run, backend_type, train, test, val,
                 beta, num_steps, num_epochs, learning_rate,
                 batch_train, batch_test, batch_val, num_workers,
                 shuffle, split, device, input_size, hidden_size, 
                 output_size, mock, alpha, result=None, circuits=None, backend=None, probabilities=None):
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
        self.alpha = alpha
        self.load_model = load_model
        self.result = result
        self.circuits = circuits
        self.backend = backend
        self.probabilities = probabilities
    

# Define the Pauli matrices
I = torch.eye(2, dtype=torch.complex64)
sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

# Define the s^(alpha) vectors for the single-qubit POVM
s_vectors = [
    torch.tensor([0, 0, 1], dtype=torch.float32),
    torch.tensor([2 * np.sqrt(2) / 3, 0, -1 / 3], dtype=torch.float32),
    torch.tensor([-np.sqrt(2) / 3, np.sqrt(2) / 3, -1 / 3], dtype=torch.float32),
    torch.tensor([-np.sqrt(2) / 3, -np.sqrt(2) / 3, -1 / 3], dtype=torch.float32)
]

# Parameters
n = 3
shots = 1000
first_run = True
load_model = False
backend_type = "AerSimulator"
train = True
test = False
val = True
beta = 0.819
num_steps = 100
num_epochs = 1
learning_rate = 1e-2
batch_train, batch_test, batch_val = (400, 200, 100)
num_workers = 0
shuffle = False
split = [0.6, 0.2, 100]
input_size = 4 * n
hidden_size = 20 * n
output_size = 2 * 2**n
mock = False
alpha = 1

# result, circuits, backend = None, None, select_backend(backend_type)
# quantum_exp = QuantumExperiment(backend, n, shots)
# result, circuits = quantum_exp.run_experiment()

# Create an instance of QSVAE_Params
params = QSVAE_Params(
    I, sigma_x, sigma_y, sigma_z, s_vectors,
    n, shots, first_run, backend_type, train, test,
    val, beta, num_steps, num_epochs, learning_rate, batch_train,
    batch_test, batch_val, num_workers, shuffle, split, device,
    input_size, hidden_size, output_size, mock, alpha, load_model
)


gc.collect()

try:
    hxtorch.init_hardware() 
except:
    print("hxtorch error")
    hxtorch.release_hardware()
else:
    # Load data
    POVM_dataset = load_data(params)

    # Instantiate the model
    model = SQVAE(params, POVM_dataset)

    # Run the model
    fidelity_score = model.run(params)

    # Release the hardware connection
    hxtorch.release_hardware()