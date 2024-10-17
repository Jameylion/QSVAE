import torch

from src.Quantum_circuits import *
from src.QSVAE_model import *
from src.POVM_dataset import *

if torch.cuda.is_available():
 dev = "cuda:0"
#  os.environ['CUDA_LAUNCH_BLOCKING']="1" 
#  os.environ['TORCH_USE_CUDA_DSA'] = "1"
#  dev = "cpu"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)
torch.__version__

n = 3 # Amount of qubits
shots = 100_000 # amount of shots taken by the quantum simulator
first_run = False
# Support for "Starmon-5" and "AerSimulator" 
backend_type = "AerSimulator"
backend = select_backend(backend_type)
train = True
test = True
val = True

# Define hyperparameters
beta = 0.819
num_steps = 200
num_epochs = 1
learning_rate = 1e-3
batch_train, batch_test, batch_val = (1000, 200, 100)
num_workers = 0
shuffle = False
split = [0.6, 0.2, 4**n *500]

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
                shots=shots, device=device, dataset=POVM_dataset, s_vectors = s_vectors)

# Run the model and get the fidelity for the current parameter setting
fidelity_score = model.run(train=train, test=test, val=val,
                           train_loader=train_loader, test_loader=test_loader,
                           val_loader=val_loader, num_epochs=num_epochs)


