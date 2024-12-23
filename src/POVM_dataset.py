import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, utils
import pickle
from qiskit.visualization import plot_histogram
from itertools import product

class QuantumPOVMDataset(Dataset):
    """Dataset for Quantum POVM measurements."""

    def __init__(self, params, transform=None):
        self.results =None# measurement_data.data
        self.n = params.n
        self.transform = transform
        self.shots = params.shots
        self.s_vectors = params.s_vectors
        self.I = params.I
        self.sigma_x = params.sigma_x
        self.sigma_y = params.sigma_y
        self.sigma_z = params.sigma_z
        self.n_qubit_povms = self.construct_n_qubit_povms()
        self.probability_true = torch.tensor(self.calculate_probabilities_true(), dtype=torch.float32)
        self.measurements = self._one_hot_encode_probabilities(self.shots)
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        # self.prob_dataset = self._calculate_probabilities()
        # self.one_hot = self._one_hot_encode_measurements(self.measurements)
        # print(self.measurements, self.measurements.shape  ) 
        # print("prob vector calculated from one hot vectors", self.prob_dataset)
        # print("true prob vector", self.probability_true)
        # lt = self.measurements.shape[0]
        # print(lt)
        # print(self.measurements.sum(0)/self.shots)
        # counts = measurement_data.get_counts()
        # for i in range(len(self.measurements)):
        #     print(self.results(0)['memory'][i],
        #           self.results(1)['memory'][i],
        #           self.results(2)['memory'][i],
        #           bin(int(self.results(3)['memory'][i],16))[2:].zfill(self.n))
        #     print(self.measurements[i])


    def _process_measurements(self):
        """Processes the measurement data into a usable format."""
        measurements = []
        for s in range(self.shots):
            bin_per_shot = []
            for c in range(4):
                mem = self.results(c)['memory']
                binary_string = bin(int(mem[s], 16))[2:].zfill(self.n)

                # Convert the binary string into an array of binary digits (0s and 1s)
                binary_digit_array = list(map(int, list(binary_string)))  
                bin_per_shot.extend(binary_digit_array)  # Add the digits to bin_per_shot

            # Append the flattened list of bits for each shot
            measurements.append(bin_per_shot)

        # Convert measurements to a NumPy array
        measurements_array = np.array(measurements, dtype=np.float32)

        # print("Measurements shape:", measurements_array.shape)
        return measurements_array
    
    
    def _one_hot_encode_probabilities(self, samples):
        batch_size = min(10, samples)  # Limit batch size to reduce memory usage
        one_hot_vectors = []
        p = self.probability_true.view([4] * self.n)
        values = torch.arange(0, 4**self.n)
        for _ in range(0, samples, batch_size):
            current_batch_size = min(batch_size, samples - _)
            rand_s = torch.multinomial(p.flatten(), current_batch_size, replacement=True)
            p_index = self._unravel_index(rand_s, p.shape)

            batch_vectors = torch.zeros((current_batch_size, 4 * self.n), dtype=torch.float32)
            for s in range(current_batch_size):
                for q in range(self.n):
                    batch_vectors[s, q * 4 + p_index[q][s]] = 1
            one_hot_vectors.append(batch_vectors)
        return torch.cat(one_hot_vectors, dim=0)
    
    def _unravel_index(self, indices, shape):
        result = []
        for dim in reversed(shape):
            result.append(indices % dim)
            indices = indices // dim
        return tuple(reversed(result))

    def calculate_probabilities_true(self):
        density_matrix = self.ghz_state_density_matrix(self.n)
        n_qubit_povms = self.construct_n_qubit_povms()

        probabilities = []
        for M in n_qubit_povms:
            P = torch.trace(torch.matmul(M, density_matrix)).real.item()
            probabilities.append(P)

        return probabilities

    def ghz_state_density_matrix(self, n):
        state = torch.zeros(2**n, dtype=torch.complex64)
        state[0] = 1 / np.sqrt(2)
        state[-1] = 1 / np.sqrt(2)
        return torch.outer(state, torch.conj(state))

    def construct_n_qubit_povms(self):
        single_qubit_povms = [
            0.25 * (self.I + s[0] * self.sigma_x + s[1] * self.sigma_y + s[2] * self.sigma_z)
            for s in self.s_vectors
        ]
        povm_indices = torch.cartesian_prod(*[torch.arange(4) for _ in range(self.n)])

        n_qubit_povms = []
        for indices in povm_indices:
            povm = single_qubit_povms[indices[0]]
            for i in range(1, self.n):
                povm = torch.kron(povm, single_qubit_povms[indices[i]])
            n_qubit_povms.append(povm)

        return n_qubit_povms

    
    def ghz_state_density_matrix(self, n):
        state = torch.zeros(2**n, dtype=torch.complex64)
        state[0] = 1 / np.sqrt(2)
        state[-1] = 1 / np.sqrt(2)
        return torch.outer(state, torch.conj(state))


    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        sample = {'POVM': self.measurements[idx]}
        if self.transform and isinstance(sample['POVM'], np.ndarray):
            sample = self.transform(sample)
        return sample

    def split_dataset(self, split, batch_size, shuffle=True, num_workers=0):
        """Splits the dataset into training, testing, and validation sets."""
        split_train = int(split[0] * len(self))
        split_test = int(split[1] * len(self))
        split_val = int(split[2] * len(self))

        train_indices = list(range(split_train))
        test_indices = list(range(split_train, split_train + split_test))
        val_indices = list(range(split_val))

        train_set = Subset(self, train_indices)
        test_set = Subset(self, test_indices)
        val_set = Subset(self, val_indices)

        self.train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=shuffle, num_workers=num_workers)
        self.test_loader = DataLoader(test_set, batch_size=batch_size[1], shuffle=shuffle, num_workers=num_workers)
        self.val_loader = DataLoader(val_set, batch_size=batch_size[2], shuffle=shuffle, num_workers=num_workers)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        povm = sample['POVM']
        if isinstance(povm, np.ndarray):
            return {'POVM': torch.from_numpy(povm)}
        return sample

    
# def load_data(result, circuits, first_run, backend, n, shots, split, batch_size, shuffle, num_workers):
#     """Loads the quantum dataset, either by running an experiment or loading saved data."""
#     filename = os.path.join("data", "datasets", f"POVM_data_{n}Qubit_{int(shots)}shots.pkl")
#     if first_run:
#         POVM_dataset = QuantumPOVMDataset(result, n, shots, transform=transforms.Compose([ToTensor()]))
#         with open(filename, 'wb') as f:
#             pickle.dump({'dataset': POVM_dataset, 'circuits': circuits, 'result': result}, f)
#             print("Dataset and circuit saved.")
#     else:
#         with open(filename, 'rb') as f:
#             data = pickle.load(f)
#             POVM_dataset = data['dataset']
#             print("Dataset loaded.")

#     train_loader, test_loader, val_loader = POVM_dataset.split_dataset(split,
#                                                                        batch_size,
#                                                                        shuffle,
#                                                                        num_workers)
#     return train_loader, test_loader, val_loader, POVM_dataset

def load_data(params):
    """Loads the quantum dataset, either by running an experiment or loading saved data."""

    # Define the dataset filename based on parameters
    filename = os.path.join("data", "datasets", f"POVM_data_{params.n}Qubit_{int(params.shots)}shots.pkl")
    
    # If first_run, create the dataset and save it
    if params.first_run:
        POVM_dataset = QuantumPOVMDataset(params,
            transform=transforms.Compose([ToTensor()])
        )
        with open(filename, 'wb') as f:
            pickle.dump({'dataset': POVM_dataset}, f)
            print("Dataset saved.")
            
            # this is for circuit 
            # pickle.dump({'dataset': POVM_dataset, 'circuits': params.circuits, 'result': params.result}, f)
            # print("Dataset and circuit saved.")
    else:
        # Load existing dataset
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            POVM_dataset = data['dataset']
            print("Dataset loaded.")

    # Split dataset into train, test, and validation sets
    POVM_dataset.split_dataset(
        params.split,
        (params.batch_train, params.batch_test, params.batch_val),
        params.shuffle,
        params.num_workers
    )

    return POVM_dataset

