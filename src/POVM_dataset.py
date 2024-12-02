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

    def __init__(self, measurement_data, n, shots, probabilities, transform=None):
        self.results = measurement_data.data
        self.n = n
        self.transform = transform
        self.shots = shots
        self.probability_true = probabilities
        self.measurements = self._one_hot_encode_probabilities(self._process_measurements())
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.prob_dataset = self._calculate_probabilities()
        # self.one_hot = self._one_hot_encode_measurements(self.measurements)
        # print(self.measurements, self.measurements.shape  ) 
        print("prob vector calculated from one hot vectors", self.prob_dataset)
        print("true prob vector", self.probability_true)
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
    
    
    def _one_hot_encode_probabilities(self, m):
        # Store one-hot encoded vectors for each shot
        one_hot_vectors = []
        p = np.asarray(self.probability_true).reshape([4] * self.n)
        values = np.arange(0, 4**self.n) 
        rand_s = np.random.choice(values, size=self.shots, p=self.probability_true)
        p_index = np.unravel_index(rand_s, p.shape)
        print(p_index)       
        for s in range(self.shots):
            one_hot_vector = []     
            for q in range(self.n):
                v = np.zeros(4)       
                v[p_index[q][s]] =  1
                one_hot_vector.extend(v)
            one_hot_vectors.append(one_hot_vector)
            # print(f"Shots sample m: {m[s,:]} is converted to one hot vector {one_hot_vector}")
        # print(one_hot_vectors)

        one_hot_vectors_array = np.array(one_hot_vectors, dtype=np.float32)
        print(one_hot_vectors_array, one_hot_vectors_array.shape)
        return one_hot_vectors_array

    def _calculate_probabilities(self):
        prob = np.zeros([4] * self.n)
        for s in range(self.shots):
            index = []
            for q in range(self.n):
                # print(self.measurements[s, (q * 4): q + (q * 4) + 4])
                index.append(np.argmax(self.measurements[s, (q * 4): q + (q+1)*4]))
            # print(index)
            prob[tuple(index)] += 1
        prob = prob.reshape(-1)
        prob = prob/self.shots
        return prob

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        sample = {'POVM': self.measurements[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def split_dataset(self, split, batch_size, shuffle=True, num_workers=0):
        """Splits the dataset into training, testing, and validation sets."""
        split_train = int(split[0] * len(self))
        split_test = int(split[1] * len(self))
        split_val = split[2]

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

        return {'POVM': torch.from_numpy(povm)}
    
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
        POVM_dataset = QuantumPOVMDataset(
            probabilities=params.probabilities,
            measurement_data=params.result,
            n=params.n,
            shots=params.shots,
            transform=transforms.Compose([ToTensor()])
        )
        with open(filename, 'wb') as f:
            pickle.dump({'dataset': POVM_dataset, 'circuits': params.circuits, 'result': params.result}, f)
            print("Dataset and circuit saved.")
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
