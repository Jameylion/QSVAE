import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, utils
import pickle

class QuantumPOVMDataset(Dataset):
    """Dataset for Quantum POVM measurements."""

    def __init__(self, measurement_data, n, shots, transform=None):
        self.results = measurement_data.data
        self.n = n
        self.transform = transform
        self.shots = shots
        self.measurements = self._process_measurements()
        self.probability_true = self.measurements.sum(0)/self.shots
        # print(self.measurements  ) 
        # lt = self.measurements.shape[0]
        # print(lt)
        # print(self.measurements.sum(0)/self.shots)
        # counts = measurement_data.get_counts()
        # # for i in range(len(self.measurements)):
        # #     print(self.results(0)['memory'][i],
        # #           self.results(1)['memory'][i],
        # #           self.results(2)['memory'][i],
        # #           bin(int(self.results(3)['memory'][i],16))[2:].zfill(self.n))
        # #     print(self.measurements[i])

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
        split_val = len(self) - split_train - split_test

        train_indices = list(range(split_train))
        test_indices = list(range(split_train, split_train + split_test))
        val_indices = list(range(split_val))

        train_set = Subset(self, train_indices)
        test_set = Subset(self, test_indices)
        val_set = Subset(self, val_indices)

        train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size[1], shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size[2], shuffle=shuffle, num_workers=num_workers)

        return train_loader, test_loader, val_loader

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        povm = sample['POVM']

        return {'POVM': torch.from_numpy(povm)}
    
def load_data(result, circuits, first_run, backend, n, shots, split, batch_size, shuffle, num_workers):
    """Loads the quantum dataset, either by running an experiment or loading saved data."""
    if first_run:
        POVM_dataset = QuantumPOVMDataset(result, n, shots, transform=transforms.Compose([ToTensor()]))
        with open(f'data\datasets\POVM_data_{n}Qubit_{int(shots)}shots.pkl', 'wb') as f:
            pickle.dump({'dataset': POVM_dataset, 'circuits': circuits, 'result': result}, f)
            print("Dataset and circuit saved.")
    else:
        with open(f'data\datasets\POVM_data_{n}Qubit_{int(shots)}shots.pkl', 'rb') as f:
            data = pickle.load(f)
            POVM_dataset = data['dataset']
            print("Dataset loaded.")

    train_loader, test_loader, val_loader = POVM_dataset.split_dataset(split,
                                                                       batch_size,
                                                                       shuffle,
                                                                       num_workers)
    return train_loader, test_loader, val_loader, POVM_dataset