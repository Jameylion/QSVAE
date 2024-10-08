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

    def __init__(self, measurement_data, n, transform=None):
        self.results = measurement_data
        self.n = n
        self.transform = transform
        self.measurements = self._process_measurements()

    def _process_measurements(self):
        """Processes the measurement data into a usable format."""
        measurements = []
        for r in range(4):
            memory = self.results(r)['memory']
            for mem in memory:
                binary_string = bin(int(mem, 16))[2:].zfill(self.n)
                binary_digit_arrays = [list(map(np.float32, list(b))) for b in binary_string]
                measurements.extend(binary_digit_arrays)
        return np.array(measurements).reshape(-1, 4 * self.n)

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
        val_indices = list(range(split_train + split_test, len(self)))

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
    
def load_data(quantum_exp, first_run, backend, n, shots, split, batch_size, shuffle, num_workers):
    """Loads the quantum dataset, either by running an experiment or loading saved data."""
    if first_run:
        result, circuits = quantum_exp.run_experiment()
        POVM_dataset = QuantumPOVMDataset(result.data, n, transform=transforms.Compose([ToTensor()]))

        with open('POVM_data.pkl', 'wb') as f:
            pickle.dump({'dataset': POVM_dataset, 'circuits': circuits, 'result': result}, f)
            print("Dataset and circuit saved.")
    else:
        with open('POVM_data.pkl', 'rb') as f:
            data = pickle.load(f)
            POVM_dataset = data['dataset']
            print("Dataset loaded.")

    train_loader, test_loader, val_loader = POVM_dataset.split_dataset(split,
                                                                       batch_size,
                                                                       shuffle,
                                                                       num_workers)
    return train_loader, test_loader, val_loader