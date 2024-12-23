import torch
import torch.nn as nn
from tqdm import tqdm
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from torch.nn import DataParallel
# from src.Quantum_circuits import *
from itertools import product
from scipy.linalg import sqrtm
from src.SNN_brainscales import *
import snntorch.functional as SF
import qiskit
import torch.nn.functional as F
import gc

import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
import itertools

# Define the Pauli matrices
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

from torch.autograd import Function

import csv
from datetime import datetime
# Dynamically generate the filename with the current date
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"training_metrics_{current_date}.csv"

def write_to_csv(filename, row_index, data):
    """
    Writes metrics to a specific row in a CSV file.
    Appends new rows if the file has fewer rows than `row_index + 1`.
    """
    # Read the existing CSV data
    try:
        with open(filename, mode='r') as f:
            reader = list(csv.reader(f))
    except FileNotFoundError:
        reader = []  # Initialize as empty if file doesn't exist

    # Ensure enough rows exist to write at `row_index`
    while len(reader) <= row_index:
        reader.append([])

    # Update the specific row with data
    reader[row_index] = data

    # Write back to the CSV file
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reader)

def save_progress(epoch, iteration, filename="progress.json"):
    with open(filename, "w") as f:
        json.dump({"epoch": epoch, "iteration": iteration}, f)

def load_progress(filename="progress.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"epoch": 0, "iteration": 0}

class SurrogateGradientFunction(Function):
    @staticmethod
    def forward(ctx, r, u, alpha):
        ctx.save_for_backward(r, u)
        ctx.alpha = alpha
        return (u < r.unsqueeze(0)).float()

    @staticmethod
    def backward(ctx, grad_output):
        r, u = ctx.saved_tensors
        alpha = ctx.alpha
        gradient = (torch.abs(r - u) < (alpha / 2)).type(torch.float32) / alpha
        grad_input = gradient * grad_output
        return grad_input, None, None

class Model(torch.nn.Module):
    """ Complete model with encoder (SNN on CPU) and decoder(SNN on Brainscales) """

    def __init__(self, encoder, decoder, params):
        """
        Initialize the model by assigning encoder, network and decoder
        :param encoder: Module to encode input data
        :param network: Network module containing layers and
            parameters / weights
        :param decoder: Module to decode network output
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_z = None
        self.device = params.device
        self.outputs = params.output_size
        # self.readout_scale = params.readout_scale
        self.bottleneckfc = nn.Linear(self.outputs, self.outputs)
        self.bottlenecksm = nn.Sigmoid()
        self.params = params

    def reparameterization(self, mean, var):
      std = torch.sqrt(var).to(self.device)
      eps = torch.randn(std.shape).to(self.device)

      return mean + eps * std
    
    def firing_rate(self, spk):
      #  print(spk.shape)
      #  print((spk.sum(0)/spk.shape[0]).shape)
       return spk.sum(0)/spk.shape[0]

    def loss_function(self, x_hat, x, mean, log_var):
        # Calculate reconstruction loss (example: MSE)
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        
        # Calculate KL divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Combine the losses
        total_loss = reconstruction_loss + kl_divergence
        return total_loss
    
    def kl_divergence(self, P, Q):
        # print(P, Q)
        # Ensure no zeros in Q to avoid log(0) issues (replace small values)
        Q = np.where(Q == 0, 1e-10, Q)

        # Compute KL divergence (element-wise calculation)
        kl_div = np.sum(P * np.log(P / Q))

        return kl_div

    def calc_mean_var(self, spk):
        spk = spk.sum(0)
        # Calculate mean and variance across time steps
        mean = spk.mean(dim=0)  # Shape: [batch_size, output_size]
        # print(mean)

        variance = spk.var(dim=0, unbiased=False)  # Shape: [batch_size, output_size]
        # print(variance)
        # print("spike mean shape per outpu neuron\n", mean.shape)
        del spk

        return mean, variance
    
    def reparam_poisson(self, r, spk):
        '''
        Args: firing rate, spikes. size: [batch, input/outputs], [steps, batch, inputs/outputs]
        Return: latent space z [steps, batch, inputs/outputs]
        '''
        # print(r.shape)
        # print(spk.shape)
        u = torch.rand(spk.shape).to(self.device)
        
        z = SurrogateGradientFunction.apply(r, u, self.params.alpha)
        # print(z.shape)
        return z
    
    def rbf_kernel(self, x, y, sigma=1.0):
        # Compute the squared Euclidean distance between x and y
        x_norm = (x ** 2).sum(1).reshape(-1, 1)
        y_norm = (y ** 2).sum(1).reshape(1, -1)
        distance = x_norm + y_norm - 2 * torch.mm(x, y.T)
    
        # Compute the RBF kernel
        return torch.exp(-distance / (2 * sigma ** 2))
    
    def mmd_rbf_loss(self, x_samples, y_samples, sigma=1.0):
        # Compute the RBF kernels
        k_xx = self.rbf_kernel(x_samples, x_samples, sigma)
        k_yy = self.rbf_kernel(y_samples, y_samples, sigma)
        k_xy = self.rbf_kernel(x_samples, y_samples, sigma)

        # Calculate the MMD loss
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return mmd

    def MMDKLLoss(self, r_p, r_q, x_hat, x, size):
        
        x_hat = x_hat/self.params.num_steps

        # reconstruction_loss = nn.BCELoss()(x_hat, x)
        reconstruction_loss = 0
            
        # Compute the MMD loss between two sets of samples (e.g., r_p and r_q)
        mmd_loss = self.mmd_rbf_loss(r_p, r_q, sigma=1.0)

        return reconstruction_loss, mmd_loss
       
    def forward(self, x):
        spk, mem = self.encoder(x)
        mean, log_var = self.calc_mean_var(spk)
        # self.plot_spikes(spk, mem, cur)
        # shape skp = [num_steps, batch_size, input_size]
        # torch.set_printoptions(profile="full")
        # print(self.firing_rate(spk)) # prints the whole tensor
        # torch.set_printoptions(profile="default") # reset
        # print("encoder out spk", spk.sum((0,1)))
        
        r_p = self.firing_rate(spk)
        # print("firing rate = ", r_p)
        # print("z = ", self.reparam_poisson(r, spk))
        # z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        z = self.reparam_poisson(r_p, spk)
        
        # print(z.shape)

        spk, mem = self.decoder(z)

        # del z
        torch.cuda.empty_cache()
        # print(type(x_hat))

        return spk, mem, mean, log_var, r_p, z

    
# @title Class Encoder
class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.lif1 = snn.Leaky(beta=params.beta)
        self.fc2 = nn.Linear(params.hidden_size, params.output_size)
        self.lif2 = snn.Leaky(beta=params.beta)
        self.num_steps = params.num_steps
        self.outputs = params.output_size
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            del spk1, cur1, cur2
        del mem1, mem2

        # After the loop, you can also clear Python's memory
        gc.collect()

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc1 = nn.Linear(params.output_size, params.hidden_size)
        self.lif1 = snn.Leaky(beta=params.beta)
        self.fc2 = nn.Linear(params.hidden_size, params.input_size)
        self.lif2 = snn.Leaky(beta=params.beta)
        self.num_steps = params.num_steps


    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

            del spk1, spk2, cur1, cur2
        del mem1, mem2


        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class SQVAE():
    def __init__(self, params, POVM_dataset):
        self.n = params.n
        self.batch_size = (params.batch_train, params.batch_test, params.batch_val)
        self.beta = params.beta
        self.num_steps = params.num_steps
        self.learning_rate = params.learning_rate
        self.device = params.device
        self.first_run = params.first_run
        self.dataset = POVM_dataset
        self.s_vectors = params.s_vectors
        self.samples = params.batch_val
        self.inputs = params.input_size
        self.hidden = params.hidden_size
        self.outputs = params.output_size
        self.shots = params.shots
        self.num_epochs = params.num_epochs
        # Define bottleneck

        
        self.MOCK          = False
        self.DT            = 2.0e-06  # s

        # Initialize encoder and decoder as part of the model
        self.encoder = Encoder(params).to(self.device)
        # if not params.first_run:
        self.decoder = SNN(
                n_in=params.output_size,
                n_hidden=params.hidden_size,
                n_out=params.input_size,
                mock=params.mock,
                calib_path="spiking2_cocolist.pbin",
                dt=self.DT,
                tau_mem=1.0e-06, #6.0e-6
                tau_syn=6.0e-06,
                alpha=300.,
                trace_shift_hidden=int(.0e-06/self.DT),
                trace_shift_out=int(.0e-06/self.DT),
                weight_init_hidden=(0.2, 0.6),
                weight_init_output=(0.4, 0.6),
                weight_scale=100, #66.39
                trace_scale=0.0147,
                input_repetitions=1 if self.MOCK else 1,
                device=params.device
            )
        # else:
        #     self.decoder = Decoder(params).to(self.device)

        # Combine encoder and decoder into the model
        self.model = Model(self.encoder, self.decoder, params).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # # If this is the first run, save the initial model state
        # if not params.load_model:
        #     torch.save(self.model.state_dict(), f"data/models/model_{params.n}qubit_{int(params.shots)}shots.pt")
        # else:
        #     self.model.load_state_dict(torch.load( f"data/models/model_{params.n}qubit_{int(params.shots)}shots.pt", map_location=torch.device(params.device)))

        # # Check for multiple GPUs and use DataParallel
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs for training.")
        #     self.model = DataParallel(self.model)


    def run(self, params):
        """Run the model with options for training, testing, and validation."""
        fidelity_score = 0
        if params.train:
            # Initialize TensorBoard writer for training
            writer = SummaryWriter(log_dir='runs/test_case_greenlight')
            self.train_model(self.dataset.train_loader,self.optimizer, self.device, writer)
            torch.save(self.model.state_dict(), f"data/models/model_{self.n}qubit_{int(self.shots)}shots.pt")

        if params.test:
            # Initialize TensorBoard writer for testing
            writer = SummaryWriter(log_dir='runs/test_experiment')
            test_loss = self.test_model(self.dataset.test_loader, self.device, writer)
            print("Average test loss: ", test_loss)

        if params.val:
            spikes = self.val_model()
            prob_rec = self.calculate_prob_from_spikes(spikes, self.batch_size[2])          
            prob_true = self.dataset.probability_true
            print(f"prob rec = {prob_rec}, prob_true = {prob_true}")
            rho_rec, rho_true = self.reconstruct_matrix_from_prob(prob_rec, prob_true)
            # fidelity_score = fidelity(rho_rec, rho_true)
            fidelity_score = fid(torch.FloatTensor(prob_true), torch.from_numpy(prob_rec))
            print(f"The fidelity for {self.n} qubits is {fidelity_score} with {self.samples} samples.")

        return fidelity_score

    def train_model(self, dataloader, optimizer, device, writer):
        self.model.train()
        
        # Load previous progress
        progress = load_progress()
        start_epoch = progress.get("epoch", 0)
        start_iteration = progress.get("iteration", 0)
        print("Start epoch: ", start_epoch)
        print("start_iteration:  ", start_iteration )

        # Early stopping parameters
        patience = 10  # Number of epochs to wait before stopping if no improvement
        best_loss = float('inf')  # Initialize the best loss to a large value
        epochs_no_improve = 0  # Counter for epochs without improvement

        # Initialize list for storing loss values
        train_loss = []
        fidelity_score = 0
        
        size = torch.Tensor(self.num_steps, self.batch_size[2], self.outputs)
        batch_size = self.batch_size[0]
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = []
            data = iter(dataloader)
            sliced_data = itertools.islice(data, start_iteration, None)
            
            for batch_idx, sample_batched in enumerate(tqdm(sliced_data, desc="Iterations", initial=start_iteration, total=len(data)-start_iteration), start=start_iteration):
                # print("batch adn epoch", batch_idx, epoch)
            # for batch_idx, sample_batched in enumerate(tqdm(data, start=start_iteration)):
                save_progress(epoch, batch_idx)
            # for batch_idx in range(0, self.shots, batch_size):
                sample_batched = sample_batched['POVM'].to(device)
                for param in self.model.parameters():
                    param.grad = None

                # Forward pass
                sample_batched = sample_batched.to(device)
                spk, mem, mean, log_var, r_p, z = self.model(sample_batched)

                # Calculate loss
                z_n = torch.randn(batch_size, self.outputs, dtype = torch.float32, device = self.device)
                r_q = self.model.bottlenecksm(self.model.bottleneckfc(z_n))
                
                # mseklloss = self.model.loss_function(spk.sum(0), sample_batched, mean, log_var)
                sample_batched = sample_batched.repeat(spk.shape[0], 1, 1)
                reconstruction_loss, mmd_loss = self.model.MMDKLLoss(r_p, r_q, spk, sample_batched, size)
                mse_loss = nn.MSELoss()(spk.sum(0), sample_batched.sum(0))     
                # mse_loss = nn.MSELoss()(spk.mean(0), sample_batched.mean(0))
                # mse_loss = nn.MSELoss()(spk.sum(0) / spk.size(0), sample_batched.sum(0) / sample_batched.size(0))
                # r_s = self.model.firing_rate(sample_batched)
                # r_o = self.model.firing_rate(spk)
                # kl_loss = self.model.kl_divergence(self.dataset.probability_true, self.calculate_prob_from_spikes(spk.detach())) 
                # reconstruction_loss = SF.ce_count_loss(population_code=True,num_classes=self.n)(spk,sample_batched)
                
                # Combine the losses with a balancing coefficient lambda_mmd
                lambda_mmd = 1
                lamdba_kl = 0.01
                lambda_fid = 1                
                loss = 0
                # loss += mseklloss
                # print(f" mseklloss is {torch.round(mseklloss, decimals=3)}")
                loss = mse_loss               
                # loss += lamdba_kl * kl_loss 
                # print(f" kl_loss is {round(lamdba_kl * kl_loss , 3)}")
                loss += mmd_loss * lambda_mmd
                # loss += lambda_fid * (1-fidelity_score)
                prob_rec = self.calculate_prob_from_spikes(spk.detach(), batch_size)          
                prob_true = self.dataset.probability_true
                # print(f"prob rec = {prob_rec}, prob_true = {prob_true}")
                rho_rec, rho_true = self.reconstruct_matrix_from_prob(prob_rec, prob_true)
                # fidelity_score = fidelity(rho_rec, rho_true)
                fidelity_score = fid(torch.FloatTensor(prob_true), torch.from_numpy(prob_rec))
                tot_dim = spk.shape[0] * spk.shape[1] * spk.shape[2]
                e_s = spk.sum((0,1,2))/tot_dim *100
                z_s = z.sum((0,1,2))/tot_dim *100
    
                # Print metrics each some iterations
                if batch_idx % 10 == 0:    

                    print(f" mse_loss is {torch.round(mse_loss, decimals=3)}")
                    print(f" mmd_loss is {torch.round(mmd_loss * lambda_mmd , decimals=3)}")
                    print(f" total loss = {torch.round(loss,decimals=3)}")              
                    print(f"The fidelity for {self.n} qubits is {torch.round(fidelity_score,decimals=3)} with {batch_size} samples.")
                    
                    
                    print(f"Spike count encoder = {spk.sum((0,1,2))} = {torch.round((e_s), decimals=2)}%, spike count latent z = {z.sum((0,1,2))} = {torch.round((z_s),decimals=2)}% ") 
                    # writer.add_scalar('Spike count encoder%-train', e_s.item(), epoch * len(dataloader) + batch_idx)
                    # writer.add_scalar('Spike count latent z%-train', z_s.item(), epoch * len(dataloader) + batch_idx)
                    # writer.add_scalar('Fidelity-train', fidelity_score.item(), epoch * len(dataloader) + batch_idx) 
                    # torch.save(self.model.state_dict(), "data/model_recover_point.pt")


            #            
                loss.backward()
                optimizer.step()
                lr = self.model.params.learning_rate
                self.model.params.learning_rate = lr/(1 + (epoch * 10 + batch_idx)  / 0.0001)

                # Append current loss
                epoch_loss.append(loss.item())
                
                
                # # Log loss to TensorBoard after each batch
                # writer.add_scalar('mse_loss-train', mse_loss.item(), epoch * len(dataloader) + batch_idx)
                # writer.add_scalar(f'mmd_loss/{lambda_mmd}-train', mmd_loss.item(), epoch * len(dataloader) + batch_idx)
                # # writer.add_scalar('kl_loss/train', kl_loss.item(), epoch * len(dataloader) + batch_idx)
                # writer.add_scalar('Loss-train', loss.item(), epoch * len(dataloader) + batch_idx)
                #            # Calculate metrics
                metrics = {
                    'Spike count encoder%-train': e_s.item(),
                    'Spike count latent z%-train': z_s.item(),
                    'Fidelity-train': fidelity_score.item(),
                    'mse_loss-train': mse_loss.item(),
                    f'mmd_loss/{lambda_mmd}-train': mmd_loss.item(),
                    'Loss-train': loss.item(),
                    'Fidelity-val': fidelity_score.item()
                }

                # Calculate row index
                row_index = (epoch * len(dataloader)) + batch_idx

                # Write metrics to CSV
                write_to_csv(csv_filename, row_index, [row_index] + list(metrics.values()))

                # Free up memory
                del spk, mem, mean, log_var, loss, z, sample_batched, r_q, z_n
                torch.cuda.empty_cache()
                gc.collect() 
                writer.flush()

                # Log memory usage if on GPU
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_cached = torch.cuda.memory_reserved(device)

                    # Log memory usage to TensorBoard
                    writer.add_scalar('Memory/Allocated_MB', memory_allocated / (1024 ** 2), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar('Memory/Cached_MB', memory_cached / (1024 ** 2), epoch * len(dataloader) + batch_idx)
                    
#             spikes = self.val_model()
#             prob_rec = self.calculate_prob_from_spikes(spikes, self.samples)    
#             print("sum prob rec ", prob_rec.sum(0))
#             prob_true = self.dataset.probability_true
#             print(f"prob rec = {prob_rec}, prob_true = {prob_true}")
#             rho_rec, rho_true = self.reconstruct_matrix_from_prob(prob_rec, prob_true)
#             # fidelity_score = fidelity(rho_rec, rho_true)
#             fidelity_score = fid(torch.FloatTensor(prob_true), torch.from_numpy(prob_rec))
#             print(f"The fidelity for {self.n} qubits is {fidelity_score} with {self.samples} samples.")
            
#             del spikes

            start_iteration = 0        
            # Calculate and store the average loss for the epoch
            avg_epoch_loss = sum(epoch_loss) / self.shots
            print(f"Epoch {epoch+1}/{self.num_epochs}, average Loss: {avg_epoch_loss:.4f}")
            train_loss.append(avg_epoch_loss)

            # Log average epoch loss to TensorBoard
            writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

#             # Check for early stopping
#             if avg_epoch_loss < best_loss:
#                 best_loss = avg_epoch_loss
#                 epochs_no_improve = 0
#                 # Save the best model
#                 torch.save(self.model.state_dict(), "data/best_model.pt")
#             else:
#                 epochs_no_improve += 1
#                 print(f"No improvement for {epochs_no_improve} epochs")

#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered")
#                 # Load the best model state before stopping
#                 self.model.load_state_dict(torch.load("data/best_model.pt"))
#                 break

        # Finalize TensorBoard logging
        writer.flush()
        writer.close()

        return self.model

    def test_model(self, dataloader, device, writer):
        """
        Function to test the model on the test dataset and log the loss and memory usage to TensorBoard.

        Args:
            model: The trained model to be tested.
            dataloader: DataLoader for the test dataset.
            device: The device (CPU or GPU) to perform testing on.

        Returns:
            avg_test_loss: The average test loss over the test dataset.
        """
        torch.cuda.empty_cache()

        self.model.eval()  # Set the model to evaluation mode
        test_loss = []

        data = iter(dataloader)
        for batch_idx, sample_batched in enumerate(tqdm(data)):
            sample_batched = sample_batched['POVM'].to(device)

            with torch.no_grad():  # Disable gradient calculation for testing
                spk, mem, mean, log_var = self.model(sample_batched)
                loss = self.model.loss_function(spk.sum(0), sample_batched, mean, log_var)

                # Log test loss to TensorBoard after each batch
                writer.add_scalar('Loss/test', loss.item(), batch_idx)

                # Append the loss for the current batch
                test_loss.append(loss.item())

                # Log memory usage if on GPU
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_cached = torch.cuda.memory_reserved(device)

                    # Log memory usage to TensorBoard
                    writer.add_scalar('Memory/Allocated_MB', memory_allocated / (1024 ** 2), batch_idx)
                    writer.add_scalar('Memory/Cached_MB', memory_cached / (1024 ** 2), batch_idx)

                # Free up memory
                del spk, mem, mean, log_var, loss
                torch.cuda.empty_cache()

        # Calculate and return the average test loss over the dataset
        avg_test_loss = sum(test_loss) / len(dataloader)

        # Log average test loss to TensorBoard
        writer.add_scalar('Loss/avg_test_loss', avg_test_loss)

        # Finalize TensorBoard logging
        writer.flush()
        writer.close()

        return avg_test_loss
    
    def val_model(self):
        """
        Function to validate the model on the val dataset.

        Args:
            model: The trained model to be tested.
            dataloader: DataLoader for the test dataset.
            device: The device (CPU or GPU) to perform testing on.

        Returns:
            probabilities: probabilities per batch.
        """
        torch.cuda.empty_cache()

        self.model.eval()  # Set the model to evaluation mode
        spikes = torch.Tensor()
       

        # data = iter(dataloader)
        # for batch_idx, sample_batched in enumerate(tqdm(data)):
        batch_size = self.batch_size[0]
        
        for i in tqdm(range(0, self.samples, batch_size)):
            
            # Random samples from N(0, I)
            samples_n = torch.randn(batch_size, self.outputs, dtype = torch.float32, device = self.device)
            r_q = self.model.bottlenecksm(self.model.bottleneckfc(samples_n))           
            samples_poisson = self.model.reparam_poisson(r_q, torch.Tensor(self.num_steps, batch_size, self.outputs))
            # print(samples_poisson, samples_poisson.shape)
            
            with torch.no_grad():  # Disable gradient calculation for testing
                spk, mem = self.model.decoder(samples_poisson) # spk.shape == [num_steps, batch, output neurons decoder]
                # print("spk: ", spk.sum(dim=(0, 1)).shape)
                # l = spk.shape[1]
                # print(spk[int(i/100), int(i/100), :])
                # prob = spk.sum(dim=(0, 1)) 
                # print(prob)
                # Append the prob for the current batch
                spikes = torch.cat((spk,spikes), 1)
                # print(spk, spk.shape
                # Free up memory
                del spk, mem, samples_n, samples_poisson, r_q
                # torch.cuda.empty_cache()
                
        # Calculate and return the average test loss over the dataset
        # avg_test_loss = sum(test_loss) / len(dataloader)
        # print(spikes.shape, spikes)
        tot_spikes_per_output = spikes
        
        # print("tot_spikes_per_output", tot_spikes_per_output)
        # tot_spikes = tot_spikes_per_output.sum(0)
        # spiking_options = self.num_steps * self.samples
        # print("tot_spikes", tot_spikes)
        # print("spiking_options", spiking_options)
        # probabilities = tot_spikes_per_output
        # # print(probabilities.shape)
        # probabilities = torch.div(tot_spikes_per_output, tot_spikes)
        # # print("probabilities", probabilities)
        return spikes
    
    def calculate_prob_from_spikes(self, spikes, batch_size):
        spikes = spikes.sum(0)
        # print(spikes.sum(0))
        P_vae = np.zeros([4] * self.n)
        for s in range(spikes.shape[0]):
            index = []
            for q in range(self.n):
                # print(self.measurements[s, (q * 4): q + (q * 4) + 4])
                index.append(np.argmax(spikes[s, (q * 4): (q + 1) * 4]))
            # print(index)
            # print(tuple(index))
            P_vae[tuple(index)] += 1
        P_vae = P_vae.reshape(-1)
        P_vae = P_vae/batch_size        
        return P_vae
    
    def tensor_product_povm_matrices(self, s):

        # Create the M^(alpha) matrices for a single qubit
        single_qubit_povm_matrices = [create_povm_matrix(s) for s in self.s_vectors]
        # Create all combinations of POVM outcomes for n qubits
        combinations = product(single_qubit_povm_matrices, repeat=self.n)

        # Calculate the tensor products for each combination
        povm_matrices_n_qubits = []
        for comb in combinations:
            povm_matrix = comb[0]
            for matrix in comb[1:]:
                  povm_matrix = np.kron(povm_matrix, matrix)  # Tensor product
            povm_matrices_n_qubits.append(povm_matrix)

        return povm_matrices_n_qubits
    
    def reconstruct_matrix_from_prob(self, prob_rec, prob_true):

        povm_matrices_n_qubits = self.dataset.n_qubit_povms

        # Initialize the density matrix for n qubits (size 2^n x 2^n)
        dim = 2**self.n
        rho_rec = np.zeros((dim, dim), dtype=np.complex128)
        rho_true = np.zeros((dim, dim), dtype=np.complex128)

        # Reconstruct the density matrix using the POVM matrices and probabilities
        for i in range(len(prob_rec)):
            rho_rec += float(prob_rec[i]) * povm_matrices_n_qubits[i].cpu().numpy()
            rho_true += float(prob_true[i]) * povm_matrices_n_qubits[i].cpu().numpy()

        # Normalize the density matrix to ensure the trace is 1
        rho_rec /= np.trace(rho_rec)
        rho_true /= np.trace(rho_true)
        
        # q_rec = qiskit.quantum_info.DensityMatrix(rho_rec, dims=None)
        # q_rec.draw(output="city", filename='q_rec.png') 

        return rho_rec, rho_true
    
def fid(P_true, P_rec):
    return torch.sqrt(torch.mul(P_true.to("cpu"), P_rec.to("cpu"))).sum(0)

def create_povm_matrix(s):
    return (1/4) * (I + s[0] * sigma_x + s[1] * sigma_y + s[2] * sigma_z)

def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False,
                     ylim_max1=1.25, ylim_max2=1.25, neuron_index=0):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,6), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input current
  ax[0].plot(cur, c="tab:orange")
  ax[0].set_ylim([0, ylim_max1])
  ax[0].set_xlim([0, 200])
  ax[0].set_ylabel("Input Current ($I_{in}$)")
  if title:
    ax[0].set_title(title)

  # Select data for the specified output neuron
  i = torch.remainder(neuron_index, outputs)
  j = neuron_index.div(outputs, rounding_mode="floor")
  # print(j)
  mem = mem[:, j, i].detach()  # Select neuron data from the first batch
  spk= spk[:, j, i].detach()  # Select neuron data from the first batch

  # Plot membrane potential
  ax[1].plot(mem)
  ax[1].set_ylim([0, ylim_max2])
  ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
  if thr_line:
    ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
  plt.xlabel("Time step")

  # Plot output spike using spikeplot
  splt.raster(spk, ax[2], s=400, c="black", marker="|")
  if vline:
    ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha = 0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
  plt.ylabel("Output spikes")
  plt.yticks([])

  plt.show()

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title):
  # Generate Plots
  fig, ax = plt.subplots(3, figsize=(8,7), sharex=True,
                        gridspec_kw = {'height_ratios': [1, 1, 0.4]})

  # Plot input spikes
  splt.raster(spk_in[:,0], ax[0], s=0.03, c="black")
  ax[0].set_ylabel("Input Spikes")
  ax[0].set_title(title)

  # Plot hidden layer spikes
  splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s = 0.05, c="black")
  ax[1].set_ylabel("Hidden Layer")

  # Plot output spikes
  splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
  ax[2].set_ylabel("Output Spikes")
  ax[2].set_ylim([0, 10])

  plt.show()
    
def plot_histogram(fidelities, parameters):
    # Create the x-tick labels as strings from the tuples
    x_labels = [f"N={tup[0]}, bs={tup[1]}, Ne={tup[2]}" for tup in parameters]

    # Generate x-axis positions
    x_positions = np.arange(len(fidelities))

    # Plot the histogram (bar chart)
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.bar(x_positions, fidelities, color='blue', alpha=0.7)

    # Add labels, title, and grid
    plt.xlabel('Parameter Settings (N, batch_size, Ne)', fontsize=12)
    plt.ylabel('Values from Function', fontsize=12)
    plt.title('Histogram of Function Values vs Parameter Settings', fontsize=14)
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')  # Rotate x-labels for better readability

    # Display the plot
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.show()

def fidelity(rho, sigma):
    # Calculate the square root of the first density matrix
    sqrt_rho = sqrtm(rho)

    # Calculate the intermediate matrix product sqrt(rho) * sigma * sqrt(rho)
    product_matrix = sqrt_rho @ sigma @ sqrt_rho

    # Calculate the square root of the product matrix
    sqrt_product_matrix = sqrtm(product_matrix)

    # Calculate the trace of the square root of the product matrix
    fidelity_value = np.trace(sqrt_product_matrix)

    # Square the trace to get the fidelity
    fidelity_value = np.real(fidelity_value) ** 2  

    return fidelity_value



