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

import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Define the Pauli matrices
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


class Model(torch.nn.Module):
    """ Complete model with encoder (SNN on CPU) and decoder(SNN on Brainscales) """

    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            readout_scale: float = 1.,
            device: str = "cpu"):
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
        self.device = device

        self.readout_scale = readout_scale

    def reparameterization(self, mean, var):
      std = torch.sqrt(var).to(self.device)
      eps = torch.randn(std.shape).to(self.device)

      return mean + eps * std

    def loss_function(self, x_hat, x, mean, log_var):
     # Calculate reconstruction loss (example: MSE)
      reconstruction_loss = nn.MSELoss()(x_hat.reshape(-1), x.reshape(-1))

      # Calculate KL divergence
      kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

      # Combine the losses
      total_loss = reconstruction_loss + kl_divergence
      return total_loss

    def calc_mean_var(self, spk):

      # Calculate mean and variance across time steps
      mean = spk.mean(dim=0)  # Shape: [batch_size, output_size]

      variance = spk.var(dim=0, unbiased=False)  # Shape: [batch_size, output_size]
      # print("spike mean shape per outpu neuron\n", mean.shape)
      del spk

      return mean, variance

    def plot_spikes(self, spk, mem, cur, ende):
      max_spk_index = torch.argmax(spk.sum(0)) % ende
      print(torch.argmax(spk.sum(0)), torch.max(spk.sum(0)), spk.shape)
      print("max spike index\n", torch.remainder(max_spk_index,  ende))
      print("spike sum shape per output over all time steps\n",spk.sum(0).shape)
      print("spike sum per output over all time steps\n",spk.sum(0))


      plot_cur_mem_spk(cur.detach().cpu().numpy(), mem.detach().cpu(),
                         spk.detach().cpu(), thr_line=1,ylim_max1=0.5,
        title="LIF Neuron that spikes the most from the output of the decoder",
                     neuron_index = max_spk_index)


    def forward(self, x):
        spk, mem = self.encoder(x)
        mean, log_var = self.calc_mean_var(spk)
        # self.plot_spikes(spk, mem, cur)

        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
#         self.latent_z = z
        spk, mem = self.decoder(z)

        del z
        torch.cuda.empty_cache()
        # print(type(x_hat))

        return spk, mem, mean, log_var

# @title Class Encoder
class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, beta, num_steps):
    super().__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.lif1 = snn.Leaky(beta=beta)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.lif2 = snn.Leaky(beta=beta)
    self.num_steps = num_steps

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

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, beta, num_steps):
    super().__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.lif1 = snn.Leaky(beta=beta)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.lif2 = snn.Leaky(beta=beta)
    self.num_steps = num_steps

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


class SQVAE(Model):
    def __init__(self, n, batch_size, beta, num_steps, learning_rate, device, shots,
                  first_run=True, dataset=None,s_vectors = None):
        super(SQVAE, self).__init__(encoder=None, decoder=None)
        # Model parameters
        self.n = n
        self.batch_size = batch_size
        self.beta = beta
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.device = device
        self.first_run = first_run
        self.dataset = dataset
        self.s_vectors = s_vectors

        # Define the input, hidden, and output sizes
        self.inputs = 4 * n
        self.hidden = 32 * n
        self.outputs = 2 * 2**n
        self.shots = shots
        
        self.MOCK          = False
        self.DT            = 2.0e-06  # s

        # Initialize encoder and decoder as part of the model
        self.encoder = Encoder(self.inputs, self.hidden, self.outputs, self.beta, self.num_steps).to(self.device)
        self.decoder = snn = SNN(
                                n_in=2 * 2**n,
                                n_hidden= 32 *n,
                                n_out=4 * n,
                                mock=self.MOCK,
                                dt=self.DT,
                                tau_mem=6.0e-06,
                                tau_syn=6.0e-06,
                                alpha=50.,
                                trace_shift_hidden=int(.0e-06/self.DT),
                                trace_shift_out=int(.0e-06/self.DT),
                                weight_init_hidden=(0.001, 0.25),
                                weight_init_output=(0.0, 0.1),
                                weight_scale=66.39,
                                trace_scale=0.0147,
                                input_repetitions=1 if self.MOCK else 5,
                                device=device).to(self.device)

        # Combine encoder and decoder into the model
        self.model = Model(self.encoder, self.decoder, device=device).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # If this is the first run, save the initial model state
        if first_run:
            torch.save(self.model.state_dict(), f"data/models/model_{n}qubit_{int(shots)}shots.pt")
        else:
            self.model.load_state_dict(torch.load( f"data/models/model_{n}qubit_{int(shots)}shots.pt"))

        # # Check for multiple GPUs and use DataParallel
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs for training.")
        #     self.model = DataParallel(self.model)

    def sample_latent_space(self, num_samples):
        """
        Samples from the latent space and reconstructs using the decoder.

        Args:
            num_samples (int): Number of latent samples to generate.

        Returns:
            reconstructed_samples: Reconstructed samples from the latent space.
        """
        # Sample from a standard normal distribution (latent space)
        latent_dim = self.outputs  # The size of the latent space

        z = torch.randn(num_samples, latent_dim).to(self.device)  # Random samples from N(0, I)

        # Pass the sampled latent vectors through the decoder to reconstruct
        with torch.no_grad():
            reconstructed_samples = self.decoder(z)

        return reconstructed_samples

    def run(self, train=True, test=True, val=True, train_loader=None, test_loader=None, val_loader=None, num_epochs=11):
        """Run the model with options for training, testing, and validation."""
        fidelity_score = 0
        if train:
            # Initialize TensorBoard writer for training
            writer = SummaryWriter(log_dir='runs/Paperreproduction')
            self.train_model(train_loader,self.optimizer, self.device, num_epochs, writer)
            torch.save(self.model.state_dict(), f"data/models/model_{self.n}qubit_{int(self.shots)}shots.pt")

        if test:
            # Initialize TensorBoard writer for testing
            writer = SummaryWriter(log_dir='runs/test_experiment')
            test_loss = self.test_model(test_loader, self.device, writer)
            print("Average test loss: ", test_loss)

        if val:
            prob_rec = self.val_model(val_loader, self.device)
            prob_true = self.dataset.probability_true
            rho_rec, rho_true = self.reconstruct_matrix_from_prob(prob_rec.cpu(), prob_true)
            fidelity_score = fidelity(rho_rec, rho_true)
            print(f"The fidelity for {self.n} qubits is {fidelity_score} with {self.batch_size[2]} samples.")

        return fidelity_score

    def train_model(self, dataloader, optimizer, device, num_epochs, writer):
        self.model.train()

        # Early stopping parameters
        patience = 3  # Number of epochs to wait before stopping if no improvement
        best_loss = float('inf')  # Initialize the best loss to a large value
        epochs_no_improve = 0  # Counter for epochs without improvement

        # Initialize list for storing loss values
        train_loss = []

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = []
            data = iter(dataloader)

            for batch_idx, sample_batched in enumerate(tqdm(data)):
                sample_batched = sample_batched['POVM'].to(device)
                optimizer.zero_grad()

                # Forward pass
                spk, mem, mean, log_var = self.model(sample_batched)

                # Calculate loss
                loss = self.model.loss_function(spk.sum(0), sample_batched, mean, log_var)
                loss.backward()
                optimizer.step()

                # Append current loss
                epoch_loss.append(loss.item())

                # Log loss to TensorBoard after each batch
                writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)

                # Free up memory
                del spk, mem, mean, log_var, loss
                torch.cuda.empty_cache()

                # Log memory usage if on GPU
                if torch.cuda.is_available():
                  memory_allocated = torch.cuda.memory_allocated(device)
                  memory_cached = torch.cuda.memory_reserved(device)

                  # Log memory usage to TensorBoard
                  writer.add_scalar('Memory/Allocated_MB', memory_allocated / (1024 ** 2), epoch * len(dataloader) + batch_idx)
                  writer.add_scalar('Memory/Cached_MB', memory_cached / (1024 ** 2), epoch * len(dataloader) + batch_idx)

            # Calculate and store the average loss for the epoch
            avg_epoch_loss = sum(epoch_loss) / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            train_loss.append(avg_epoch_loss)

            # Log average epoch loss to TensorBoard
            writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

            # Check for early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), "data/best_model.pt")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                # Load the best model state before stopping
                self.model.load_state_dict(torch.load("data/best_model.pt"))
                break

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
    def val_model(self, dataloader, device):
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
        probabilities = []

        data = iter(dataloader)
        for batch_idx, sample_batched in enumerate(tqdm(data)):
            sample_batched = sample_batched['POVM'].to(device)

            samples_z = self.sample_latent_space(sample_batched.shape[0])

            with torch.no_grad():  # Disable gradient calculation for testing
                if isinstance(samples_z, tuple):
                    samples_z = samples_z[0]  # Ensure it is a Tensor

                spk, mem, mean, log_var = self.model(samples_z)
                l = spk.shape[1]
                prob = spk.sum(dim=(0, 1)) / l

                # Append the prob for the current batch
                probabilities.append(prob)

                # Free up memory
                del spk, mem, mean, log_var, prob
                torch.cuda.empty_cache()

        # Calculate and return the average test loss over the dataset
        # avg_test_loss = sum(test_loss) / len(dataloader)
        lt = sample_batched.shape[0]
        # print(probabilities[0].shape)
        return probabilities[0].sum(0)/lt
    
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

      povm_matrices_n_qubits = self.tensor_product_povm_matrices(4)

      # Initialize the density matrix for n qubits (size 2^n x 2^n)
      dim = 2**self.n
      rho_rec = np.zeros((dim, dim), dtype=np.complex128)
      rho_true = np.zeros((dim, dim), dtype=np.complex128)


      # Reconstruct the density matrix using the POVM matrices and probabilities
      for i in range(len(prob_rec)):
          rho_rec += prob_rec[i].item() * povm_matrices_n_qubits[i]
          rho_true += prob_true[i] * povm_matrices_n_qubits[i] #.cpu().item()

      # Normalize the density matrix to ensure the trace is 1
      rho_rec /= np.trace(rho_rec)
      rho_true /= np.trace(rho_true)

      return rho_rec, rho_true

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


# Function to create tensor products of POVM matrices for n qubits
