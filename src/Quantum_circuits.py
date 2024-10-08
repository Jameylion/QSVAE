import os
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import BackendSampler
from qiskit.circuit import QuantumCircuit
import matplotlib.pyplot as plt
from itertools import product
from scipy.linalg import sqrtm
import cmath

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

class QuantumExperiment:
    """Handles the creation and execution of quantum experiments."""

    def __init__(self, backend, n, shots):
        self.backend = backend
        self.n = n
        self.shots = shots
        self.backend_name = backend.name if hasattr(backend, 'name') else backend.name
        self.circuits_compiled = None
        self.results = None

    def run_experiment(self):
        """Run the quantum experiment and return the result and compiled circuits."""
        state_circuit = self.get_qc_for_n_qubit_GHZ_state(self.n, self.backend_name)
        povm_circuit_list = self.povm_circuits(self.n, self.backend_name)

        # Combine state preparation and POVM measurements
        measurement_circuits = self.create_measurement_circuits(state_circuit, povm_circuit_list, self.backend_name)

        # Compile and run the circuits
        circuits_compiled = transpile(measurement_circuits, self.backend)
        job = self.backend.run(circuits_compiled, shots=self.shots, memory=True)
        result = job.result()

        self.circuits_compiled = circuits_compiled
        self.results = result

        return result, circuits_compiled

    def get_qc_for_n_qubit_GHZ_state(self, n, backend_name):
        """Creates a quantum circuit for an n-qubit GHZ state."""
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        return qc

    def povm_circuits(self, n, backend_name):
            # Create circuits that prepare the tetrahedral states
        circuits = []

        if backend_name == "Starmon-5":
          r = range(1,n+1)
          n = 5
        else:
          r = range(n)

        qc_0 = QuantumCircuit(n)
        qc_1 = QuantumCircuit(n)
        qc_2 = QuantumCircuit(n)
        qc_3 = QuantumCircuit(n)

        for i in r:
          # POVM state |psi_1> = 1/sqrt(3) (|0> + sqrt(2)|1>)
          qc_1.ry(2 * np.arccos(1 / np.sqrt(3)), i)  # Ry rotation to align state

          # POVM state |psi_2> = 1/sqrt(3) (|0> + exp(i2pi/3) sqrt(2)|1>)
          qc_2.ry(2 * np.arccos(1 / np.sqrt(3)), i)
          qc_2.rz(2 * np.pi / 3, i)  # Add phase rotation

          # POVM state |psi_3> = 1/sqrt(3) (|0> + exp(-i2pi/3) sqrt(2)|1>)
          qc_3.ry(2 * np.arccos(1 / np.sqrt(3)), i)
          qc_3.rz(-2 * np.pi / 3, i)  # Add phase rotation

        circuits.append(qc_0)
        circuits.append(qc_1)
        circuits.append(qc_2)
        circuits.append(qc_3)

        return circuits


    def create_measurement_circuits(self, state_circuit, povm_circuits, backend_name):
        """Combines state preparation and POVM measurement circuits."""
        measurement_circuits = []
        for povm_circ in povm_circuits:
            qc = state_circuit.compose(povm_circ)
            qc.measure_all()
            measurement_circuits.append(qc)
        return measurement_circuits

    def print_circuits_with_counts(self):
        """Prints the compiled circuits and measurement data."""
        circuits_compiled = self.circuits_compiled
        result = self.results
        # Print circuits and measurement data
        counts = result.get_counts()
        for i, circuit in enumerate(circuits_compiled):
          display(circuit.draw("mpl", style="iqp"))
          # latex_source = circuit.draw("latex_source", style="iqp")
          # print(latex_source)
          print(counts[i])

def select_backend(backend_type):
  if backend_type == "Starmon-5":
    import os
    from quantuminspire.credentials import get_authentication
    from quantuminspire.qiskit import QI
    from quantuminspire.credentials import get_token_authentication
    from coreapi.auth import TokenAuthentication
    token = TokenAuthentication('6af0e322481376e6785741e7af617420b671b522', scheme="token")
    QI.set_authentication(token)
    auth = get_token_authentication()

    print(QI.backends())
    backend = QI.get_backend('Starmon-5')

  elif backend_type == "AerSimulator":
    # Use AerSimulator
    from qiskit_aer import AerSimulator
    backend = AerSimulator()

    return backend

def create_povm_matrix(s):
    return (1/4) * (I + s[0] * sigma_x + s[1] * sigma_y + s[2] * sigma_z)

# Function to create tensor products of POVM matrices for n qubits
def tensor_product_povm_matrices(n, s):

    # Create the M^(alpha) matrices for a single qubit
    single_qubit_povm_matrices = [create_povm_matrix(s) for s in s_vectors]
    # Create all combinations of POVM outcomes for n qubits
    combinations = product(single_qubit_povm_matrices, repeat=n)

    # Calculate the tensor products for each combination
    povm_matrices_n_qubits = []
    for comb in combinations:
        povm_matrix = comb[0]
        for matrix in comb[1:]:
            povm_matrix = np.kron(povm_matrix, matrix)  # Tensor product
        povm_matrices_n_qubits.append(povm_matrix)

    return povm_matrices_n_qubits


# Define a function to calculate fidelity between two density matrices
def fidelity(rho, sigma):
    # Step 1: Calculate the square root of the first density matrix
    sqrt_rho = sqrtm(rho)

    # Step 2: Calculate the intermediate matrix product sqrt(rho) * sigma * sqrt(rho)
    product_matrix = sqrt_rho @ sigma @ sqrt_rho

    # Step 3: Calculate the square root of the product matrix
    sqrt_product_matrix = sqrtm(product_matrix)

    # Step 4: Calculate the trace of the square root of the product matrix
    fidelity_value = np.trace(sqrt_product_matrix)

    # Step 5: Square the trace to get the fidelity
    fidelity_value = np.real(fidelity_value) ** 2  # Take real part to avoid numerical issues

    return fidelity_value

def reconstruct_matrix_from_prob(n, s_vectors, prob_rec, prob_true):

    povm_matrices_n_qubits = tensor_product_povm_matrices(n, s_vectors)

    # Initialize the density matrix for n qubits (size 2^n x 2^n)
    dim = 2**n
    rho_rec = np.zeros((dim, dim), dtype=np.complex128)
    rho_true = np.zeros((dim, dim), dtype=np.complex128)


    # Reconstruct the density matrix using the POVM matrices and probabilities
    for i in range(len(prob_rec)):
        rho_rec += prob_rec[i].cpu().item()  * povm_matrices_n_qubits[i]
        rho_true += prob_true[i].cpu().item() * povm_matrices_n_qubits[i]

    # Normalize the density matrix to ensure the trace is 1
    rho_rec /= np.trace(rho_rec)
    rho_true /= np.trace(rho_true)

    return rho_rec, rho_true

def calculate_angles(state_vector):
    """
    Calculate the angles theta and phi that correspond to rotations
    from the |0> state to the given quantum state.

    Args:
        state_vector (array-like): The quantum state represented as a vector [a, b],
                                   where |psi> = a|0> + b|1>.

    Returns:
        tuple: (theta, phi) angles in radians.
    """
    # Extract the components of the state vector
    a, b = state_vector

    # Calculate theta and phi
    theta = 2 * np.arccos(np.abs(a))  # Angle for Ry rotation
    phi = np.angle(b) - np.angle(a)   # Angle for Rz rotation

    return theta, phi

# # Example state: Let's take a sample state |psi> = (1/√3)|0> + (√2/√3)|1>
# # Corresponds to tetrahedral POVM states, e.g., (1/√3, √2/√3)
# example_state = [1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)]

# # Calculate the angles
# theta, phi = calculate_angles(example_state)
# print(f"Calculated Angles: theta = {theta:.4f} rad, phi = {phi:.4f} rad")