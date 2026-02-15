import pennylane as qml
from pennylane import numpy as np
import pathlib
import torch
import json
from tqdm import tqdm

def generate_random_pure_state(num_qubits):
    """Generates a Haar random pure state vector."""
    dim = 2**num_qubits
    real_parts = np.random.randn(dim)
    imag_parts = np.random.randn(dim)
    psi = real_parts + 1j * imag_parts
    return psi / np.linalg.norm(psi)

def measure_shadow(psi, num_qubits):
    """
    Performs randomized Pauli measurements on the state psi.
    Returns: basis (0=X, 1=Y, 2=Z) and outcome (-1 or 1).
    """
    dev = qml.device("default.qubit", wires=num_qubits, shots=1)
    
    # 0=X, 1=Y, 2=Z
    basis = np.random.randint(0, 3, size=num_qubits)
    
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(psi, wires=range(num_qubits))
        for i, b in enumerate(basis):
            if b == 0: qml.Hadamard(wires=i)
            elif b == 1: qml.adjoint(qml.S)(wires=i); qml.Hadamard(wires=i)
        return [qml.sample(qml.PauliZ(i)) for i in range(num_qubits)]

    outcome = circuit()
    # For shots=1, outcome is a list of arrays. Flatten it to a simple array.
    outcome = np.array(outcome).flatten()
    return basis, outcome

def generate_dataset(num_samples=1000, shadows_per_state=50, num_qubits=2, data_dir="data/assignment_2"):
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = []
    print(f"Generating {num_samples} states with {shadows_per_state} shadows each...")
    
    for _ in tqdm(range(num_samples)):
        psi = generate_random_pure_state(num_qubits)
        rho_gt = np.outer(psi, psi.conj())
        
        shadows = []
        for _ in range(shadows_per_state):
            basis, outcome = measure_shadow(psi, num_qubits)
            shadows.append({
                "basis": basis.tolist(),
                "outcome": outcome.tolist()
            })
            
        dataset.append({
            "rho_gt": rho_gt.tolist(), # Real/imag parts will be handled by json
            "shadows": shadows
        })

    # Save as torch tensors for easier training later
    gt_matrices = []
    all_bases = []
    all_outcomes = []
    
    for item in dataset:
        gt_matrices.append(item["rho_gt"])
        state_bases = [s["basis"] for s in item["shadows"]]
        state_outcomes = [s["outcome"] for s in item["shadows"]]
        all_bases.append(state_bases)
        all_outcomes.append(state_outcomes)
        
    torch.save({
        "gt_matrices": torch.tensor(np.array(gt_matrices), dtype=torch.complex64),
        "bases": torch.tensor(np.array(all_bases), dtype=torch.long),
        "outcomes": torch.tensor(np.array(all_outcomes), dtype=torch.float32)
    }, pathlib.Path(data_dir) / "shadow_dataset_2q.pt")
    
    print(f"Dataset saved to {data_dir}/shadow_dataset_2q.pt")

if __name__ == "__main__":
    # For a quick implementation, let's use 200 states with 50 shadows each
    generate_dataset(num_samples=200, shadows_per_state=50, num_qubits=2)
