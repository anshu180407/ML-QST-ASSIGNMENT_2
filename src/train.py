import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from model import QSTModel
from scipy.linalg import sqrtm
import time
import pathlib

def calculate_fidelity_batch(rho_est, rho_gt):
    """Calculates fidelity between two density matrices."""
    # This is slow for large batches but okay for verification
    fids = []
    for i in range(rho_est.shape[0]):
        r1 = rho_est[i].detach().cpu().numpy()
        r2 = rho_gt[i].detach().cpu().numpy()
        # F = (tr(sqrt(sqrt(r1)r2sqrt(r1))))^2
        try:
            sqrt_r1 = sqrtm(r1)
            fid = np.trace(sqrtm(sqrt_r1 @ r2 @ sqrt_r1)).real**2
            fids.append(fid)
        except:
            fids.append(0.0)
    return np.mean(fids)

def trace_distance_batch(rho_est, rho_gt):
    """Calculates trace distance: 0.5 * Tr|rho_est - rho_gt|"""
    diff = rho_est - rho_gt
    # Trace distance is 0.5 * sum of singular values
    # Actually for Hermitian, singular values are absolute eigenvalues
    u, s, v = torch.linalg.svd(diff)
    td = 0.5 * s.sum(dim=-1).mean()
    return td.item()

def train_model():
    data_path = "Assignment_2/data/assignment_2/shadow_dataset_2q.pt"
    if not pathlib.Path(data_path).exists():
        print(f"Data not found at {data_path}. Run generate_data.py first.")
        return

    data = torch.load(data_path)
    gt_matrices = data["gt_matrices"]
    bases = data["bases"]
    outcomes = data["outcomes"]

    # Combine bases and outcomes into one input tensor
    # (batch, num_shadows, num_qubits, 2)
    inputs = torch.stack([bases.float(), outcomes], dim=-1)

    dataset = TensorDataset(inputs, gt_matrices)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = QSTModel(num_qubits=2, num_shadows=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Loss: MSE on the matrix elements
    def frobenius_loss(rho_est, rho_gt):
        diff = rho_est - rho_gt
        return torch.mean(torch.abs(diff)**2)

    num_epochs = 50
    print("Starting training (Transformer model)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = frobenius_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")

    # Evaluation
    model.eval()
    all_fids = []
    all_tds = []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in test_loader:
            y_pred = model(x)
            for i in range(y_pred.shape[0]):
                r1 = y_pred[i].cpu().numpy()
                r2 = y[i].cpu().numpy()
                try:
                    sqrt_r1 = sqrtm(r1)
                    fid = np.trace(sqrtm(sqrt_r1 @ r2 @ sqrt_r1)).real**2
                    all_fids.append(fid)
                except:
                    all_fids.append(0.0)
            
            # Trace distance
            diff = y_pred - y
            u, s, v = torch.linalg.svd(diff)
            tds = 0.5 * s.sum(dim=-1)
            all_tds.extend(tds.tolist())

    end_time = time.time()
    latency = (end_time - start_time) / test_size

    mean_fid = np.mean(all_fids)
    mean_td = np.mean(all_tds)

    print("\n--- Evaluation Metrics ---")
    print(f"Mean Fidelity: {mean_fid:.4f}")
    print(f"Mean Trace Distance: {mean_td:.4f}")
    print(f"Inference Latency: {latency*1000:.4f} ms per state")
    print(f"Test Set Size: {test_size} states")

    # Save model
    output_dir = pathlib.Path("Assignment_2/outputs")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "qst_model_2q.pt")
    print(f"Model saved to {output_dir}/qst_model_2q.pt")

if __name__ == "__main__":
    train_model()
