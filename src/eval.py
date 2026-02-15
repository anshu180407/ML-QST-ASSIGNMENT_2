import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import sys
import pathlib
import time
from scipy.linalg import sqrtm

# Add src to path to import QSTModel
sys.path.append(str(pathlib.Path(__file__).parent))
from model import QSTModel

def evaluate():
    data_path = "Assignment_2/data/assignment_2/shadow_dataset_2q.pt"
    model_path = "Assignment_2/outputs/qst_model_2q.pt"
    
    if not pathlib.Path(data_path).exists():
        print(f"Data not found at {data_path}")
        return

    data = torch.load(data_path)
    gt_matrices = data["gt_matrices"]
    bases = data["bases"]
    outcomes = data["outcomes"]

    inputs = torch.stack([bases.float(), outcomes], dim=-1)

    dataset = TensorDataset(inputs, gt_matrices)
    # Use same seed as training for consistent split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    test_loader = DataLoader(test_dataset, batch_size=1) # batch_size=1 for latency measurement

    model = QSTModel(num_qubits=2, num_shadows=50)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_fids = []
    all_tds = []
    
    latencies = []
    
    with torch.no_grad():
        for x, y in test_loader:
            start_time = time.time()
            y_pred = model(x)
            end_time = time.time()
            latencies.append(end_time - start_time)
            
            r1 = y_pred[0].cpu().numpy()
            r2 = y[0].cpu().numpy()
            
            # Fidelity
            try:
                sqrt_r1 = sqrtm(r1)
                fid = np.trace(sqrtm(sqrt_r1 @ r2 @ sqrt_r1)).real**2
                all_fids.append(fid)
            except:
                all_fids.append(0.0)
            
            # Trace distance
            diff = y_pred - y
            u, s, v = torch.linalg.svd(diff)
            td = 0.5 * s.sum(dim=-1).item()
            all_tds.append(td)

    print("\n--- Final Results for Assignment 2 ---")
    print(f"Track: Track 1 (Classical Shadows)")
    print(f"Model: Transformer Encoder")
    print(f"Test Set Size: {test_size} states")
    print(f"Shadows per State (K): 50")
    print(f"Mean Fidelity: {np.mean(all_fids):.6f}")
    print(f"Mean Trace Distance: {np.mean(all_tds):.6f}")
    print(f"Inference Latency: {np.mean(latencies)*1000:.4f} ms per state")

if __name__ == "__main__":
    evaluate()
