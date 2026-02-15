import torch
import torch.nn as nn
import torch.nn.functional as F

class QSTModel(nn.Module):
    def __init__(self, num_qubits=2, num_shadows=50, d_model=64, nhead=4, num_layers=2):
        super(QSTModel, self).__init__()
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits
        self.num_shadows = num_shadows
        
        # Input projection: (num_qubits * 2) -> d_model
        self.input_proj = nn.Linear(num_qubits * 2, d_model)
        
        # Transformer Encoder (Track 1 requirement)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final layers to Cholesky params
        self.fc_common = nn.Linear(d_model, d_model // 2)
        self.fc_real = nn.Linear(d_model // 2, self.dim * (self.dim + 1) // 2)
        self.fc_imag = nn.Linear(d_model // 2, self.dim * (self.dim - 1) // 2)

    def forward(self, x):
        # x: (batch, num_shadows, num_qubits, 2)
        batch_size = x.shape[0]
        
        # Flatten qubits and features: (batch, num_shadows, num_qubits*2)
        x_flat = x.view(batch_size, self.num_shadows, -1)
        
        # Project to d_model: (batch, num_shadows, d_model)
        h = self.input_proj(x_flat)
        
        # Transformer processing: (batch, num_shadows, d_model)
        h = self.transformer(h)
        
        # Global Average Pooling over shadow sequence: (batch, d_model)
        h_pool = torch.mean(h, dim=1)
        
        feat = F.relu(self.fc_common(h_pool))
        real_params = self.fc_real(feat)
        imag_params = self.fc_imag(feat)
        
        # Construct L
        L_real = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        L_imag = torch.zeros(batch_size, self.dim, self.dim, device=x.device)
        
        # Fill L_real (lower triangular including diagonal)
        idx_real = 0
        for i in range(self.dim):
            for j in range(i + 1):
                L_real[:, i, j] = real_params[:, idx_real]
                idx_real += 1
                
        # Fill L_imag (lower triangular excluding diagonal)
        idx_imag = 0
        for i in range(1, self.dim):
            for j in range(i):
                L_imag[:, i, j] = imag_params[:, idx_imag]
                idx_imag += 1
                
        L = torch.complex(L_real, L_imag)
        
        # rho = LL^H / Tr(LL^H)
        rho = torch.matmul(L, L.adjoint())
        
        # Normalize trace
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).real
        rho = rho / trace.unsqueeze(-1).to(torch.complex64)
        
        return rho

if __name__ == "__main__":
    model = QSTModel(num_qubits=2, num_shadows=50)
    test_input = torch.randn(8, 50, 2, 2) # batch=8, shadows=50, qubits=2, features=2
    rho = model(test_input)
    print(f"Output shape: {rho.shape}")
    print(f"Trace (should be 1): {torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)}")
