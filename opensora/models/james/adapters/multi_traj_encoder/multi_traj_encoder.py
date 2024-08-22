import torch
import torch.nn as nn

class FourierFeatures(nn.Module):
    def __init__(self, num_frames, d_model, dtype):
        super().__init__()
        self.num_frames = num_frames
        self.d_model = d_model
        self.freq_bands = nn.Parameter(torch.linspace(1, num_frames / 2, d_model // 4).view(1, -1), requires_grad=False)
        self.freq_bands.data = self.freq_bands.data.to(device="cuda", dtype=dtype)
        
    def forward(self, t):
        # t: [num_frames]
        t = t.unsqueeze(1)  # [num_frames, 1]
        freq_sin = torch.sin(2 * torch.pi * self.freq_bands * t / self.num_frames)
        freq_cos = torch.cos(2 * torch.pi * self.freq_bands * t / self.num_frames)
        return torch.cat([freq_sin, freq_cos, freq_sin, freq_cos], dim=-1)

#TODO: d_model should come from hidden_dim in main diffuser
#TODO: ablate hyperparameters
class MultiTrajEncoder(nn.Module):
    def __init__(self, num_frames=64, num_instances=10, d_model=1152, nhead=4, num_layers=2, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        
        # Embedding layer to project 4D bounding box to d_model dimensions
        self.embed = nn.Linear(4, d_model, dtype=dtype)
        
        # Fourier feature positional encoding
        self.pos_encoding = FourierFeatures(num_frames, d_model, dtype)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dtype=dtype)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # Ensure input is on the correct device and has the correct dtype
        x = x.to(device=self.device, dtype=self.dtype)
        
        # x shape: [batch_size, num_instances, num_frames, 4]
        batch_size, num_instances, num_frames, _ = x.shape

        # Embed the bounding boxes
        x = self.embed(x)  # Shape: [batch_size, num_instances, num_frames, d_model]
        
        # Generate and add positional encoding
        t = torch.arange(num_frames, device=self.device, dtype=self.dtype)
        pos_enc = self.pos_encoding(t)
        x = x + pos_enc.unsqueeze(0).unsqueeze(0)
        
        # Reshape for transformer input
        x = x.view(batch_size * num_instances, num_frames, self.d_model)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Reshape back
        x = x.view(batch_size, num_instances, num_frames, self.d_model)
        
        return x