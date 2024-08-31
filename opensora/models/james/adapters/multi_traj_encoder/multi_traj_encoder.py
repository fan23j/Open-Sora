import torch
import torch.nn as nn

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)

class MultiTrajEncoder(nn.Module):
    def __init__(self, num_frames=300, num_instances=10, d_model=1152, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.fourier_embedder = FourierEmbedder(num_freqs=8)
        self.instance_embeddings = nn.Parameter(torch.zeros(num_instances, d_model))
        self.frame_embeddings = nn.Parameter(torch.zeros(num_frames, d_model))
        
        self.input_proj = nn.Linear(64, d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [batch_size, num_instances, num_frames, 4]
        batch_size, num_instances, num_frames, _ = x.shape

        # Apply Fourier embedding
        x = x.view(-1, 4)
        x = self.fourier_embedder(x)
        x = self.input_proj(x)
        x = x.view(batch_size, num_instances, num_frames, -1)
        
        # Add instance and frame embeddings
        x = x + self.instance_embeddings.unsqueeze(1).unsqueeze(0)
        x = x + self.frame_embeddings.unsqueeze(0).unsqueeze(0)
        
        # Reshape for transformer layers
        x = x.view(-1, num_instances * num_frames, self.d_model)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final projection
        x = self.output_proj(x)
        
        return x