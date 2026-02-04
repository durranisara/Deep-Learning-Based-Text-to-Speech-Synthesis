import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveGlow(nn.Module):
    """
    WaveGlow: A Flow-based Generative Network for Speech Synthesis
    Simplified version for TTS synthesis
    """
    def __init__(self, config):
        super().__init__()
        
        self.n_mels = config.audio.n_mels
        self.n_flows = 12
        self.n_group = 8
        self.early_every = 4
        self.early_size = 2
        
        # Affine coupling layers
        self.flows = nn.ModuleList()
        for k in range(self.n_flows):
            self.flows.append(AffineCouplingLayer(self.n_group, self.n_mels))
        
        # 1x1 invertible convolutions
        self.convs = nn.ModuleList()
        for k in range(self.n_flows):
            self.convs.append(Invertible1x1Conv(self.n_group))
        
    def forward(self, spect, audio):
        # audio: [B, T]
        # spect: [B, n_mels, T']
        
        # Pad or trim audio to match spectrogram length
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        
        log_det_tot = 0
        for i in range(self.n_flows):
            audio = self.convs[i](audio)
            audio, log_det = self.flows[i](audio, spect)
            log_det_tot = log_det_tot + log_det
        
        return audio, log_det_tot
    
    def inference(self, spect):
        """Generate audio from mel-spectrogram"""
        with torch.no_grad():
            # Start with random noise
            audio = torch.randn(spect.size(0), self.n_group, spect.size(2)).to(spect.device)
            
            for i in reversed(range(self.n_flows)):
                audio = self.flows[i].inverse(audio, spect)
                audio = self.convs[i].inverse(audio)
            
            # Reshape to waveform
            audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        
        return audio

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for WaveGlow"""
    def __init__(self, in_channels, cond_channels, filter_size=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels // 2 + cond_channels, filter_size, 1),
            nn.ReLU(),
            nn.Conv1d(filter_size, filter_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(filter_size, in_channels, 1)
        )
        
        self.scale = nn.Parameter(torch.zeros(in_channels // 2))
        
    def forward(self, x, spect):
        # Split input
        x_a, x_b = x.chunk(2, 1)
        
        # Process
        log_s, t = self.net(torch.cat([x_a, spect], 1)).chunk(2, 1)
        log_s = self.scale.unsqueeze(0).unsqueeze(-1) * torch.tanh(log_s)
        
        # Affine transform
        x_b = torch.exp(log_s) * x_b + t
        
        # Combine
        x = torch.cat([x_a, x_b], 1)
        
        # Compute log determinant
        log_det = log_s.sum(dim=[1, 2])
        
        return x, log_det
    
    def inverse(self, z, spect):
        # Split
        z_a, z_b = z.chunk(2, 1)
        
        # Process
        log_s, t = self.net(torch.cat([z_a, spect], 1)).chunk(2, 1)
        log_s = self.scale.unsqueeze(0).unsqueeze(-1) * torch.tanh(log_s)
        
        # Inverse affine transform
        z_b = (z_b - t) * torch.exp(-log_s)
        
        # Combine
        z = torch.cat([z_a, z_b], 1)
        
        return z

class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution layer"""
    def __init__(self, channels):
        super().__init__()
        
        # Initialize with random orthogonal matrix
        w = torch.randn(channels, channels)
        q, _ = torch.qr(w)
        
        # Ensure determinant is positive
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        
        self.w = nn.Parameter(q)
        
    def forward(self, x):
        # Compute log determinant
        log_det = torch.slogdet(self.w)[1] * x.size(2)
        
        # Apply transformation
        x = F.conv1d(x, self.w.unsqueeze(2))
        
        return x, log_det
    
    def inverse(self, z):
        # Compute inverse
        w_inv = self.w.inverse()
        
        # Apply inverse transformation
        z = F.conv1d(z, w_inv.unsqueeze(2))
        
        return z
