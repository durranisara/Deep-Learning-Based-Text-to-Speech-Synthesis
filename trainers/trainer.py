import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

class TTSTrainer:
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            eps=config.training.eps,
            weight_decay=config.training.weight_decay
        )
        
        # Loss functions
        self.mel_loss = nn.MSELoss()
        self.gate_loss = nn.BCEWithLogitsLoss()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir='runs/tts')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def train_epoch(self, train_loader, val_loader=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mel_loss = 0
        total_gate_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move data to device
            texts = batch['texts']
            mels = batch['mels'].to(self.device)
            mel_lengths = batch['mel_lengths'].to(self.device)
            
            # Forward pass
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = self.model(
                texts, mel_lengths, mels
            )
            
            # Compute losses
            mel_loss = self.mel_loss(mel_outputs, mels)
            mel_loss_postnet = self.mel_loss(mel_outputs_postnet, mels)
            
            # Gate loss (simple stop token prediction)
            gate_target = torch.zeros_like(gate_outputs)
            for i, length in enumerate(mel_lengths):
                gate_target[i, length-1:, 0] = 1.0
            
            gate_loss = self.gate_loss(gate_outputs, gate_target)
            
            # Total loss
            loss = mel_loss + mel_loss_postnet + self.config.training.gate_loss_weight * gate_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip_thresh
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mel_loss += (mel_loss.item() + mel_loss_postnet.item()) / 2
            total_gate_loss += gate_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mel_loss': (mel_loss.item() + mel_loss_postnet.item()) / 2,
                'gate_loss': gate_loss.item()
            })
            
            # TensorBoard logging
            if self.global_step % self.config.training.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/mel_loss', 
                                      (mel_loss.item() + mel_loss_postnet.item()) / 2,
                                      self.global_step)
                self.writer.add_scalar('train/gate_loss', gate_loss.item(), self.global_step)
            
            self.global_step += 1
        
        # Validation
        if val_loader is not None:
            val_loss, val_mel_loss, val_gate_loss = self.validate(val_loader)
            self.writer.add_scalar('val/loss', val_loss, self.current_epoch)
            self.writer.add_scalar('val/mel_loss', val_mel_loss, self.current_epoch)
            self.writer.add_scalar('val/gate_loss', val_gate_loss, self.current_epoch)
        
        self.current_epoch += 1
        self.scheduler.step()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mel_loss = 0
        total_gate_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                texts = batch['texts']
                mels = batch['mels'].to(self.device)
                mel_lengths = batch['mel_lengths'].to(self.device)
                
                # Forward pass
                mel_outputs, mel_outputs_postnet, gate_outputs, _ = self.model(
                    texts, mel_lengths, mels
                )
                
                # Compute losses
                mel_loss = self.mel_loss(mel_outputs, mels)
                mel_loss_postnet = self.mel_loss(mel_outputs_postnet, mels)
                gate_loss = self.gate_loss(gate_outputs, torch.zeros_like(gate_outputs))
                
                loss = mel_loss + mel_loss_postnet + self.config.training.gate_loss_weight * gate_loss
                
                total_loss += loss.item()
                total_mel_loss += (mel_loss.item() + mel_loss_postnet.item()) / 2
                total_gate_loss += gate_loss.item()
        
        self.model.train()
        
        avg_loss = total_loss / len(val_loader)
        avg_mel_loss = total_mel_loss / len(val_loader)
        avg_gate_loss = total_gate_loss / len(val_loader)
        
        print(f"Validation - Loss: {avg_loss:.4f}, Mel Loss: {avg_mel_loss:.4f}, "
              f"Gate Loss: {avg_gate_loss:.4f}")
        
        return avg_loss, avg_mel_loss, avg_gate_loss
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
