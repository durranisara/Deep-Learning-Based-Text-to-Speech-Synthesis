#!/usr/bin/env python3
"""
Training script for TTS model
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TTSConfig
from data.dataset import TTSDataset, collate_fn
from models.tacotron2 import Tacotron2
from trainers.trainer import TTSTrainer
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train TTS model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load or create config
    config = TTSConfig()
    if args.config:
        config.load(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(output_dir / "config.json")
    
    # Create dataset
    print("Loading dataset...")
    dataset = TTSDataset(args.data_dir, config)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = Tacotron2(config)
    
    # Create trainer
    trainer = TTSTrainer(model, config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    print("Starting training...")
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        
        # Train one epoch
        avg_loss = trainer.train_epoch(train_loader, val_loader)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.training.checkpoint_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(checkpoint_path)
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

if __name__ == "__main__":
    main()
