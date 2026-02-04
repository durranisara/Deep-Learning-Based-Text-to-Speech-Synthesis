#!/usr/bin/env python3
"""
Inference script for TTS synthesis
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config.config import TTSConfig
from models.tacotron2 import Tacotron2
from models.waveglow import WaveGlow
from inference.synthesizer import Synthesizer

def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize speech from text")
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--vocoder_checkpoint", type=str, default=None,
                       help="Path to vocoder checkpoint (optional)")
    parser.add_argument("--output", type=str, default="output.wav",
                       help="Output audio file path")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = TTSConfig()
    if args.config:
        config.load(args.config)
    else:
        # Try to load config from checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            config.load(config_path)
    
    # Load model
    print("Loading model...")
    model = Tacotron2(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load vocoder if specified
    vocoder = None
    if args.vocoder_checkpoint:
        print("Loading vocoder...")
        vocoder = WaveGlow(config)
        vocoder_checkpoint = torch.load(args.vocoder_checkpoint, map_location='cpu')
        vocoder.load_state_dict(vocoder_checkpoint['model_state_dict'])
    
    # Create synthesizer
    synthesizer = Synthesizer(model, vocoder, config)
    
    # Synthesize
    print(f"Synthesizing: {args.text}")
    audio, mel, alignment = synthesizer.synthesize(
        args.text,
        output_path=args.output,
        plot_alignment=True
    )
    
    print(f"Audio saved to {args.output}")

if __name__ == "__main__":
    main()
