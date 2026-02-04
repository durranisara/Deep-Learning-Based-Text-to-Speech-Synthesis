# Deep Learning-Based Text-to-Speech Synthesis

A complete implementation of Tacotron 2 with WaveGlow vocoder for high-quality text-to-speech synthesis.

## Features

- **Tacotron 2**: Sequence-to-sequence model with attention for mel-spectrogram generation
- **WaveGlow**: Flow-based neural vocoder for audio synthesis
- **Modular Architecture**: Easy to modify and extend
- **TensorBoard Integration**: Training visualization
- **Checkpointing**: Model saving and loading
- **Batch Processing**: Efficient training and inference

## Clone the repository:
```bash
git clone https://github.com/durranisara/Deep-Learning-Based-Text-to-Speech-Synthesis.git
cd Deep-Learning-Based-Text-to-Speech-Synthesis
```
## Install dependencies:
```bash
pip install -r requirements.txt
```
## Training
To train the model:
```bash
python train.py --data_dir data/ --output_dir checkpoints/
```
## Synthesis
To synthesize speech from text:
```bash
python synthesize.py --text "Hello, world!" --checkpoint checkpoints/final_model.pt --output hello.wav
```

Key Features of this Implementation:
Complete Tacotron 2 Architecture: Includes encoder, decoder with attention, and postnet

WaveGlow Vocoder: High-quality neural vocoder for audio synthesis

Modular Design: Easy to modify and extend

Training Pipeline: Complete with data loading, training loops, and validation

Inference System: Ready-to-use synthesis pipeline

Configuration Management: Easy hyperparameter tuning

Visualization Tools: For monitoring training and results
