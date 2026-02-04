import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from .text_processing import TextProcessor

class Synthesizer:
    def __init__(self, model, vocoder, config, device=None):
        self.model = model
        self.vocoder = vocoder
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        if self.vocoder is not None:
            self.vocoder.to(self.device)
            self.vocoder.eval()
        
        self.text_processor = TextProcessor(config)
    
    def synthesize(self, text, output_path=None, plot_alignment=False):
        """Synthesize speech from text"""
        # Process text
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_length = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel_output, _, alignment = self.model.inference(text_tensor, text_length)
        
        # Convert mel to audio
        if self.vocoder is not None:
            audio = self.vocoder.inference(mel_output)
            audio = audio.squeeze().cpu().numpy()
        else:
            # Griffin-Lim as fallback
            audio = self.griffin_lim(mel_output.squeeze().cpu().numpy())
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save audio
        if output_path:
            sf.write(output_path, audio, self.config.audio.sample_rate)
            print(f"Audio saved to {output_path}")
        
        # Plot alignment
        if plot_alignment:
            self.plot_alignment(alignment.squeeze().cpu().numpy())
        
        return audio, mel_output.squeeze().cpu().numpy(), alignment.squeeze().cpu().numpy()
    
    def griffin_lim(self, mel_spec, n_iter=50):
        """Griffin-Lim algorithm for mel-to-audio conversion"""
        # Inverse mel basis
        mel_basis = librosa.filters.mel(
            sr=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            n_mels=self.config.audio.n_mels,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax
        )
        
        # Inverse mel scaling (approximate)
        mel_basis_inv = np.linalg.pinv(mel_basis)
        
        # Reconstruct magnitude spectrogram
        mag_spec = np.dot(mel_basis_inv, np.exp(mel_spec))
        
        # Griffin-Lim phase reconstruction
        angles = np.exp(2j * np.pi * np.random.rand(*mag_spec.shape))
        
        for i in range(n_iter):
            # Inverse STFT
            audio = librosa.istft(
                mag_spec * angles,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length
            )
            
            # Forward STFT
            stft = librosa.stft(
                audio,
                n_fft=self.config.audio.n_fft,
                hop_length=self.config.audio.hop_length,
                win_length=self.config.audio.win_length
            )
            
            angles = np.exp(1j * np.angle(stft))
        
        return audio
    
    def plot_alignment(self, alignment, text=None):
        """Plot attention alignment"""
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(alignment.T, aspect='auto', origin='lower', interpolation='none')
        ax.set_xlabel('Decoder Steps')
        ax.set_ylabel('Encoder Steps')
        
        if text:
            # Add text labels if provided
            pass
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
    
    def batch_synthesize(self, texts, output_dir):
        """Synthesize multiple texts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"output_{i:03d}.wav"
            audio, mel, alignment = self.synthesize(text, output_path)
            results.append({
                'text': text,
                'audio': audio,
                'mel': mel,
                'alignment': alignment,
                'output_path': output_path
            })
        
        return results
