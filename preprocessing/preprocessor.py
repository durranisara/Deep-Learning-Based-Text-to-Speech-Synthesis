import os
import re
import json
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from unidecode import unidecode

class TextProcessor:
    """Text preprocessing for TTS"""
    
    # Character set for English
    _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
    
    # Mappings
    _symbol_to_id = {s: i for i, s in enumerate(_characters)}
    _id_to_symbol = {i: s for i, s in enumerate(_characters)}
    
    def __init__(self, config):
        self.config = config
        self.symbols = self._characters
        
    def text_to_sequence(self, text):
        """Convert text to sequence of ids"""
        # Clean text
        text = self.clean_text(text)
        
        # Convert to sequence
        sequence = []
        for symbol in text:
            if symbol in self._symbol_to_id:
                sequence.append(self._symbol_to_id[symbol])
            else:
                # Replace unknown symbols with space
                sequence.append(self._symbol_to_id[' '])
        
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of ids to text"""
        return ''.join([self._id_to_symbol[id] for id in sequence])
    
    def clean_text(self, text):
        """Clean text for TTS"""
        # Convert to lowercase if configured
        if self.config.preprocessing.text_lowercase:
            text = text.lower()
        
        # Remove unwanted characters
        text = unidecode(text)
        
        # Apply cleaners
        for cleaner in self.config.preprocessing.text_cleaners:
            if cleaner == 'english_cleaners':
                text = self.english_cleaners(text)
        
        return text
    
    def english_cleaners(self, text):
        """Basic English text cleaner"""
        # Expand abbreviations
        abbreviations = {
            'mr.': 'mister',
            'mrs.': 'misses',
            'dr.': 'doctor',
            'st.': 'saint',
            'co.': 'company',
            'jr.': 'junior',
            'sr.': 'senior',
            'vs.': 'versus',
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters not in symbol set
        text = re.sub(r'[^\w\s\'-.,!?]', '', text)
        
        return text.strip()

class AudioProcessor:
    """Audio preprocessing for TTS"""
    
    def __init__(self, config):
        self.config = config
    
    def load_audio(self, filepath):
        """Load audio file"""
        audio, sr = librosa.load(filepath, sr=self.config.audio.sample_rate)
        
        # Trim silence
        if self.config.preprocessing.trim_silence:
            audio, _ = librosa.effects.trim(
                audio, top_db=self.config.preprocessing.trim_top_db
            )
        
        return audio
    
    def save_audio(self, audio, filepath):
        """Save audio file"""
        sf.write(filepath, audio, self.config.audio.sample_rate)
    
    def audio_to_mel(self, audio):
        """Convert audio to mel-spectrogram"""
        return librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            n_mels=self.config.audio.n_mels,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax
        )
    
    def mel_to_audio(self, mel_spec):
        """Convert mel-spectrogram to audio (Griffin-Lim)"""
        return librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax
        )
