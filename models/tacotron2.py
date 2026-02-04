import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import LocationSensitiveAttention

class Encoder(nn.Module):
    """Text Encoder for Tacotron 2"""
    def __init__(self, config):
        super().__init__()
        
        self.embedding = nn.Embedding(
            config.vocab_size, config.model.encoder_embedding_dim
        )
        
        convolutions = []
        for _ in range(config.model.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    config.model.encoder_embedding_dim,
                    config.model.encoder_embedding_dim,
                    kernel_size=config.model.encoder_kernel_size,
                    stride=1,
                    padding=(config.model.encoder_kernel_size - 1) // 2,
                    dilation=1,
                    bias=False
                ),
                nn.BatchNorm1d(config.model.encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            convolutions.append(conv_layer)
        
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            config.model.encoder_embedding_dim,
            config.model.encoder_embedding_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, text, text_lengths):
        # Text embedding
        embedded = self.embedding(text)
        embedded = embedded.transpose(1, 2)
        
        # Convolutional layers
        for conv in self.convolutions:
            embedded = conv(embedded)
        
        embedded = embedded.transpose(1, 2)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        outputs, _ = self.lstm(packed)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs

class Prenet(nn.Module):
    """Prenet for decoder"""
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    """Decoder for Tacotron 2"""
    def __init__(self, config):
        super().__init__()
        
        self.n_mels = config.audio.n_mels
        self.encoder_embedding_dim = config.model.encoder_embedding_dim
        self.attention_dim = config.model.attention_dim
        
        # Prenet
        self.prenet = Prenet(
            self.n_mels,
            config.model.prenet_dim
        )
        
        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            config.model.prenet_dim + self.encoder_embedding_dim,
            config.model.attention_dim
        )
        
        # Attention
        self.attention = LocationSensitiveAttention(
            config.model.attention_dim,
            self.encoder_embedding_dim,
            config.model.attention_location_n_filters,
            config.model.attention_location_kernel_size
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            config.model.attention_dim + self.encoder_embedding_dim,
            config.model.decoder_rnn_dim
        )
        
        # Linear projections
        self.mel_projection = nn.Linear(
            config.model.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mels
        )
        
        self.gate_projection = nn.Linear(
            config.model.decoder_rnn_dim + self.encoder_embedding_dim,
            1
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.model.p_decoder_dropout)
    
    def forward(self, encoder_outputs, mels, memory_lengths=None):
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)
        
        # Initialize states
        attention_hidden = torch.zeros(batch_size, self.attention_dim).to(encoder_outputs.device)
        attention_cell = torch.zeros(batch_size, self.attention_dim).to(encoder_outputs.device)
        decoder_hidden = torch.zeros(batch_size, self.model.decoder_rnn_dim).to(encoder_outputs.device)
        decoder_cell = torch.zeros(batch_size, self.model.decoder_rnn_dim).to(encoder_outputs.device)
        
        # Initialize attention weights
        attention_weights = torch.zeros(batch_size, max_len).to(encoder_outputs.device)
        attention_weights_cum = torch.zeros(batch_size, max_len).to(encoder_outputs.device)
        attention_context = torch.zeros(batch_size, self.encoder_embedding_dim).to(encoder_outputs.device)
        
        # Process memory for attention
        processed_memory = self.attention.memory_layer(encoder_outputs)
        
        # Mask for attention
        if memory_lengths is not None:
            mask = self._get_mask(max_len, memory_lengths)
        
        # Outputs
        mel_outputs, gate_outputs, alignments = [], [], []
        
        # Decode step by step
        for i in range(mels.size(1)):
            # Prenet
            prenet_input = mels[:, i, :] if i > 0 else torch.zeros_like(mels[:, 0, :])
            prenet_output = self.prenet(prenet_input)
            
            # Attention RNN
            attention_rnn_input = torch.cat([prenet_output, attention_context], dim=-1)
            attention_hidden, attention_cell = self.attention_rnn(
                attention_rnn_input, (attention_hidden, attention_cell)
            )
            attention_hidden = self.dropout(attention_hidden)
            
            # Attention
            attention_weights_cat = torch.cat([
                attention_weights.unsqueeze(1),
                attention_weights_cum.unsqueeze(1)
            ], dim=1)
            
            attention_context, attention_weights = self.attention(
                attention_hidden, encoder_outputs, processed_memory,
                attention_weights_cat, mask
            )
            
            attention_weights_cum += attention_weights
            
            # Decoder RNN
            decoder_rnn_input = torch.cat([attention_hidden, attention_context], dim=-1)
            decoder_hidden, decoder_cell = self.decoder_rnn(
                decoder_rnn_input, (decoder_hidden, decoder_cell)
            )
            decoder_hidden = self.dropout(decoder_hidden)
            
            # Projections
            decoder_output = torch.cat([decoder_hidden, attention_context], dim=1)
            mel_output = self.mel_projection(decoder_output)
            gate_output = self.gate_projection(decoder_output)
            
            # Store outputs
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)
        
        # Stack outputs
        mel_outputs = torch.stack(mel_outputs, dim=1)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        
        return mel_outputs, gate_outputs, alignments
    
    def _get_mask(self, max_len, lengths):
        """Create mask for padded sequences"""
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(lengths.device)
        mask = mask >= lengths.unsqueeze(1)
        return mask

class Postnet(nn.Module):
    """Postnet for refining mel-spectrograms"""
    def __init__(self, config):
        super().__init__()
        
        self.n_mels = config.audio.n_mels
        
        convolutions = []
        for i in range(config.model.postnet_n_convolutions):
            in_channels = self.n_mels if i == 0 else config.model.postnet_embedding_dim
            out_channels = self.n_mels if i == config.model.postnet_n_convolutions - 1 else config.model.postnet_embedding_dim
            
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=config.model.postnet_kernel_size,
                    stride=1,
                    padding=(config.model.postnet_kernel_size - 1) // 2,
                    dilation=1,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.Tanh() if i < config.model.postnet_n_convolutions - 1 else nn.Identity()
            )
            convolutions.append(conv_layer)
        
        self.convolutions = nn.ModuleList(convolutions)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = self.dropout(conv(x))
        
        return x.transpose(1, 2)

class Tacotron2(nn.Module):
    """Complete Tacotron 2 model"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.postnet = Postnet(config)
        
    def forward(self, text, text_lengths, mels, max_decoder_steps=None):
        # Encode text
        encoder_outputs = self.encoder(text, text_lengths)
        
        # Decode mel-spectrogram
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, text_lengths
        )
        
        # Postnet processing
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments
    
    def inference(self, text):
        """Inference mode for text-to-speech synthesis"""
        # TODO: Implement inference logic
        pass
