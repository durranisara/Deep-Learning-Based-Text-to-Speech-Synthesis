import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocationSensitiveAttention(nn.Module):
    """
    Location Sensitive Attention mechanism for Tacotron 2
    """
    def __init__(self, attention_dim, embedding_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super().__init__()
        
        self.query_layer = nn.Linear(attention_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        self.location_conv = nn.Conv1d(
            2, attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=(attention_location_kernel_size - 1) // 2,
            bias=False
        )
        
        self.location_layer = nn.Linear(
            attention_location_n_filters, attention_dim, bias=False
        )
        
        self.score_mask_value = -float("inf")
        
    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        Compute alignment energies
        """
        # Process query
        processed_query = self.query_layer(query.unsqueeze(1))
        
        # Process memory
        processed_memory = self.memory_layer(processed_memory)
        
        # Process location features
        processed_attention_weights = self.location_conv(attention_weights_cat)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_layer(processed_attention_weights)
        
        # Compute energies
        energies = self.v(
            torch.tanh(processed_query + processed_memory + processed_attention_weights)
        )
        
        energies = energies.squeeze(-1)
        return energies
    
    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        Forward pass
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )
        
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
        
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        
        return attention_context, attention_weights
