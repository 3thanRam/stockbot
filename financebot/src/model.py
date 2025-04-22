import torch
import torch.nn as nn
import math
import os
import config


class PositionalEncoding(nn.Module):
    """ Standard Positional Encoding """

    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(
            0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # Original code added batch dim then transposed. Let's align with batch_first=True directly.
        # Shape: (1, max_len, dim_model) - Ready for broadcasting with (N, S, E)
        pos_encoding = pos_encoding.unsqueeze(0)
        # No need to save PE in state_dict
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # token_embedding shape: (batch_size, seq_len, dim_model)
        # self.pos_encoding shape: (1, max_len, dim_model)
        # Slicing ensures we only use encoding up to the current sequence length
        # Shape: (batch_size, seq_len, dim_model)
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    Assumes batch_first=True.
    """

    def __init__(
        self,
        input_size=config.INPUT_SIZE,
        dim_model=config.DIM_MODEL,
        num_heads=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dropout_p=config.DROPOUT_P,
        max_len=config.MAX_LEN_POSITIONAL_ENCODING
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.input_size = input_size

        self.input_linear = nn.Linear(input_size, dim_model)
        self.positional_encoder = PositionalEncoding(
            dim_model, dropout_p, max_len)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_model * 4,
            dropout=dropout_p,
            activation='relu',
            batch_first=True  # Explicitly set batch_first
        )
        self.out_linear = nn.Linear(dim_model, input_size)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass assuming batch_first=True.
        src: (batch_size, src_seq_len, input_size)
        tgt: (batch_size, tgt_seq_len, input_size)
        tgt_mask: (tgt_seq_len, tgt_seq_len) - Look-ahead mask
        """
        # 1. Embed inputs
        src = self.input_linear(src) * math.sqrt(self.dim_model)
        tgt = self.input_linear(tgt) * math.sqrt(self.dim_model)

        # 2. Add positional encoding
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # 3. Pass through Transformer
        transformer_out = self.transformer(src, tgt,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask)
        # transformer_out shape: (batch_size, tgt_seq_len, dim_model)

        output = self.out_linear(transformer_out)
        # output shape: (batch_size, tgt_seq_len, input_size)

        return output  # Return directly as (N, T, E)

    @staticmethod
    def get_tgt_mask(size) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask.to(config.DEVICE)

    def save_model(self, path=config.MODEL_SAVE_PATH):
        """Saves the model state dictionary."""
        try:
            torch.save(self.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

    def load_model(self, path=config.MODEL_SAVE_PATH):
        """Loads the model state dictionary."""
        if os.path.exists(path):
            try:
                # Load state dict onto the correct device
                self.load_state_dict(torch.load(
                    path, map_location=config.DEVICE))
                self.eval()  # Set model to evaluation mode
                print(f"Model loaded from {path}")
                return True
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                return False
        else:
            print(f"Model file not found at {path}")
            return False
