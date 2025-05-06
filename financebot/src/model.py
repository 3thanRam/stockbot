# This script defines the Transformer model architecture and related components
# for time series prediction, including Positional Encoding and model saving/loading.

import torch # PyTorch library for building and training neural networks
import torch.nn as nn # Neural network modules from PyTorch
import math # Standard Python math library (used for sine/cosine and sqrt)
import os # For interacting with the operating system (e.g., checking file existence)
# Import application configuration settings
import financebot.src.config as config


# Class to implement standard Positional Encoding
# This module adds positional information to the input embeddings,
# crucial for sequence models like Transformers that lack inherent order awareness.
class PositionalEncoding(nn.Module):
    """ Standard Positional Encoding """

    # Constructor: Initializes the positional encoding table and dropout layer
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_p)
        # Create a buffer for the positional encoding table (max_len x dim_model)
        pos_encoding = torch.zeros(max_len, dim_model)
        # Create a tensor of positions (0 to max_len-1)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1) # Shape: (max_len, 1)
        # Calculate the division term for the positional encoding formula
        division_term = torch.exp(torch.arange(
            0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # Shape: (dim_model/2,)
        # Apply sine function to even indices of the encoding
        # Broadcasting positions_list (max_len, 1) with division_term (dim_model/2,)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # Apply cosine function to odd indices of the encoding
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # Add a batch dimension at the beginning for easy broadcasting with input batches
        # Shape: (1, max_len, dim_model)
        pos_encoding = pos_encoding.unsqueeze(0)
        # Register the positional encoding as a buffer; it's not a learnable parameter
        # and should persist with the model state but not require gradients.
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)

    # Forward pass: adds the positional encoding to the input token embedding
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # token_embedding shape: (batch_size, seq_len, dim_model)
        # self.pos_encoding shape: (1, max_len, dim_model)
        # Add positional encoding to the token embedding.
        # Slice the pos_encoding to match the sequence length of the current batch.
        # Broadcasting applies the same encoding to all items in the batch.
        # Shape: (batch_size, seq_len, dim_model)
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])


# The main Transformer model class for time series data
# Uses PyTorch's built-in Transformer module.
class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.
    Assumes batch_first=True.
    """

    # Constructor: Initializes the layers of the Transformer model
    def __init__(
        self,
        input_size=config.INPUT_SIZE, # Number of input features
        dim_model=config.DIM_MODEL, # Dimension of the model's internal representations
        num_heads=config.NUM_HEADS, # Number of attention heads
        num_encoder_layers=config.NUM_ENCODER_LAYERS, # Number of encoder layers
        num_decoder_layers=config.NUM_DECODER_LAYERS, # Number of decoder layers
        dropout_p=config.DROPOUT_P, # Dropout probability
        max_len=config.MAX_LEN_POSITIONAL_ENCODING # Maximum sequence length for positional encoding
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.input_size = input_size

        # Linear layer to project input features to the model dimension
        self.input_linear = nn.Linear(input_size, dim_model)
        # Positional encoding layer
        self.positional_encoder = PositionalEncoding(
            dim_model, dropout_p, max_len)
        # PyTorch's standard Transformer module
        self.transformer = nn.Transformer(
            d_model=dim_model, # Input/output feature dimension
            nhead=num_heads, # Number of attention heads
            num_encoder_layers=num_encoder_layers, # Number of layers in the encoder
            num_decoder_layers=num_decoder_layers, # Number of layers in the decoder
            dim_feedforward=dim_model * 4, # Dimension of the feedforward networks
            dropout=dropout_p, # Dropout probability within the Transformer
            activation='relu', # Activation function for feedforward networks
            batch_first=True  # Specify that the batch dimension is the first dimension
        )
        # Linear layer to project the Transformer output back to the original input feature size
        self.out_linear = nn.Linear(dim_model, input_size)

    # Forward pass: Defines how data flows through the model
    # src: Source sequence (input to encoder)
    # tgt: Target sequence (input to decoder, often shifted)
    # Masks and padding masks are used to control attention
    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass assuming batch_first=True.
        src: (batch_size, src_seq_len, input_size)
        tgt: (batch_size, tgt_seq_len, input_size)
        tgt_mask: (tgt_seq_len, tgt_seq_len) - Look-ahead mask for the decoder
        """
        # 1. Embed inputs: Project input features to the model dimension and scale
        # The scaling factor (sqrt(dim_model)) is standard practice in Transformers
        src = self.input_linear(src) * math.sqrt(self.dim_model)
        tgt = self.input_linear(tgt) * math.sqrt(self.dim_model)

        # 2. Add positional encoding to the embedded inputs
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # 3. Pass through the Transformer module
        # Masks are applied internally by the transformer layer
        transformer_out = self.transformer(src, tgt,
                                           tgt_mask=tgt_mask, # Causal mask for the decoder
                                           src_key_padding_mask=src_key_padding_mask, # Mask for ignoring padding in source
                                           tgt_key_padding_mask=tgt_key_padding_mask, # Mask for ignoring padding in target
                                           memory_key_padding_mask=memory_key_padding_mask) # Mask for ignoring padding in encoder output (memory)
        # transformer_out shape: (batch_size, tgt_seq_len, dim_model) (output from decoder)

        # Project the Transformer output back to the original input feature size
        output = self.out_linear(transformer_out)
        # output shape: (batch_size, tgt_seq_len, input_size)

        # Return the final predicted output sequence
        return output

    # Static method to generate a causal (look-ahead) mask for the decoder
    # This mask prevents the decoder from attending to future time steps.
    @staticmethod
    def get_tgt_mask(size) -> torch.Tensor:
        """Generates a square causal mask for the sequence."""
        # Create an upper triangular matrix filled with negative infinity
        # Diagonal=1 means masking elements at index (i, j) where j > i
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        # Move the mask to the configured device (CPU/GPU)
        return mask.to(config.DEVICE)

    # Method to save the model's learnable parameters (state dictionary)
    def save_model(self, path=config.MODEL_SAVE_PATH):
        """Saves the model state dictionary."""
        try:
            # Save the state dictionary to the specified path
            torch.save(self.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

    # Method to load a saved model state dictionary
    def load_model(self, path=config.MODEL_SAVE_PATH):
        """Loads the model state dictionary."""
        # Check if the model file exists
        if os.path.exists(path):
            try:
                # Load the state dictionary from the file
                # map_location ensures it's loaded onto the correct device
                self.load_state_dict(torch.load(
                    path, map_location=config.DEVICE))
                self.eval()  # Set the model to evaluation mode (disables dropout, batchnorm updates)
                print(f"Model loaded from {path}")
                return True # Return True on successful load
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
                return False # Return False if loading fails
        else:
            print(f"Model file not found at {path}")
            return False # Return False if the file doesn't exist