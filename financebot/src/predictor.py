# This script provides functions for running inference with the trained
# Transformer model, specifically for generating autoregressive predictions.

import torch # PyTorch library for model inference
import financebot.src.config as config # Import application configuration settings
import numpy as np # For numerical operations and array handling

# Function to generate predictions from the model in an autoregressive manner
# This means predicting one step, then using that prediction as part of the input
# to predict the next step, and so on.
def predict_autoregressive(model, input_sequence_norm,n_pred = config.N_PRED_INFERENCE):
    """
    Generates predictions autoregressively.

    Args:
        model: The trained TransformerModel instance.
        input_sequence_norm: Normalized input history, shape (seq_len, input_size).
                             This is the initial sequence fed to the encoder.
        n_pred: The number of future steps to predict.

    Returns:
        List of numpy arrays, where each array represents the predicted values
        for a single future time step (shape: input_size).
    """
    # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()
    predictions_list = [] # List to store the predictions for each future step
    device = config.DEVICE # Get the device (CPU/GPU) from config
    #seq_len = config.SEQ_LEN # Sequence length (length of input_sequence_norm)
    input_size = config.INPUT_SIZE # Number of features per time step

    # Prepare the initial input sequence for the model's encoder
    # Convert numpy array to PyTorch tensor, add a batch dimension (size 1), and move to device
    current_input = torch.tensor(
        input_sequence_norm, dtype=torch.float32).unsqueeze(0).to(device)
    # current_input shape is now: (1, seq_len, input_size)

    # Initialize the decoder input sequence for the first prediction step
    # It starts with the last time step of the input sequence (teacher forcing concept for inference)
    last_known_value = current_input[:, -1:, :]  # Take the last element along the sequence dimension. Shape: (1, 1, input_size)
    decoder_input = last_known_value # The decoder input starts with this single time step

    #print(f"Starting autoregressive prediction for {n_pred} steps...") # Optional print
    # Disable gradient calculation during inference for efficiency
    with torch.no_grad():
        # Loop to generate predictions for each future step up to n_pred
        for i in range(n_pred):
            # Get the current length of the sequence being fed to the decoder
            tgt_sequence_length = decoder_input.size(1)
            # Generate the causal mask for the decoder input based on its current length
            tgt_mask = model.get_tgt_mask(tgt_sequence_length).to(device)

            # Perform a forward pass through the model
            # The encoder receives the full initial input sequence.
            # The decoder receives the sequence built so far (last known value + previous predictions).
            # The mask prevents the decoder from seeing 'future' steps in its own input.
            output = model(current_input, decoder_input, tgt_mask=tgt_mask)

            # Extract the prediction for the *next* time step
            # The model's output for decoder_input sequence of length T_current has length T_current.
            # The prediction for the next step is the output corresponding to the *last* element of decoder_input.
            last_pred_step_tensor = output[:, -1:, :] # Shape: (1, 1, input_size)

            # Convert the single-step prediction tensor to a numpy array and add to the list
            prediction_np = last_pred_step_tensor.squeeze().cpu().numpy()  # Squeeze removes the size 1 dimensions. Shape: (input_size,)
            predictions_list.append(prediction_np)

            # --- Autoregressive step: Use the predicted step as input for the next iteration ---
            # Concatenate the newly predicted step to the decoder input sequence
            decoder_input = torch.cat(
                [decoder_input, last_pred_step_tensor], dim=1)
            # decoder_input shape grows by 1 in the sequence dimension in each iteration

    # Return the list containing the prediction (as numpy arrays) for each future step
    return predictions_list


# Function to perform inference on "real" input data, handling normalization and denormalization
def infer_realdata(model,inputdata,predlength):
    """
    Handles normalization, runs autoregressive prediction, and denormalizes results
    for a given input data sequence.

    Args:
        model: The trained TransformerModel.
        inputdata: The input data sequence (likely raw or pre-formatted, including date),
                   shape (n_features_incl_date, seq_len).
        predlength: The number of future steps to predict.

    Returns:
        NumPy array of denormalized predictions, shape (predlength, input_size).
    """
    # --- Normalize the input data sequence ---
    # Create a copy to avoid modifying the original input array
    data_values = inputdata.copy()
    # Select only the feature data (excluding the date row, index 0) and ensure float type
    data_values = data_values[1:].astype(
        float)  # Shape: (input_size, seq_len)

    # Calculate mean and standard deviation for normalization using only the input sequence data
    mean = data_values.mean(axis=1, keepdims=True) # Mean for each feature, shape (input_size, 1)
    std = data_values.std(axis=1, keepdims=True) + 1e-6 # Std dev for each feature, add epsilon to prevent division by zero
    # Apply Z-score normalization
    norm_values = (data_values - mean) / std  # Shape: (input_size, seq_len)

    # --- Prepare for Prediction ---
    # Transpose the normalized data to match the expected input shape for predict_autoregressive
    # Expected shape: (seq_len, input_size)
    norm_sequence_transposed = norm_values.T

    # --- Run Autoregressive Prediction ---
    # Call the prediction function with the normalized input and desired prediction length
    predictions_norm_list = predict_autoregressive(model, norm_sequence_transposed,predlength)

    # --- Denormalize Predictions ---
    # Convert the list of normalized prediction arrays into a single NumPy array
    predictions_norm_array = np.array(predictions_norm_list)  # Shape: (predlength, input_size)
    # Denormalize the predictions back to the original data scale
    # Using the mean and std calculated from the input sequence.
    # Broadcasting mean.T/std.T (shape 1, input_size) with predictions_norm_array (shape predlength, input_size)
    predictions_denorm = predictions_norm_array *std.T + mean.T  # Final shape: (predlength, input_size)

    # Return the denormalized predictions
    return predictions_denorm