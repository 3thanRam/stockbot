import torch
import config  # Import configuration
import numpy as np

def predict_autoregressive(model, input_sequence_norm,n_pred = config.N_PRED_INFERENCE):
    """
    Generates predictions autoregressively.

    Args:
        model: The trained TransformerModel.
        input_sequence_norm: Normalized input history, shape (seq_len, input_size).

    Returns:
        List of numpy arrays, each representing a predicted step (shape: input_size).
    """
    model.eval()
    predictions_list = []
    device = config.DEVICE
    seq_len = config.SEQ_LEN
    #n_pred = input_sequence_norm.shape[1]#config.N_PRED_INFERENCE
    
    input_size = config.INPUT_SIZE

    # Prepare initial encoder input
    current_input = torch.tensor(
        input_sequence_norm, dtype=torch.float32).unsqueeze(0).to(device)
    # current_input shape: (1, seq_len, input_size)

    # Initialize decoder input with the last known value
    last_known_value = current_input[:, -1:, :]  # Shape: (1, 1, input_size)
    decoder_input = last_known_value

    #print(f"Starting autoregressive prediction for {n_pred} steps...")
    with torch.no_grad():
        for i in range(n_pred):
            tgt_sequence_length = decoder_input.size(
                1)  # Current length of decoder input
            tgt_mask = model.get_tgt_mask(tgt_sequence_length).to(device)

            # Model forward pass: src=(1,S,E), tgt=(1,T_current,E) -> output=(1,T_current,E)
            output = model(current_input, decoder_input, tgt_mask=tgt_mask)

            # Get the *last* time step from the output sequence
            # output[:, -1:, :] shape: (1, 1, input_size)
            last_pred_step_tensor = output[:, -1:, :]

            prediction_np = last_pred_step_tensor.squeeze().cpu().numpy()  # Shape: (input_size,)
            predictions_list.append(prediction_np)

            decoder_input = torch.cat(
                [decoder_input, last_pred_step_tensor], dim=1)


    return predictions_list


def infer_realdata(model,inputdata,predlength):
    
    # Normalize using stats ONLY from this input sequence
    data_values = inputdata.copy()
    data_values = data_values[1:].astype(
        float)  # Features only, ensure float
    
    mean = data_values.mean(axis=1, keepdims=True)
    std = data_values.std(axis=1, keepdims=True) + 1e-6
    norm_values = (data_values - mean) / std  # Shape: (input_size, seq_len)
    
    # Transpose to (seq_len, input_size) for predictor input format
    norm_sequence_transposed = norm_values.T
    predictions_norm_list = predict_autoregressive(model, norm_sequence_transposed,predlength)
    predictions_norm_array = np.array(predictions_norm_list)  # Shape: (N_pred, input_size)
    predictions_denorm = predictions_norm_array *std.T + mean.T  # Shape: (N_pred, input_size)
    return predictions_denorm