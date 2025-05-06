# This script contains functions for training the Transformer model,
# including epoch loops, curriculum learning logic, and data batching.

import torch # PyTorch library for building and training neural networks
import financebot.src.config as config # Import application configuration settings
import numpy as np # For numerical operations (though less used directly here)


# Helper function to determine the prediction horizon for the current epoch
# based on the defined curriculum schedule.
def get_current_horizon(epoch, schedule):
    """
    Gets the training horizon (number of steps to predict) based on the
    curriculum schedule for the given epoch.
    """
    current_h = 1 # Initialize with a minimum horizon
    # Sort the schedule keys (epochs) to ensure we check them in increasing order
    sorted_epochs = sorted(schedule.keys())
    # Iterate through the sorted start epochs in the schedule
    for start_epoch in sorted_epochs:
        # If the current epoch is greater than or equal to the start epoch in the schedule
        if epoch >= start_epoch:
            # Update the current horizon to the value specified for this epoch range
            current_h = schedule[start_epoch]
        else:
            # If the current epoch is less than the start epoch, we've passed the relevant range
            # for this epoch, so we can stop checking.
            break
    # Return the determined training horizon for the current epoch
    return current_h


# Function to perform a single epoch of training
def train_loop(model, opt, loss_fn, dataloader,current_horizon):
    """
    Performs one epoch of training over the entire training dataset.

    Args:
        model: The TransformerModel instance.
        opt: The optimizer.
        loss_fn: The loss function.
        dataloader: DataLoader for the training data batches.
        current_horizon: The prediction horizon to use for this training epoch
                         (determined by the curriculum schedule).

    Returns:
        The average training loss for the epoch.
    """
    # Set the model to training mode (enables dropout, batchnorm updates, etc.)
    model.train()
    total_loss = 0 # Accumulator for the total loss across all batches
    processed_batches = 0 # Counter for the number of batches processed
    device = config.DEVICE # Get the device (CPU/GPU) from config
    seq_len = config.SEQ_LEN # Input sequence length from config
    max_horizon = config.N_PRED_TRAINING_MAX # Maximum prediction horizon from config
    input_size = config.INPUT_SIZE # Number of input features from config

    # Iterate through batches provided by the dataloader
    for X_enc, X_dec_in, Y_tgt in dataloader:
        # Skip empty batches if any
        if X_enc.size(0) == 0:
            continue

        # Move the batch data to the specified device (CPU/GPU)
        X_enc = X_enc.to(device)
        X_dec_in = X_dec_in.to(device)
        Y_tgt = Y_tgt.to(device)

        # Reshape the flattened input tensors back to their original sequence shapes
        # X_enc: (batch_size * seq_len * input_size) -> (batch_size, seq_len, input_size)
        src = X_enc.view(-1, seq_len, input_size)
        # X_dec_in, Y_tgt: (batch_size * max_horizon * input_size) -> (batch_size, max_horizon, input_size)
        tgt_full = X_dec_in.view(-1, max_horizon, input_size) # Full decoder input (teacher forcing)
        y_expected_full = Y_tgt.view(-1, max_horizon, input_size) # Full target output

        # --- Apply Curriculum Learning ---
        # Slice the decoder input and target output to match the current training horizon
        tgt_sliced = tgt_full[:, :current_horizon, :]           # Shape: (N, current_h, E)
        y_expected_sliced = y_expected_full[:, :current_horizon, :] # Shape: (N, current_h, E)

        # Create the causal mask for the decoder, sized according to the current training horizon
        tgt_mask = model.get_tgt_mask(current_horizon).to(device)

        # Perform the forward pass through the model
        # src (encoder input): (N, S, E)
        # tgt_sliced (decoder input): (N, current_h, E)
        # tgt_mask: (current_h, current_h)
        # The model returns predictions for the tgt_sliced sequence. Output shape: (N, current_h, E)
        pred = model(src, tgt_sliced, tgt_mask=tgt_mask)

        # Calculate the loss between the model's predictions and the actual target values
        loss = loss_fn(pred, y_expected_sliced)

        # --- Backpropagation ---
        opt.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients of the loss with respect to model parameters
        opt.step() # Update model parameters using the optimizer

        total_loss += loss.item()  # Add the scalar loss value to the total loss
        processed_batches += 1 # Increment the batch counter

    # Calculate and return the average loss for the epoch
    return total_loss / processed_batches if processed_batches > 0 else 0.0


# Function to perform a single epoch of validation
def validation_loop(model, loss_fn, dataloader, validation_horizon):
    """
    Performs one epoch of validation over the entire validation dataset.
    No gradient calculations or parameter updates occur here.

    Args:
        model: The TransformerModel instance.
        loss_fn: The loss function.
        dataloader: DataLoader for the validation data batches.
        validation_horizon: The fixed prediction horizon to use for validation.

    Returns:
        The average validation loss for the epoch.
    """
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    total_loss = 0 # Accumulator for total loss
    processed_batches = 0 # Counter for batches
    device = config.DEVICE # Get device
    seq_len = config.SEQ_LEN # Input sequence length
    max_horizon = config.N_PRED_TRAINING_MAX # Maximum prediction horizon
    input_size = config.INPUT_SIZE # Number of input features

    # Disable gradient calculations within this block
    with torch.no_grad():
        # Iterate through batches in the validation dataloader
        for X_enc, X_dec_in, Y_tgt in dataloader:
            # Skip empty batches
            if X_enc.size(0) == 0:
                continue

            # Move data to device
            X_enc = X_enc.to(device)
            X_dec_in = X_dec_in.to(device)
            Y_tgt = Y_tgt.to(device)

            # Reshape inputs
            src = X_enc.view(-1, seq_len, input_size)
            # tgt_full/y_expected_full: (N, T_max, E)
            tgt_full = X_dec_in.view(-1, max_horizon, input_size)
            y_expected_full = Y_tgt.view(-1, max_horizon, input_size)

            # Slice decoder input and target based on the fixed validation horizon
            tgt_val = tgt_full[:, :validation_horizon, :]       # Shape: (N, val_h, E)
            y_expected_val = y_expected_full[:, :validation_horizon, :] # Shape: (N, val_h, E)

            # Create causal mask for the validation horizon
            tgt_mask = model.get_tgt_mask(validation_horizon).to(device)

            # Forward pass (no gradient calculation)
            pred = model(src, tgt_val, tgt_mask=tgt_mask) # Output shape: (N, val_h, E)

            # Calculate loss
            loss = loss_fn(pred, y_expected_val)
            total_loss += loss.item()
            processed_batches += 1

    # Calculate and return the average validation loss
    return total_loss / processed_batches if processed_batches > 0 else 0.0


# Main function to train the model over multiple epochs
def fit(model, train_dataloader, val_dataloader, optimizer, loss_fn):
    """
    Trains the model for a specified number of epochs, performing training
    and validation loops and applying curriculum learning.

    Args:
        model: The TransformerModel instance.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        optimizer: The optimizer.
        loss_fn: The loss function.

    Returns:
        Lists of training and validation losses per epoch.
    """
    train_loss_list, validation_loss_list = [], [] # Lists to store loss history
    epochs = config.EPOCHS # Total number of epochs from config
    schedule = config.CURRICULUM_SCHEDULE # Curriculum schedule from config
    # Use the maximum training horizon for consistent validation evaluation
    validation_horizon = config.N_PRED_TRAINING_MAX
    print(f"Starting training for {epochs} epochs with curriculum learning...")
    print(f"Curriculum Schedule (Epoch: Horizon): {schedule}")
    print(f"Validation Horizon: {validation_horizon}")

    # Loop through each epoch
    for epoch in range(epochs):
        # --- Determine the prediction horizon for the current training epoch ---
        current_train_horizon = get_current_horizon(epoch, schedule)

        # Print epoch information including the current training horizon
        print("-" * 25, f"Epoch {epoch + 1}/{epochs} (Train Horizon: {current_train_horizon})", "-" * 25)

        # --- Perform the training loop for the current epoch ---
        # Pass the model, optimizer, loss function, training data, and the current training horizon
        train_loss = train_loop(model, optimizer, loss_fn, train_dataloader, current_train_horizon)
        train_loss_list.append(train_loss) # Store the training loss

        # --- Perform the validation loop for the current epoch ---
        # Pass the model, loss function, validation data, and the fixed validation horizon
        validation_loss = validation_loop(model, loss_fn, val_dataloader, validation_horizon)
        validation_loss_list.append(validation_loss) # Store the validation loss

        # Print the training and validation losses for the epoch
        print(f"Training loss (h={current_train_horizon}): {train_loss:.6f}")
        print(f"Validation loss (h={validation_horizon}): {validation_loss:.6f}")
        # Print a separator line
        print("-" * (50 + len(str(epoch+1)) + len(str(epochs)) + len(str(current_train_horizon)) + 10)) # Adjust separator length

    print("Training finished.")
    # Return the lists of training and validation losses
    return train_loss_list, validation_loss_list


# Function to divide the dataset into batches for training or validation
def batchify_data(data, batch_size=config.BATCH_SIZE):
    """
    Batches the training/validation data tuple into a list of batches.

    Args:
        data: A tuple containing the three data components
              (X_enc_data, X_dec_in_data, Y_tgt_data),
              where each component is a tensor of flattened sequences.
        batch_size: The desired size of each batch.

    Returns:
        A list of tuples, where each tuple is a batch
        (X_enc_batch, X_dec_in_batch, Y_tgt_batch).
        Returns an empty list if data is empty or inconsistent.
    """
    batches = [] # List to store the generated batches
    X_enc_data, X_dec_in_data, Y_tgt_data = data # Unpack the data tuple

    # Check if the three data components have the same number of samples
    if not (len(X_enc_data) == len(X_dec_in_data) == len(Y_tgt_data)):
        print("Error: Data components have different lengths. Cannot batch.")
        return [] # Return empty list on error
    # Check if there is any data to batchify
    if len(X_enc_data) == 0:
        print("No data to batchify.")
        return [] # Return empty list if no data

    num_samples = len(X_enc_data) # Total number of sequences/samples
    # Iterate through the data, creating batches of size batch_size
    for i in range(0, num_samples, batch_size):
        # Slice the data tensors to get the current batch
        X_enc_batch = X_enc_data[i:i + batch_size]
        X_dec_in_batch = X_dec_in_data[i:i + batch_size]
        Y_tgt_batch = Y_tgt_data[i:i + batch_size]
        # If the batch is not empty (handles the last batch which might be smaller)
        if len(X_enc_batch) > 0:
            batches.append((X_enc_batch, X_dec_in_batch, Y_tgt_batch)) # Append the batch tuple to the list

    # Print the number of batches created
    print(f"Created {len(batches)} batches of size <= {batch_size}")
    # Return the list of batches
    return batches