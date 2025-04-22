import torch
import config
import numpy as np


def get_current_horizon(epoch, schedule):
    """Gets the training horizon based on the curriculum schedule."""
    current_h = 1 # Default minimum horizon
    # Sort schedule keys to ensure correct order
    sorted_epochs = sorted(schedule.keys())
    for start_epoch in sorted_epochs:
        if epoch >= start_epoch:
            current_h = schedule[start_epoch]
        else:
            # Stop checking once we pass the current epoch's range
            break
    return current_h


def train_loop(model, opt, loss_fn, dataloader,current_horizon):
    """ Performs one epoch of training. """
    model.train()
    total_loss = 0
    processed_batches = 0
    device = config.DEVICE
    seq_len = config.SEQ_LEN
    max_horizon = config.N_PRED_TRAINING_MAX
    input_size = config.INPUT_SIZE

    for X_enc, X_dec_in, Y_tgt in dataloader:
        if X_enc.size(0) == 0:
            continue

        X_enc = X_enc.to(device)
        X_dec_in = X_dec_in.to(device)
        Y_tgt = Y_tgt.to(device)

        # Reshape inputs (batch_first=True)
        # src: (N, S, E)
        src = X_enc.view(-1, seq_len, input_size)
        # tgt_full/y_expected_full: (N, T_max, E)
        tgt_full = X_dec_in.view(-1, max_horizon, input_size)
        y_expected_full = Y_tgt.view(-1, max_horizon, input_size)
        
        # --- Slice targets and decoder inputs based on current curriculum horizon ---
        tgt_sliced = tgt_full[:, :current_horizon, :]           # Shape: (N, current_h, E)
        y_expected_sliced = y_expected_full[:, :current_horizon, :] # Shape: (N, current_h, E)

        # Create decoder target mask FOR THE CURRENT HORIZON
        tgt_mask = model.get_tgt_mask(current_horizon).to(device) # Use current_horizon

        # Forward pass - Model expects (N, S, E), (N, current_h, E) -> returns (N, current_h, E)
        pred = model(src, tgt_sliced, tgt_mask=tgt_mask)

        # Calculate loss
        loss = loss_fn(pred, y_expected_sliced)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()  # Use item() to get scalar value
        processed_batches += 1

    return total_loss / processed_batches if processed_batches > 0 else 0.0


def validation_loop(model, loss_fn, dataloader, validation_horizon):
    """ Performs one epoch of validation. """
    model.eval()
    total_loss = 0
    processed_batches = 0
    device = config.DEVICE
    seq_len = config.SEQ_LEN
    max_horizon = config.N_PRED_TRAINING_MAX
    input_size = config.INPUT_SIZE

    with torch.no_grad():
        for X_enc, X_dec_in, Y_tgt in dataloader:
            if X_enc.size(0) == 0:
                continue

            X_enc = X_enc.to(device)
            X_dec_in = X_dec_in.to(device)
            Y_tgt = Y_tgt.to(device)

            # Reshape inputs
            src = X_enc.view(-1, seq_len, input_size)
            # tgt_full/y_expected_full: (N, T_max, E)
            tgt_full = X_dec_in.view(-1, max_horizon, input_size)
            y_expected_full = Y_tgt.view(-1, max_horizon, input_size)

            tgt_val = tgt_full[:, :validation_horizon, :]       # Shape: (N, val_h, E)
            y_expected_val = y_expected_full[:, :validation_horizon, :] # Shape: (N, val_h, E)

            # Create mask for the validation horizon
            tgt_mask = model.get_tgt_mask(validation_horizon).to(device)

            # Forward pass
            pred = model(src, tgt_val, tgt_mask=tgt_mask) # Output shape: (N, val_h, E)

            # Calculate loss using validation slices
            loss = loss_fn(pred, y_expected_val)
            total_loss += loss.item()
            processed_batches += 1

    return total_loss / processed_batches if processed_batches > 0 else 0.0


def fit(model, train_dataloader, val_dataloader, optimizer, loss_fn):
    """ Trains the model for a number of epochs. """
    train_loss_list, validation_loss_list = [], []
    epochs = config.EPOCHS
    schedule = config.CURRICULUM_SCHEDULE
    # Use the max horizon for validation consistently
    validation_horizon = config.N_PRED_TRAINING_MAX
    print(f"Starting training for {epochs} epochs with curriculum learning...")
    print(f"Curriculum Schedule (Epoch: Horizon): {schedule}")
    print(f"Validation Horizon: {validation_horizon}")
    for epoch in range(epochs):
        # --- Determine current training horizon ---
        current_train_horizon = get_current_horizon(epoch, schedule)

        print("-" * 25, f"Epoch {epoch + 1}/{epochs} (Train Horizon: {current_train_horizon})", "-" * 25)

        # --- Pass the current training horizon to train_loop ---
        train_loss = train_loop(model, optimizer, loss_fn, train_dataloader, current_train_horizon)
        train_loss_list.append(train_loss)

        # --- Pass the fixed validation horizon to validation_loop ---
        validation_loss = validation_loop(model, loss_fn, val_dataloader, validation_horizon)
        validation_loss_list.append(validation_loss)

        print(f"Training loss (h={current_train_horizon}): {train_loss:.6f}")
        print(f"Validation loss (h={validation_horizon}): {validation_loss:.6f}")
        print("-" * (50 + len(str(epoch+1)) + len(str(epochs)) + len(str(current_train_horizon)) + 10)) # Adjust separator

    print("Training finished.")
    return train_loss_list, validation_loss_list


def batchify_data(data, batch_size=config.BATCH_SIZE):
    """ Batches the training/validation data tuple. """
    batches = []
    X_enc_data, X_dec_in_data, Y_tgt_data = data

    if not (len(X_enc_data) == len(X_dec_in_data) == len(Y_tgt_data)):
        print("Error: Data components have different lengths. Cannot batch.")
        return []
    if len(X_enc_data) == 0:
        print("No data to batchify.")
        return []

    num_samples = len(X_enc_data)
    for i in range(0, num_samples, batch_size):
        X_enc_batch = X_enc_data[i:i + batch_size]
        X_dec_in_batch = X_dec_in_data[i:i + batch_size]
        Y_tgt_batch = Y_tgt_data[i:i + batch_size]
        if len(X_enc_batch) > 0:
            batches.append((X_enc_batch, X_dec_in_batch, Y_tgt_batch))

    print(f"Created {len(batches)} batches of size <= {batch_size}")
    return batches
