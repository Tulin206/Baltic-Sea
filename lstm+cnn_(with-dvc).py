import os
import dvc.api
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Reshape, TimeDistributed, LSTM, Dense, Flatten, Dropout, MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from tensorflow.keras.callbacks import EarlyStopping

# Load raw data
def load_raw_data(filepath):
    data = np.load(filepath)
    return data["train"], data["test"], data["mask"]

# Save processed data
def save_data(train, test, mask, train_dir="train", test_dir="test"):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    np.save(f"{train_dir}/data.npy", train)
    np.save(f"{test_dir}/data.npy", test)
    np.save(f"{train_dir}/mask.npy", mask)
    np.save(f"{test_dir}/mask.npy", mask)

    '''
    # Track the saved files with DVC
    os.system(f"dvc add {train_dir}/data.npy")
    os.system(f"dvc add {test_dir}/data.npy")
    os.system(f"dvc add {train_dir}/mask.npy")
    os.system(f"dvc add {test_dir}/mask.npy")

    # Commit the changes to Git
    os.system("git add .")
    os.system("git commit -m 'Add processed data to DVC'")
    '''

    print("Data split and saved!")

# Load DVC-tracked data
def load_data():
    # Fetch paths from DVC
    train_path = dvc.api.get_url("train/data.npy")
    test_path = dvc.api.get_url("test/data.npy")
    mask_path = dvc.api.get_url("train/mask.npy")

    # Load data from the fetched paths
    train = np.load(train_path)
    test = np.load(test_path)
    mask = np.load(mask_path)
    return train, test, mask

# Define look-back and look-ahead window sizes, this is fixed for the task
look_back = int(7 * 24 / 6)  # 28 time steps (7 days) -> our input
look_ahead = 14 * 24 // 6  # 56 time steps (14 days) -> our prediction window
step_size = 24 // 6 # daily step size

def create_sample_pairs(subset, look_back: int, look_ahead: int, step_size: int):
    # Initialize lists to hold input-output pairs
    X, y = [], []

    # Loop over time series to create samples
    for i in range(look_back, subset.shape[0] - look_ahead, step_size):
        # Extract input sequence (look-back window) and output (look-ahead window)
        X_sample = subset[i - look_back: i, :, :, :]
        y_sample = subset[i + look_ahead, :, :, :]

        # Append to training data lists
        X.append(X_sample)
        y.append(y_sample)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

def split_data(X, y, split_ratio=0.8):
    # Define the split index for training and validation (80% training, 20% validation)
    split_index = int(split_ratio * len(X))

    # Split into training and validation sets
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    return X_train, X_val, y_train, y_val

# Function to replace NaN values with 0.0
def replace_nan_with_zero(X, y):
    X[np.isnan(X)] = 0.0
    y[np.isnan(y)] = 0.0

    # Print dataset shapes and check for NaNs/Infs
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("NaNs in X:", np.isnan(X).any())
    print("NaNs in y:", np.isnan(y).any())
    print("Infs in X:", np.isinf(X).any())
    print("Infs in y:", np.isinf(y).any())

    return X, y

# Model Definition
def build_model(time_steps, channels, height, width):
    # Define the model
    input_layer = Input(shape=(time_steps, channels, height, width))

    # Apply CNN to each time step
    cnn_layer = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(input_layer)
    cnn_layer = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(cnn_layer)
    cnn_layer = TimeDistributed(MaxPooling2D((2, 2)))(cnn_layer)  # Add pooling for feature reduction (2nd version)

    # Print the shape after the CNN layer
    print("Shape after CNN layers (cnn_layer) before flatten():", cnn_layer.shape)

    cnn_layer = TimeDistributed(Flatten())(cnn_layer)
    # Print the shape after the CNN layer
    print("Shape after CNN layers (cnn_layer) after flatten():", cnn_layer.shape)

    # Reshape for LSTM input
    reshaped_layer = Reshape((time_steps, -1))(cnn_layer)

    # Print the shape after the CNN layer
    print("Shape after CNN layers (cnn_layer):", reshaped_layer.shape)

    # Optional: Create a model and print the summary to verify shapes
    model = Model(inputs=input_layer, outputs=reshaped_layer)
    model.summary()

    # LSTM layers for temporal processing (2nd version of the model)
    lstm_layer = LSTM(128, return_sequences=True, dropout=0.2)(reshaped_layer)  # First LSTM layer
    lstm_layer = LSTM(64, return_sequences=False, dropout=0.2)(lstm_layer)  # Second LSTM layer

    # LSTM layer for temporal processing (1st version of the model)
    #lstm_layer = LSTM(64, return_sequences=False)(reshaped_layer)
    #print("Shape of LSTM layers:", lstm_layer.shape)

    # Separate Dense output layers for temperature and salinity predictions
    # Temperature prediction (channel 0)
    temperature_output = Dense(height * width, activation='linear')(lstm_layer)
    print("Shape of dense layers for temperature:", temperature_output.shape)
    temperature_output = Reshape((1, height, width), name='temperature_output')(temperature_output)  # 1 channel for temperature
    print("Shape of reshape layers for temperature:", temperature_output.shape)

    # Salinity prediction (channel 1)
    salinity_output = Dense(height * width, activation='linear')(lstm_layer)
    print("Shape of dense layers for salinity:", salinity_output.shape)
    salinity_output = Reshape((1, height, width), name='salinity_output')(salinity_output)  # 1 channel for salinity
    print("Shape of reshape layers for salinity:", salinity_output.shape)

    model = Model(inputs=input_layer, outputs=[temperature_output, salinity_output])
    return model

# Training
def train_model(model, X_train, y_train_temperature, y_train_salinity, X_val, y_val_temperature, y_val_salinity, batch_size=1, epochs=10):
    # Compile the model with two separate losses for temperature and salinity
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'temperature_output': 'mse', 'salinity_output': 'mse'},
        metrics={'temperature_output': 'mae', 'salinity_output': 'mae'}
    )

    # Check output names
    print("Model output names:", model.output_names)

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss',  # monitor validation loss
                                   patience=10,  # number of epochs with no improvement before stopping
                                   restore_best_weights=True,  # restore model weights from the best epoch
                                   verbose=1)  # print out details of early stopping
    # Train the model
    history = model.fit(
        X_train,
        [y_train_temperature, y_train_salinity],  # for multiple outputs
        epochs=epochs,  # the number of epochs to train
        batch_size=batch_size,  # batch size for training
        validation_data=(X_val, [y_val_temperature, y_val_salinity]),  # for multiple outputs
        verbose=2,  # Shows more detailed logs
        callbacks=[early_stopping],  # Add EarlyStopping callback
    )

    model.save("model_v2.keras")           # Save the trained model
    model.save("model_v2.h5")
    np.save("metrics_v2.npy", history.history)         # Save training metrics

    return history  # Return history object


# Main
def main():
    # Phase 1: Initial raw data processing (only done once)
    raw_data_path = "C:/Users/Tim/Desktop/ISRAT/RostockUniversity/data.npz"  # Actual path to data.npz
    # Load and preprocess data
    train, test, mask = load_raw_data(raw_data_path)
    # the shape is (t, c, w, h) -> (time, channel [temperature, salinity], width, height)
    # nan values are not of interest (land masses in this case)
    # each time step is 6 hours
    print(train.shape)
    print(train.shape[0])
    print(mask.shape)

    save_data(train, test, mask)

    # Phase 2: Use DVC-tracked data
    train, test, mask = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Mask shape: {mask.shape}")

    # Create sample pairs
    look_back, look_ahead, step_size = 28, 56, 4
    X_train, y_train = create_sample_pairs(train, look_back, look_ahead, step_size)
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    X_test, y_test = create_sample_pairs(test, look_back, look_ahead, step_size)

    # Replace NaN values with 0.0 for training and validation data
    X_train, y_train = replace_nan_with_zero(X_train, y_train)
    X_val, y_val = replace_nan_with_zero(X_val, y_val)

    # Replace Nan values with 0.0 for test data
    X_test, y_test = replace_nan_with_zero(X_test, y_test)

    # Separate outputs channel
    #y_train_temp, y_train_sal = y_train[:, 0], y_train[:, 1]
    #y_val_temp, y_val_sal = y_val[:, 0], y_val[:, 1]

    # Split y_train and y_val into separate temperature and salinity arrays
    y_train_temperature = y_train[:, 0, :, :]  # Select the temperature channel (index 0)
    y_train_salinity = y_train[:, 1, :, :]  # Select the salinity channel (index 1)
    print("y_train_temperature shape:", y_train_temperature.shape)  # Should be (num_samples, c, w, h)
    print("y_train_salinity shape:", y_train_salinity.shape)  # Should be (num_samples, c, w, h)

    y_val_temperature = y_val[:, 0, :, :]  # Select the temperature channel (index 0)
    y_val_salinity = y_val[:, 1, :, :]  # Select the salinity channel (index 1)
    print("y_val_temperature shape:", y_val_temperature.shape)  # Should be (val_samples, c, w, h)
    print("y_val_salinity shape:", y_val_salinity.shape)  # Should be (val_samples, c, w, h)

    # Build and train model
    #time_steps, channels, height, width = X_train.shape[1:]
    # Input dimensions
    time_steps = X_train.shape[1]  # 28
    channels = X_train.shape[2]  # 2 (temperature, salinity)
    height = X_train.shape[3]  # 111
    width = X_train.shape[4]  # 251

    model = build_model(time_steps, channels, height, width)

    # Train the model
    history = train_model(model, X_train, y_train_temperature, y_train_salinity, X_val, y_val_temperature, y_val_salinity)

    # Optionally, visualize history (if needed)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()