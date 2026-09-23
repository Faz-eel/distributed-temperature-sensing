"""
A 1D convolutional neural network for locating an inflow event along a
pipe from its DTS temperature profile.

This exists to fix a specific weakness found in the random forest
approach: when fed the raw 500-point profile directly, the random forest
performed WORSE than a naive threshold detector at locating events,
because each of its 500 input columns corresponds to a fixed physical
position, and a tree-based model has no way to recognise "a bump" as a
pattern independent of exactly where in the 500 columns it happens to
fall. It has to relearn the same rule separately for every possible
position, which it cannot do well from a few hundred training examples.

A 1D convolution fixes this directly. A small kernel (say, 15 points wide)
slides along the whole profile, computing the same learned pattern-match
at every position using the SAME weights each time. This gives the model
translation invariance for free: it only has to learn "this is what an
inflow bump looks like" once, and it can then recognise that shape
wherever it occurs along the pipe.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import uniform_filter1d


def build_deviation_curves(X_raw, smoothing_window=25):
    """Same preprocessing as before: subtract a smoothed baseline from
    each raw profile, leaving just the deviation (the 'bump') to learn from."""
    baselines = np.array([uniform_filter1d(row, size=smoothing_window) for row in X_raw])
    return X_raw - baselines


def build_cnn_model(n_points):
    """
    A small 1D CNN for regression: input is the deviation curve (n_points
    long), output is a single number - the predicted inflow position in
    metres (normalised to 0-1 during training, rescaled afterwards).
    """
    model = keras.Sequential([
        layers.Input(shape=(n_points, 1)),
        layers.Conv1D(filters=16, kernel_size=15, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=4),
        layers.Conv1D(filters=32, kernel_size=9, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=4),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),  # predicts position as a 0-1 fraction
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_cnn(X_train_raw, y_train, pipe_length, epochs=60, batch_size=32):
    """
    Trains the 1D CNN on scenarios that have an inflow. Returns the
    trained model (with early stopping's best weights restored).
    """
    train_mask = y_train["has_inflow"].values
    X_train_dev = build_deviation_curves(X_train_raw)[train_mask]
    y_train_pos = y_train.loc[train_mask, "position_m"].values / pipe_length
    X_train_dev = X_train_dev[..., np.newaxis]

    model = build_cnn_model(n_points=X_train_dev.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train_dev, y_train_pos,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=2,
    )
    return model


def evaluate_cnn(model, X_test_raw, y_test, pipe_length):
    """Returns the CNN's location error (metres) for every scenario in
    the test set that actually has an inflow."""
    test_mask = y_test["has_inflow"].values
    X_test_dev = build_deviation_curves(X_test_raw)[test_mask]
    X_test_dev = X_test_dev[..., np.newaxis]
    y_test_pos = y_test.loc[test_mask, "position_m"].values / pipe_length

    pred_frac = model.predict(X_test_dev, verbose=0).flatten()
    pred_position = pred_frac * pipe_length
    true_position = y_test_pos * pipe_length

    return np.abs(pred_position - true_position)