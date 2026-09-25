"""
A 1D convolutional neural network that predicts inflow location as a
single regressed number.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import uniform_filter1d


def build_deviation_curve(temperature_profile, smoothing_window=25):
    """Subtracts a smoothed baseline from a raw profile, leaving just the
    deviation (the 'bump') to learn from."""

    # moving average of the profile = what the temperature "should" look like
    smoothed_baseline = uniform_filter1d(temperature_profile, size=smoothing_window)

    # what's left over is the bump caused by an inflow (plus noise)
    return temperature_profile - smoothed_baseline


def build_deviation_curves(temperature_profiles, smoothing_window=25):
    """Applies build_deviation_curve() across a whole batch of profiles."""

    # one deviation curve per scenario, stacked into shape (n_scenarios, n_points)
    return np.array([build_deviation_curve(profile, smoothing_window)
                     for profile in temperature_profiles])


def build_cnn_regressor(n_points):
    """
    A single-layer 1D CNN for regression: input is the deviation curve
    (n_points long), output is a single number - the predicted inflow
    position in metres (normalised to 0-1 during training, rescaled
    afterwards).
    """
    model = keras.Sequential([
        # one sample = n_points readings with 1 channel (the deviation value)
        layers.Input(shape=(n_points, 1)),

        # 16 learned 15-point patterns slide along the curve, keeping length n_points
        layers.Conv1D(filters=16, kernel_size=15, activation="relu", padding="same"),

        # keep the strongest response in every block of 4 points (length / 4)
        layers.MaxPooling1D(pool_size=4),

        # unroll the (length, 16) grid into one long list of numbers
        layers.Flatten(),

        # learned mixing of all those numbers
        layers.Dense(64, activation="relu"),

        # randomly switch off 20% of units while training to limit overfitting
        layers.Dropout(0.2),

        # predicts position as a 0-1 fraction of the pipe length
        layers.Dense(1, activation="sigmoid"),
    ])

    # adam adjusts the weights; mse penalises the squared position error
    model.compile(optimizer="adam", loss="mse")
    return model


def train_cnn_regressor(train_temperature_profiles, train_truths, pipe_length,
                        epochs=30, batch_size=32):
    """
    Trains the regressor on scenarios that have an inflow. Returns the
    trained model (with early stopping's best weights restored).
    """
    # only scenarios with an inflow have a position to learn
    has_inflow_mask = train_truths["has_inflow"].values

    # deviation curves for the inflow scenarios only
    train_deviation_curves = build_deviation_curves(train_temperature_profiles)[has_inflow_mask]

    # true positions as a 0-1 fraction of the pipe (matches the sigmoid output)
    train_position_fractions = train_truths.loc[has_inflow_mask, "position_m"].values / pipe_length

    # add a channel axis: (n_scenarios, n_points) -> (n_scenarios, n_points, 1)
    train_deviation_curves = train_deviation_curves[..., np.newaxis]

    model = build_cnn_regressor(n_points=train_deviation_curves.shape[1])

    # stop when validation loss hasn't improved for 5 epochs, keep the best weights
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        train_deviation_curves, train_position_fractions,
        validation_split=0.15,  # hold out 15% to check generalisation
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model


def predict_cnn_regressor(model, test_temperature_profiles, pipe_length):
    """Returns the regressor's position guess (metres) for every scenario.
    It has no way of saying "no inflow" - it always outputs a position."""
    # same preprocessing as training: deviation curves + channel axis
    test_deviation_curves = build_deviation_curves(test_temperature_profiles)[..., np.newaxis]

    # model predicts a 0-1 fraction, scale back up to metres
    predicted_fractions = model.predict(test_deviation_curves, verbose=0).flatten()
    return predicted_fractions * pipe_length
