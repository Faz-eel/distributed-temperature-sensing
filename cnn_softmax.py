"""
A 1D convolutional neural network that answers BOTH questions this
project cares about - is there an inflow, and if so, where - using a
single softmax output over every position along the pipe, rather than
two separate models.
The key structural difference from cnn.py: there is no
Flatten() layer feeding into a Dense stack here. The regressor's
Flatten() step destroys the direct correspondence between "a feature was
found here" and "here is a physical position on the pipe" - everything
after Flatten() has to relearn that mapping from raw, unordered numbers.
This model instead keeps one output permanently tied to each physical
chainage, all the way from input to output, via a second Conv1D layer
with kernel_size=1 that collapses the 16 filter channels down to one
score per position, followed directly by softmax. Position is never
flattened away.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from cnn import build_deviation_curves


def build_cnn_location_softmax(n_points):
    """
    Kept deliberately shallow (conv layer collapsed straight into the
    output) so the spatial position of each unit is preserved as
    directly as possible.
    """
    model = keras.Sequential([
        # one sample = n_points readings with 1 channel (the deviation value)
        layers.Input(shape=(n_points, 1)),

        # 16 learned 15-point patterns slide along the curve, keeping length n_points
        layers.Conv1D(filters=16, kernel_size=15, activation="relu", padding="same"),

        # Collapses the 16 filter channels down to a single channel, one
        # score per position - this becomes the raw "score" for each
        # chainage before softmax turns those scores into a probability
        # distribution over position.
        layers.Conv1D(filters=1, kernel_size=1, activation="linear", padding="same"),

        # drop the channel axis: (n_points, 1) -> (n_points,)
        layers.Reshape((n_points,)),

        # turn the scores into probabilities that sum to 1
        layers.Softmax(),
    ])

    # crossentropy penalises putting low probability on the true position
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def train_cnn_location_softmax(train_temperature_profiles, train_truths, pipe_length=1000,
                                epochs=30, batch_size=32):
    """
    The target for each training example is a one-hot vector: 1.0 at the
    true position's index, 0.0 everywhere else - the standard way to
    train a softmax classifier when there is one correct category per
    example.
    """
    n_points = train_temperature_profiles.shape[1]

    # only scenarios with an inflow have a position to learn
    has_inflow_mask = train_truths["has_inflow"].values

    # deviation curves for the inflow scenarios, with a channel axis added
    train_deviation_curves = build_deviation_curves(train_temperature_profiles)[has_inflow_mask][..., np.newaxis]

    # the chainage (in metres) of each of the n_points sensor positions
    positions_m = np.linspace(0, pipe_length, n_points)
    true_positions_m = train_truths.loc[has_inflow_mask, "position_m"].values

    # one-hot target per scenario: 1.0 at the sensor point closest to the true position
    train_targets = np.zeros((len(true_positions_m), n_points))
    for i, true_position_m in enumerate(true_positions_m):
        closest_index = np.argmin(np.abs(positions_m - true_position_m))
        train_targets[i, closest_index] = 1.0

    model = build_cnn_location_softmax(n_points=n_points)

    # stop when validation loss hasn't improved for 5 epochs, keep the best weights
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        train_deviation_curves, train_targets,
        validation_split=0.15,  # hold out 15% to check generalisation
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model


def predict_cnn_location_softmax(model, test_temperature_profiles, positions,
                                 detection_threshold=0.1):
    """
    Runs every scenario in the test set - both with and without an
    inflow - through the model, and answers both questions from the
    same set of predicted probabilities.

    Returns:
        detected: boolean array, one per scenario - True if the model's
            peak probability exceeds detection_threshold
        predicted_position_m: position guess per scenario, in metres
        peak_probabilities: the raw peak probability for every scenario,
            for reference
    """
    # deviation curves for ALL test scenarios (with and without an inflow)
    test_deviation_curves = build_deviation_curves(test_temperature_profiles)[..., np.newaxis]

    # one probability per sensor point, per scenario
    predicted_probabilities = model.predict(test_deviation_curves, verbose=0)  # shape: (n_scenarios, n_points)

    # the most likely position, and how confident the model is in it
    predicted_index = np.argmax(predicted_probabilities, axis=1)
    peak_probabilities = predicted_probabilities[np.arange(len(predicted_probabilities)), predicted_index]

    # detection: call it an inflow if the model is confident enough in one position
    detected = peak_probabilities > detection_threshold

    # convert the winning index to metres
    predicted_position_m = positions[predicted_index]

    return detected, predicted_position_m, peak_probabilities