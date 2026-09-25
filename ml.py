"""
A simple machine learning approach to the same problem the naive detector
tackles: given a DTS temperature profile, (1) decide whether an inflow
event occurred, and if so, (2) estimate where along the pipe it happened.

Rather than feeding the raw 500-point profile directly into a model (which
would need many more training examples to learn from reliably), we first
extract a handful of descriptive features from each profile - summary
statistics that capture the shape of the curve. This is standard practice
for scikit-learn style models, as opposed to deep learning approaches
(like Lansey's CNN) which learn directly from raw grid/image data.

Two separate models are trained:
  - a classifier: does this profile contain an inflow event at all?
  - a regressor: if so, roughly where along the pipe does it start?
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


def extract_features(temperature_profile, smoothing_window=25):
    """
    Turns a raw 500-point temperature profile into a small set of
    descriptive features for the model to learn from.
    """
    # moving average = expected temperature; what's left over is the deviation
    smoothed_baseline = uniform_filter1d(temperature_profile, size=smoothing_window)
    deviation = temperature_profile - smoothed_baseline

    return {
        "max_abs_deviation": np.max(np.abs(deviation)),
        "mean_abs_deviation": np.mean(np.abs(deviation)),
        "std_deviation": np.std(deviation),
        "max_deviation": np.max(deviation),      # signed - captures warm events
        "min_deviation": np.min(deviation),      # signed - captures cold events
        "argmax_position_frac": np.argmax(np.abs(deviation)) / len(temperature_profile),
        "range_temp": np.max(temperature_profile) - np.min(temperature_profile),
    }


def build_feature_matrix(temperature_profiles, feature_names=None):
    # one dictionary of features per scenario
    features_per_profile = [extract_features(profile) for profile in temperature_profiles]

    # fix the column order from the first profile if none was given
    if feature_names is None:
        feature_names = list(features_per_profile[0].keys())

    # stack into a table: one row per scenario, one column per feature
    feature_matrix = np.array([[features[name] for name in feature_names]
                               for features in features_per_profile])
    return feature_matrix, feature_names


def deviation_curve(temperature_profile, smoothing_window=25):
    """The raw temperature profile minus its own smoothed baseline -
    used as input to the regressor, since it keeps the full spatial
    resolution a compressed feature vector would otherwise lose."""
    return temperature_profile - uniform_filter1d(temperature_profile, size=smoothing_window)


def train_ml_classifier(train_temperature_profiles, train_truths):
    """Trains a random forest classifier on summary features to decide
    whether a profile contains an inflow event at all."""
    train_features, feature_names = build_feature_matrix(train_temperature_profiles)

    # 200 decision trees vote on inflow / no inflow
    classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(train_features, train_truths["has_inflow"])
    return classifier, feature_names


def train_ml_regressor(train_temperature_profiles, train_truths):
    """Trains a random forest regressor on the raw deviation curve to
    estimate where along the pipe an inflow occurs."""
    train_deviation_curves = np.array([deviation_curve(profile)
                                       for profile in train_temperature_profiles])

    # only scenarios with an inflow have a position to learn
    has_inflow_mask = train_truths["has_inflow"].values

    # 200 decision trees, each predicts a position in metres; their average is the answer
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(train_deviation_curves[has_inflow_mask],
                  train_truths.loc[has_inflow_mask, "position_m"])
    return regressor


def evaluate_ml_classifier(classifier, test_temperature_profiles, test_truths):
    test_features, _ = build_feature_matrix(test_temperature_profiles)
    predicted_has_inflow = classifier.predict(test_features)

    # fraction of yes/no calls that match the ground truth
    return accuracy_score(test_truths["has_inflow"], predicted_has_inflow)


def evaluate_ml_regressor(regressor, test_temperature_profiles, test_truths):
    test_deviation_curves = np.array([deviation_curve(profile)
                                      for profile in test_temperature_profiles])

    # only score scenarios that really have an inflow
    has_inflow_mask = test_truths["has_inflow"].values

    predicted_position_m = regressor.predict(test_deviation_curves[has_inflow_mask])
    true_position_m = test_truths.loc[has_inflow_mask, "position_m"].values

    # absolute error in metres for each scenario
    return np.abs(predicted_position_m - true_position_m)
