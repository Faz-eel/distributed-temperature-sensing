"""
A simple machine learning approach to the same problem the naive detector
tackles: given a DTS temperature profile, (1) decide whether an inflow
event occurred, and if so, (2) estimate where along the pipe it happened.

Rather than feeding the raw 500-point profile directly into a model (which
would need many more training examples to learn from reliably), we first
extract a handful of descriptive features from each profile - summary
statistics that capture the shape of the curve. 

Two separate models are trained:
  - a classifier: does this profile contain an inflow event at all?
  - a regressor: if so, roughly where along the pipe does it start?
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


def extract_features(temps, smoothing_window=25):
    """
    Turns a raw 500-point temperature profile into a small set of
    descriptive features for the model to learn from.
    """
    baseline = uniform_filter1d(temps, size=smoothing_window)
    deviation = temps - baseline

    return {
        "max_abs_deviation": np.max(np.abs(deviation)),
        "mean_abs_deviation": np.mean(np.abs(deviation)),
        "std_deviation": np.std(deviation),
        "max_deviation": np.max(deviation),      # signed - captures warm events
        "min_deviation": np.min(deviation),      # signed - captures cold events
        "argmax_position_frac": np.argmax(np.abs(deviation)) / len(temps),
        "range_temp": np.max(temps) - np.min(temps),
    }


def build_feature_matrix(X, feature_names=None):
    feats = [extract_features(row) for row in X]
    if feature_names is None:
        feature_names = list(feats[0].keys())
    return np.array([[f[k] for k in feature_names] for f in feats]), feature_names


def deviation_curve(row, window=25):
    """The raw temperature profile minus its own smoothed baseline -
    used as input to the regressor, since it keeps the full spatial
    resolution a compressed feature vector would otherwise lose."""
    return row - uniform_filter1d(row, size=window)


def train_ml_classifier(X_train_raw, y_train):
    """Trains a random forest classifier on summary features to decide
    whether a profile contains an inflow event at all."""
    X_train, feature_names = build_feature_matrix(X_train_raw)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train["has_inflow"])
    return clf, feature_names


def train_ml_regressor(X_train_raw, y_train):
    """Trains a random forest regressor on the raw deviation curve to
    estimate where along the pipe an inflow occurs."""
    X_train_dev = np.array([deviation_curve(row) for row in X_train_raw])
    train_mask = y_train["has_inflow"].values

    reg = RandomForestRegressor(n_estimators=200, random_state=0)
    reg.fit(X_train_dev[train_mask], y_train.loc[train_mask, "position_m"])
    return reg


def evaluate_ml_classifier(clf, X_test_raw, y_test):
    X_test, _ = build_feature_matrix(X_test_raw)
    pred = clf.predict(X_test)
    return accuracy_score(y_test["has_inflow"], pred)


def evaluate_ml_regressor(reg, X_test_raw, y_test):
    X_test_dev = np.array([deviation_curve(row) for row in X_test_raw])
    test_mask = y_test["has_inflow"].values

    pred_position = reg.predict(X_test_dev[test_mask])
    true_position = y_test.loc[test_mask, "position_m"].values
    return np.abs(pred_position - true_position)