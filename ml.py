"""
A simple machine learning approach to the same problem the naive detector
tackles: given a DTS temperature profile, (1) decide whether an inflow
event occurred, and if so, (2) estimate where along the pipe it happened.

Rather than feeding the raw 500-point profile directly into a model (which
would need many more training examples to learn from reliably), we first
extract a handful of descriptive features from each profile - summary
statistics that capture the shape of the curve. This is standard practice
for scikit-learn style models, as opposed to deep learning approaches
which learn directly from raw grid/image data.

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


if __name__ == "__main__":
    from scenarios import generate_scenarios
    from naive import naive_detect

    print("Generating training and test data...")
    X_train_raw, y_train, positions = generate_scenarios(n_scenarios=800, seed=10)
    X_test_raw, y_test, _ = generate_scenarios(n_scenarios=200, seed=99)

    X_train, feature_names = build_feature_matrix(X_train_raw)
    X_test, _ = build_feature_matrix(X_test_raw)

    print(f"Features used: {feature_names}\n")

    # --- Classifier: is there an inflow at all? ---
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train["has_inflow"])
    pred_has_inflow = clf.predict(X_test)

    ml_accuracy = accuracy_score(y_test["has_inflow"], pred_has_inflow)

    # --- Regressor: where does the inflow occur? ---
    # Unlike the classifier above, this uses the raw deviation curve (500
    # points) rather than the compressed summary features. Location is a
    # spatial question, and compressing the profile down to a handful of
    # statistics first throws away exactly the fine-grained positional
    # detail needed to answer it precisely - the same reason Lansey's CNN
    # works directly on the full residual grid rather than on summary
    # statistics of it. Here we get that same benefit cheaply, without a
    # CNN, by simply giving the regressor the full deviation curve.
    from scipy.ndimage import uniform_filter1d as _smooth

    def deviation_curve(row, window=25):
        return row - _smooth(row, size=window)

    X_train_dev = np.array([deviation_curve(row) for row in X_train_raw])
    X_test_dev = np.array([deviation_curve(row) for row in X_test_raw])

    train_mask = y_train["has_inflow"].values
    test_mask = y_test["has_inflow"].values

    reg = RandomForestRegressor(n_estimators=200, random_state=0)
    reg.fit(X_train_dev[train_mask], y_train.loc[train_mask, "position_m"])
    pred_position = reg.predict(X_test_dev[test_mask])
    true_position = y_test.loc[test_mask, "position_m"].values

    ml_location_error = np.abs(pred_position - true_position)

    # --- Compare against the naive threshold detector on the same test set ---
    naive_correct = 0
    naive_location_errors = []
    for i in range(len(y_test)):
        detected, loc_idx = naive_detect(X_test_raw[i])
        truth = y_test.iloc[i]
        if detected == truth["has_inflow"]:
            naive_correct += 1
        if truth["has_inflow"] and detected:
            naive_location_errors.append(abs(truth["position_m"] - positions[loc_idx]))

    print("=" * 55)
    print("DETECTION (inflow present or not)")
    print(f"  Naive threshold accuracy : "
          f"{100*naive_correct/len(y_test):.1f}%")
    print(f"  ML classifier accuracy   : {100*ml_accuracy:.1f}%")
    print()
    print("LOCATION ESTIMATE (metres of error, when an inflow is present)")
    print(f"  Naive threshold - mean   : {np.mean(naive_location_errors):.1f} m"
          f"   median: {np.median(naive_location_errors):.1f} m")
    print(f"  ML regressor    - mean   : {np.mean(ml_location_error):.1f} m"
          f"   median: {np.median(ml_location_error):.1f} m")
    print("=" * 55)

    # Feature importance - which summary statistics mattered most?
    importances = sorted(zip(feature_names, clf.feature_importances_),
                          key=lambda x: -x[1])
    print("\nWhich features mattered most for detection:")
    for name, imp in importances:
        print(f"  {name:25s} {imp:.3f}")