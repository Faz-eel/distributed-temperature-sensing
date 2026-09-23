"""
Runs the full distributed temperature sensing (DTS) project end to end:

  1. Preview the underlying physics (three example temperature profiles)
  2. Generate a training and test set of scenarios
  3. Evaluate the naive threshold detector
  4. Train and evaluate the random forest classifier + regressor
  5. Train and evaluate the 1D CNN, and compare it fairly against naive

Each step's real logic lives in its own module (dts_physics.py,
scenarios.py, naive_detector.py, ml_detector.py, cnn_detector.py), which
are now pure collections of functions with no executable code of their
own. This file is the only place anything actually runs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import simulate_dts_profile
from scenarios import generate_scenarios
from naive import evaluate_naive_detector
from ml import (
    train_ml_classifier, train_ml_regressor,
    evaluate_ml_classifier, evaluate_ml_regressor,
)
from cnn import train_cnn, evaluate_cnn


PIPE_LENGTH = 1000


def preview_physics():
    """Plots three example scenarios to sanity-check the underlying
    physics before generating a full dataset from it."""
    fig, ax = plt.subplots(figsize=(10, 5))

    pos, temp_normal = simulate_dts_profile(seed=1)
    ax.plot(pos, temp_normal, label="No inflow (normal)", alpha=0.7)

    pos, temp_cold = simulate_dts_profile(
        inflow_position=400, inflow_temp=6.0, inflow_rate=2.0, seed=1)
    ax.plot(pos, temp_cold, label="Cold inflow at 400m (groundwater)", alpha=0.7)

    pos, temp_warm = simulate_dts_profile(
        inflow_position=650, inflow_temp=28.0, inflow_rate=1.0, seed=1)
    ax.plot(pos, temp_warm, label="Warm inflow at 650m (e.g. discharge)", alpha=0.7)

    ax.set_xlabel("Position along pipe (m)")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("Simulated DTS temperature profiles")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dts_profiles_preview.png", dpi=120)
    print("Saved dts_profiles_preview.png\n")


def run_naive_baseline(X_test_raw, y_test, positions):
    print("=" * 65)
    print("STEP 1: NAIVE THRESHOLD DETECTOR")
    print("=" * 65)
    result = evaluate_naive_detector(X_test_raw, y_test, positions)
    print(f"Detect/no-detect accuracy : {100*result['accuracy']:.1f}%")
    print(f"Location error (n={result['n_detected']}) - "
          f"mean: {result['location_errors'].mean():.1f} m   "
          f"median: {np.median(result['location_errors']):.1f} m\n")
    return result


def run_random_forest(X_train_raw, y_train, X_test_raw, y_test, naive_result):
    print("=" * 65)
    print("STEP 2: RANDOM FOREST (summary features + raw deviation curve)")
    print("=" * 65)

    clf, feature_names = train_ml_classifier(X_train_raw, y_train)
    ml_accuracy = evaluate_ml_classifier(clf, X_test_raw, y_test)

    reg = train_ml_regressor(X_train_raw, y_train)
    ml_location_error = evaluate_ml_regressor(reg, X_test_raw, y_test)

    print(f"Detection accuracy - naive: {100*naive_result['accuracy']:.1f}%   "
          f"random forest: {100*ml_accuracy:.1f}%")
    print(f"Location error - naive median: "
          f"{np.median(naive_result['location_errors']):.1f} m   "
          f"random forest median: {np.median(ml_location_error):.1f} m")

    importances = sorted(zip(feature_names, clf.feature_importances_),
                          key=lambda x: -x[1])
    print("\nWhich features mattered most for detection:")
    for name, imp in importances:
        print(f"  {name:25s} {imp:.3f}")
    print()


def run_cnn(X_train_raw, y_train, X_test_raw, y_test, positions, naive_result):
    print("=" * 65)
    print("STEP 3: 1D CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 65)

    model = train_cnn(X_train_raw, y_train, pipe_length=PIPE_LENGTH)
    cnn_error_all = evaluate_cnn(model, X_test_raw, y_test, pipe_length=PIPE_LENGTH)

    # Fair comparison: only look at the same scenarios the naive detector
    # actually flagged, since it is never penalised for the ones it skips
    test_mask = y_test["has_inflow"].values
    naive_detected_mask = naive_result["detected_mask"][test_mask]
    cnn_error_same_subset = cnn_error_all[naive_detected_mask]

    print(f"\nAll {len(cnn_error_all)} inflow scenarios in the test set:")
    print(f"  1D CNN (forced to guess on every scenario) - "
          f"mean: {cnn_error_all.mean():.1f} m   "
          f"median: {np.median(cnn_error_all):.1f} m")

    print(f"\nOnly the {naive_detected_mask.sum()} scenarios the naive "
          f"detector actually flagged (fair comparison):")
    print(f"  Naive threshold - mean: {naive_result['location_errors'].mean():.1f} m"
          f"   median: {np.median(naive_result['location_errors']):.1f} m")
    print(f"  1D CNN          - mean: {cnn_error_same_subset.mean():.1f} m"
          f"   median: {np.median(cnn_error_same_subset):.1f} m")
    print("=" * 65)


def main():
    preview_physics()

    print("Generating training and test data...\n")
    X_train_raw, y_train, positions = generate_scenarios(
        n_scenarios=3000, pipe_length=PIPE_LENGTH, seed=10)
    X_test_raw, y_test, _ = generate_scenarios(
        n_scenarios=500, pipe_length=PIPE_LENGTH, seed=99)

    naive_result = run_naive_baseline(X_test_raw, y_test, positions)
    run_random_forest(X_train_raw, y_train, X_test_raw, y_test, naive_result)
    run_cnn(X_train_raw, y_train, X_test_raw, y_test, positions, naive_result)


if __name__ == "__main__":
    main()