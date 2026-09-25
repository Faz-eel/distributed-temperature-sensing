"""
Runs the full distributed temperature sensing (DTS) project end to end:

  1. Preview the underlying physics (three example temperature profiles)
  2. Generate a training and test set of scenarios
  3. Evaluate the naive threshold detector
  4. Train and evaluate the random forest classifier + regressor
  5. Train and evaluate two neural network approaches (a regressor, and a
     combined softmax-over-position model), and compare all methods
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
from cnn import train_cnn_regressor, evaluate_cnn_regressor
from cnn_softmax import train_cnn_location_softmax, evaluate_cnn_location_softmax


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

    return ml_accuracy


def run_cnn(X_train_raw, y_train, X_test_raw, y_test, positions, naive_result, rf_accuracy):
    print("=" * 65)
    print("STEP 3: NEURAL NETWORK MODELS")
    print("=" * 65)

    print("Training regressor (single number output)...")
    reg_model = train_cnn_regressor(X_train_raw, y_train, pipe_length=PIPE_LENGTH)
    cnn_error = evaluate_cnn_regressor(reg_model, X_test_raw, y_test, pipe_length=PIPE_LENGTH)

    print("\nTraining softmax-over-position model "
          "(one model, answers detection and location together)...")
    softmax_model = train_cnn_location_softmax(X_train_raw, y_train, pipe_length=PIPE_LENGTH)
    softmax_detected, softmax_accuracy, softmax_errors, softmax_peak_probs = \
        evaluate_cnn_location_softmax(softmax_model, X_test_raw, y_test, positions)

    print("\n" + "-" * 65)
    print("DETECTION (inflow present or not)")
    print(f"  Naive threshold        : {100*naive_result['accuracy']:.1f}%")
    print(f"  Random forest          : {100*rf_accuracy:.1f}%")
    print(f"  Neural net (softmax, from its own peak probability) : "
          f"{100*softmax_accuracy:.1f}%")

    print("\nLOCATION ESTIMATE (metres of error)")
    print(f"  Naive threshold (only the {naive_result['n_detected']} scenarios it "
          f"was confident enough to flag) - mean: "
          f"{naive_result['location_errors'].mean():.1f} m   "
          f"median: {np.median(naive_result['location_errors']):.1f} m")
    print(f"  Neural net, regressor (forced to guess on all "
          f"{len(cnn_error)} scenarios) - mean: {cnn_error.mean():.1f} m   "
          f"median: {np.median(cnn_error):.1f} m")
    print(f"  Neural net, softmax-over-position (all "
          f"{len(softmax_errors)} scenarios with an inflow) - mean: "
          f"{softmax_errors.mean():.1f} m   median: {np.median(softmax_errors):.1f} m")
    print("=" * 65)


def main():
    preview_physics()

    print("Generating training and test data...\n")
    X_train_raw, y_train, positions = generate_scenarios(
        n_scenarios=3000, pipe_length=PIPE_LENGTH, seed=10)
    X_test_raw, y_test, _ = generate_scenarios(
        n_scenarios=500, pipe_length=PIPE_LENGTH, seed=99)

    naive_result = run_naive_baseline(X_test_raw, y_test, positions)
    rf_accuracy = run_random_forest(X_train_raw, y_train, X_test_raw, y_test, naive_result)
    run_cnn(X_train_raw, y_train, X_test_raw, y_test, positions, naive_result, rf_accuracy)


if __name__ == "__main__":
    main()