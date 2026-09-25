"""
Runs the full distributed temperature sensing (DTS) project end to end:

  1. Preview the underlying physics (three example temperature profiles)
  2. Generate a training and test set of scenarios
  3. Run the naive threshold detector
  4. Train and run the random forest classifier + regressor
  5. Train and run two neural network approaches (a regressor, and a
     combined softmax-over-position model)
  6. Score every method on every test scenario, with and without an
     inflow, and print one side-by-side comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import simulate_dts_profile
from scenarios import generate_scenarios
from evaluate import evaluate_predictions
from naive import predict_naive
from ml import train_ml_classifier, train_ml_regressor, predict_ml
from cnn import train_cnn_regressor, predict_cnn_regressor
from cnn_softmax import train_cnn_location_softmax, predict_cnn_location_softmax


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


def run_naive(test_temperature_profiles, test_truths, positions):
    print("Running naive threshold detector...")
    detected, predicted_position_m = predict_naive(test_temperature_profiles, positions)
    return evaluate_predictions(test_truths, detected, predicted_position_m)


def run_random_forest(train_temperature_profiles, train_truths,
                      test_temperature_profiles, test_truths):
    print("Training random forest classifier + regressor...")
    classifier, feature_names = train_ml_classifier(train_temperature_profiles, train_truths)
    regressor = train_ml_regressor(train_temperature_profiles, train_truths)

    detected, predicted_position_m = predict_ml(classifier, regressor, test_temperature_profiles)

    # which summary statistics the classifier relied on most
    importances = sorted(zip(feature_names, classifier.feature_importances_),
                         key=lambda pair: -pair[1])
    print("  Which features mattered most for detection:")
    for name, importance in importances:
        print(f"    {name:25s} {importance:.3f}")

    return evaluate_predictions(test_truths, detected, predicted_position_m)


def run_cnn_regressor(train_temperature_profiles, train_truths,
                      test_temperature_profiles, test_truths):
    print("Training CNN regressor (single number output)...")
    model = train_cnn_regressor(train_temperature_profiles, train_truths, pipe_length=PIPE_LENGTH)
    predicted_position_m = predict_cnn_regressor(model, test_temperature_profiles,
                                                 pipe_length=PIPE_LENGTH)

    # this model cannot say "no inflow", so detected is None
    return evaluate_predictions(test_truths, None, predicted_position_m)


def run_cnn_softmax(train_temperature_profiles, train_truths,
                    test_temperature_profiles, test_truths, positions):
    print("Training CNN softmax-over-position model "
          "(one model, answers detection and location together)...")
    model = train_cnn_location_softmax(train_temperature_profiles, train_truths,
                                       pipe_length=PIPE_LENGTH)
    detected, predicted_position_m, _ = predict_cnn_location_softmax(
        model, test_temperature_profiles, positions)

    return evaluate_predictions(test_truths, detected, predicted_position_m)


def format_percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def print_comparison(results, test_truths):
    n_scenarios = len(test_truths)
    n_inflow = int(test_truths["has_inflow"].sum())

    print("\n" + "=" * 96)
    print(f"ALL {n_scenarios} TEST SCENARIOS: {n_inflow} with an inflow, "
          f"{n_scenarios - n_inflow} without")
    print("=" * 96)
    print(f"{'method':32s}{'detection':>11s}{'false':>9s}{'missed':>9s}"
          f"{'median':>10s}{'mean':>9s}{'located':>10s}")
    print(f"{'':32s}{'accuracy':>11s}{'alarms':>9s}{'inflows':>9s}"
          f"{'error m':>10s}{'error m':>9s}{'<10 m':>10s}")
    print("-" * 96)

    for name, result in results.items():
        errors = result["location_errors"]
        print(f"{name:32s}"
              f"{format_percent(result['detection_accuracy']):>11s}"
              f"{format_percent(result['false_alarm_rate']):>9s}"
              f"{format_percent(result['miss_rate']):>9s}"
              f"{np.median(errors):>10.1f}"
              f"{errors.mean():>9.1f}"
              f"{format_percent(result['hit_rate']):>10s}")

    print("=" * 96)
    print("detection accuracy: right yes/no call, over all scenarios")
    print("false alarms: no-inflow scenarios wrongly flagged   "
          "missed inflows: inflow scenarios not flagged")
    print("location error: only scenarios with an inflow that the method flagged")
    print("located <10 m: share of ALL inflow scenarios flagged AND placed within 10 m "
          "(misses count against it)")
    print("CNN regressor cannot say 'no inflow', so it is scored as if it flagged everything")


def main():
    preview_physics()

    print("Generating training and test data...\n")
    train_temperature_profiles, train_truths, positions = generate_scenarios(
        n_scenarios=3000, pipe_length=PIPE_LENGTH, seed=10)
    test_temperature_profiles, test_truths, _ = generate_scenarios(
        n_scenarios=500, pipe_length=PIPE_LENGTH, seed=99)

    results = {}
    results["Naive threshold"] = run_naive(
        test_temperature_profiles, test_truths, positions)
    results["Random forest"] = run_random_forest(
        train_temperature_profiles, train_truths, test_temperature_profiles, test_truths)
    results["CNN regressor"] = run_cnn_regressor(
        train_temperature_profiles, train_truths, test_temperature_profiles, test_truths)
    results["CNN softmax-over-position"] = run_cnn_softmax(
        train_temperature_profiles, train_truths, test_temperature_profiles, test_truths,
        positions)

    print_comparison(results, test_truths)


if __name__ == "__main__":
    main()
