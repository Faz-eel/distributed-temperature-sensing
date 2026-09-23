"""
A naive, non-ML baseline detector: flag an inflow if any point along the
pipe deviates from a smoothed "expected" baseline by more than a fixed
threshold. This mirrors how a simple real-world DTS alarm system often
works in practice (a fixed deviation threshold), and gives us something
concrete to compare a proper ML approach against.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def naive_detect(temps, threshold=0.5, smoothing_window=25):
    """
    Returns:
        detected: bool, whether an anomaly was flagged anywhere
        location_estimate: the position index of the largest deviation
            (only meaningful if detected is True)
    """
    baseline = uniform_filter1d(temps, size=smoothing_window)
    deviation = np.abs(temps - baseline)

    if deviation.max() > threshold:
        return True, int(np.argmax(deviation))
    else:
        return False, None


def evaluate_naive_detector(X, y, positions):
    """
    Runs naive_detect() across every scenario in X and returns a summary:
    overall detect/no-detect accuracy, and location error only for the
    scenarios it actually flagged as containing an inflow.
    """
    correct_detection = 0
    location_errors = []
    detected_mask = np.zeros(len(y), dtype=bool)

    for i in range(len(y)):
        detected, loc_idx = naive_detect(X[i])
        truth = y.iloc[i]

        if detected == truth["has_inflow"]:
            correct_detection += 1

        if truth["has_inflow"] and detected:
            detected_mask[i] = True
            location_errors.append(abs(truth["position_m"] - positions[loc_idx]))

    return {
        "accuracy": correct_detection / len(y),
        "n_detected": len(location_errors),
        "location_errors": np.array(location_errors),
        "detected_mask": detected_mask,
    }