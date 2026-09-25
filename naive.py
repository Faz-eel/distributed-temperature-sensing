"""
A naive, non-ML baseline detector: flag an inflow if any point along the
pipe deviates from a smoothed "expected" baseline by more than a fixed
threshold.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def naive_detect(temperature_profile, threshold=0.5, smoothing_window=25):
    """
    Returns:
        detected: bool, whether the deviation was large enough to be
            confidently flagged as an inflow
        location_estimate: the position index of the largest deviation,
            or None if the method was not confident enough to flag
            anything at all.
    """

    # moving average of the profile = the "expected" temperature at each point
    smoothed_baseline = uniform_filter1d(temperature_profile, size=smoothing_window)

    # how far each point sits from its expected value
    deviation = np.abs(temperature_profile - smoothed_baseline)

    # flag an inflow only if the biggest deviation crosses the threshold
    if deviation.max() > threshold:
        # location estimate = index of that biggest deviation
        return True, int(np.argmax(deviation))
    else:
        return False, None


def evaluate_naive_detector(temperature_profiles, truths, positions):
    """
    Runs naive_detect() across every scenario in temperature_profiles and
    returns a summary: overall detect/no-detect accuracy, and location
    error only for the scenarios it actually flagged as containing an
    inflow.
    """
    correct_detection = 0
    location_errors = []

    # marks scenarios that truly have an inflow AND were flagged by the detector
    detected_mask = np.zeros(len(truths), dtype=bool)

    for i in range(len(truths)):
        detected, estimated_index = naive_detect(temperature_profiles[i])
        truth = truths.iloc[i]

        # count the yes/no call as correct if it matches the ground truth
        if detected == truth["has_inflow"]:
            correct_detection += 1

        # location error only counts when there was an inflow and it was flagged
        if truth["has_inflow"] and detected:
            detected_mask[i] = True

            # convert the estimated index to metres and compare with the true position
            location_errors.append(abs(truth["position_m"] - positions[estimated_index]))

    return {
        "accuracy": correct_detection / len(truths),
        "n_detected": len(location_errors),
        "location_errors": np.array(location_errors),
        "detected_mask": detected_mask,
    }
