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


def predict_naive(temperature_profiles, positions):
    """
    Runs naive_detect() on every scenario. Returns:
        detected: True/False per scenario
        predicted_position_m: position guess per scenario in metres
            (NaN wherever nothing was flagged)
    """
    detected = np.zeros(len(temperature_profiles), dtype=bool)
    predicted_position_m = np.full(len(temperature_profiles), np.nan)

    for i, profile in enumerate(temperature_profiles):
        is_flagged, estimated_index = naive_detect(profile)
        detected[i] = is_flagged

        # convert the estimated index to metres
        if is_flagged:
            predicted_position_m[i] = positions[estimated_index]

    return detected, predicted_position_m
