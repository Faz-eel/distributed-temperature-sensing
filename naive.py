"""
A naive, non-ML baseline detector: flag an inflow if any point along the
pipe deviates from a smoothed "expected" baseline by more than a fixed
threshold. 
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
    # for each chainage in temps, establish a baseline value to suppress noise 
    baseline = uniform_filter1d(temps, size=smoothing_window)

    # check the deviation from the actual measured value for each smoothed point 
    deviation = np.abs(temps - baseline)

    # large deviation values signify anomalies
    if deviation.max() > threshold:
        return True, int(np.argmax(deviation))
    else:
        return False, None


if __name__ == "__main__":
    from scenarios import generate_scenarios

    X, y, positions = generate_scenarios(n_scenarios=200, seed=1)

    correct_detection = 0
    location_errors = []

    for i in range(len(y)):
        detected, loc_idx = naive_detect(X[i])
        truth = y.iloc[i]

        if detected == truth["has_inflow"]:
            correct_detection += 1

        if truth["has_inflow"] and detected:
            true_pos = truth["position_m"]
            guessed_pos = positions[loc_idx]
            location_errors.append(abs(true_pos - guessed_pos))

    print(f"Naive threshold detector:")
    print(f"  Correct detect/no-detect calls: {correct_detection}/{len(y)} "
          f"({100*correct_detection/len(y):.0f}%)")
    print(f"  Mean location error (when detected): "
          f"{np.mean(location_errors):.1f} m")
    print(f"  Median location error: {np.median(location_errors):.1f} m")