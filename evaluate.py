"""
One shared way to score every method on every scenario in a test set,
including the scenarios with no inflow at all.

Each method reports, for every scenario, whether it thinks there is an
inflow (detected) and where it thinks the inflow is (predicted_position_m).
This module compares those against the ground truth.
"""

import numpy as np


def evaluate_predictions(truths, detected, predicted_position_m, hit_tolerance_m=10.0):
    """
    truths: ground truth table for ALL scenarios in the test set
    detected: True/False per scenario - did the method flag an inflow?
        None if the method has no way of saying (location is then scored
        as if it had flagged every scenario)
    predicted_position_m: the method's position guess per scenario
        (ignored wherever detected is False)

    Returns a dictionary of:
        detection_accuracy: fraction of ALL scenarios called correctly
        false_alarm_rate: fraction of no-inflow scenarios wrongly flagged
        miss_rate: fraction of inflow scenarios not flagged
        location_errors: metres of error for scenarios that have an inflow
            AND were flagged
        hit_rate: fraction of ALL inflow scenarios that were flagged AND
            located within hit_tolerance_m - misses count against it
    """
    has_inflow = truths["has_inflow"].values.astype(bool)
    true_position_m = truths["position_m"].values.astype(float)

    result = {}

    if detected is None:
        # no detection ability, so location is scored on every inflow scenario
        detected = np.ones(len(truths), dtype=bool)
        result["detection_accuracy"] = None
        result["false_alarm_rate"] = None
        result["miss_rate"] = None
    else:
        result["detection_accuracy"] = (detected == has_inflow).mean()
        result["false_alarm_rate"] = detected[~has_inflow].mean()
        result["miss_rate"] = (~detected[has_inflow]).mean()

    # location only makes sense where there really is an inflow and it was flagged
    located = has_inflow & detected
    location_errors = np.abs(predicted_position_m[located] - true_position_m[located])

    result["location_errors"] = location_errors
    result["hit_rate"] = (location_errors <= hit_tolerance_m).sum() / has_inflow.sum()
    return result
