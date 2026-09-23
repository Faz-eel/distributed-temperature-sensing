"""
Generates many random inflow scenarios, each with a
randomly chosen hidden inflow location, temperature, and flow rate -
or no inflow at all, for "normal" examples.
"""

import numpy as np
import pandas as pd

from physics import simulate_dts_profile


def generate_scenarios(n_scenarios=500, pipe_length=1000, n_points=500,
                        edge_margin_frac=0.1, seed=0):
    """
    Returns:
        temperature_profiles: array of shape (n_scenarios, n_points) - the DTS temperature
           profile for each scenario
        truths: DataFrame with the hidden ground truth for each scenario-
           whether there was an inflow, and if so, where and how large
        chainages: The array of all chainages along the pipe length (equal for all scenarios)
    """
    # random number generator object
    rng = np.random.default_rng(seed)

    # edge_margin_frac: fraction of pipe_length to keep clear at each end
    # when placing an inflow, so inflow events aren't generated right at
    # the very start or end of the pipe (which would be physically hard
    # to distinguish from the ordinary baseline behaviour there)
    margin = pipe_length * edge_margin_frac
    position_low, position_high = margin, pipe_length - margin

    # initialize a numpy array for storing temperature profiles for each scenario
    temperature_profiles = np.zeros((n_scenarios, n_points))

    # keep record of the truth for each induced scenario, to compare with ML predictions
    records = []

    for i in range(n_scenarios):

        # 70% of scenarios have an inflow event
        has_inflow = rng.random() < 0.7  

        if has_inflow:

            # get a random inflow point between the specified ends 
            position = rng.uniform(position_low, position_high)

            # generate a random temperature 
            temp = rng.uniform(4.0, 30.0)

            # generate a random inflow strength              
            rate = rng.uniform(0.3, 3.0)               

        else:

            # conditions if no inflow
            position = temp = rate = None

        # simulate the dts profile given the generated conditions
        chainages, temps = simulate_dts_profile(
            pipe_length=pipe_length,
            n_points=n_points,
            inflow_position=position,
            inflow_temp=temp,
            inflow_rate=rate,
            seed=rng.integers(0, 1_000_000),
        )

        # append the profiles list with the current scenario's profile
        temperature_profiles[i] = temps

        # append the records
        records.append({
            "scenario": i,
            "has_inflow": has_inflow,
            "position_m": position,
            "inflow_temp_C": temp,
            "inflow_rate": rate,
        })

    # convert the records list to a pandas list after generating all scenarios
    truths = pd.DataFrame(records)


    return temperature_profiles, truths, chainages  


if __name__ == "__main__":
    temperature_profiles, truths, chainages = generate_scenarios(n_scenarios=10)
    print(truths)
    print(f"\nX shape: {temperature_profiles.shape}  (scenarios x sensor points along pipe)")