"""
Simulates the temperature profile a distributed temperature sensing (DTS)
cable would read along a pipe, when one hidden inflow point mixes water
of a different temperature into the main flow.
"""

import numpy as np


def mixed_temperature(t_main, q_main, t_inflow, q_inflow, c_main=4186.0, c_inflow=4186.0):
    """
    Energy-weighted mixing: the resulting temperature when two flows
    combine
 
    t_main, t_inflow: temperatures of the two flows (deg C)
    q_main, q_inflow: flow rates of the two flows (any consistent unit, e.g. L/s)
    c_main, c_inflow: specific heat capacities (J/(kg*K)); default is water's
    """

    numerator = (t_main * q_main * c_main) + (t_inflow * q_inflow * c_inflow)
    denominator = (q_main * c_main) + (q_inflow * c_inflow)

    return numerator / denominator


def simulate_dts_profile(
    pipe_length=1000,      # metres
    n_points=500,          # sensor resolution along the pipe
    t_main=15.0,           # main flow temperature, deg C
    q_main=10.0,           # main flow rate
    ground_temp=12.0,      # ambient/ground temperature the pipe relaxes toward
    inflow_position=None,  # chainage along pipe where inflow was introduced; None = no inflow (normal condition)
    inflow_temp=None,      # deg C
    inflow_rate=None,      # same units as q_main
    relax_length=150.0,    # metres over which the temperature bump fades back to ambient
    noise_std=0.15,        # sensor measurement noise, deg C
    seed=None,
):
    """
    Returns (positions, temperatures) - the DTS cable's reading along the
    pipe, in metres and degrees C.
    """
    # create a random number generator object for intoducing noise to the temperature data set
    rng = np.random.default_rng(seed)

    # list the chainages along the pipe 
    positions = np.linspace(0, pipe_length, n_points)
 
    # Baseline: main flow gradually settles toward ground temperature
    # over the length of the pipe, in the absence of any inflow
    # exp produces a curve from 1 in the initial positions towards 0 in the later position.
    # subtracting from 1 reverses it to start from 0 so the ground temp distribution increases toward the end 
    temps = t_main + (ground_temp - t_main) * (1 - np.exp(-positions / (pipe_length * 2)))
 
    if inflow_position is not None:

        # find the mixed temperature given the introduced inflow
        mixed = mixed_temperature(t_main, q_main, inflow_temp, inflow_rate)

        # find the normal temperature at that inflow point only
        temp_position = (t_main + (ground_temp - t_main) *
                         (1 - np.exp(-inflow_position / (pipe_length * 2))))
         
        # subtract it from the mixed temp to get the temperature bump
        bump = mixed - temp_position
 
        # Downstream of the inflow point, the temperature bump decays
        # exponentially back toward the baseline over relax_length metres.
        # Upstream of the inflow point, nothing changes (water hasn't
        # reached the inflow yet).
        downstream = positions >= inflow_position
        decay = np.exp(-(positions - inflow_position) / relax_length)
        temps = np.where(downstream, temps + bump * decay, temps)
 
    # Add sensor measurement noise
    temps = temps + rng.normal(0, noise_std, size=temps.shape)
 
    return positions, temps
 
 