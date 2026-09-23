# DTS Inflow Detection

Simulates and detects hidden inflow events along a pipe using distributed
temperature sensing (DTS) data, and compares two detection approaches: a
naive threshold-based baseline and a machine-learning model.

## Background

A DTS cable run alongside a pipe reports a temperature reading at many
points along its length. If an inflow of water at a different temperature
mixes into the pipe at some hidden location, it produces a localized bump
(or dip) in the temperature profile at and downstream of that point, which
then decays back toward the ambient baseline over some distance.

The goal is, given only the temperature profile along the pipe:
1. **Detect** whether an inflow event occurred at all.
2. **Locate** where along the pipe it happened, if so.

## Files

- **`physics.py`** - Simulates the DTS temperature profile for a single
  scenario: a baseline curve that relaxes toward ambient ground temperature
  along the pipe, plus (optionally) a temperature bump from an inflow event
  that decays exponentially downstream, plus sensor noise.
- **`scenarios.py`** - Generates a batch of random scenarios (some with an
  inflow at a random position/temperature/rate, some without) using
  `physics.py`, and returns the temperature profiles alongside the hidden
  ground truth for each one.
- **`naive.py`** - A simple, non-ML baseline detector: smooths each profile
  with a moving-average filter to estimate the "expected" baseline, then
  flags an inflow if any point deviates from that baseline by more than a
  fixed threshold. The location estimate is the point of largest deviation.
- **`ml.py`** - A machine-learning approach to the same problem. Trains a
  `RandomForestClassifier` (has an inflow occurred?) and a
  `RandomForestRegressor` (where?) on features extracted from each profile,
  then compares their accuracy against the naive detector on the same test
  scenarios.

## Setup

```
pip install -r requirements.txt
```

## Running

```
python naive.py   # runs the naive detector over generated scenarios, prints accuracy
python ml.py       # trains the ML models, prints accuracy vs. the naive detector
python scenarios.py  # generates a small batch of scenarios and prints the ground truth
```
