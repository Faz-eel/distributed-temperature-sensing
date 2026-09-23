# DTS Inflow Detection

![Simulated DTS temperature profiles](dts_profiles_preview.png)

Simulates hidden inflow events along a pipe using distributed temperature
sensing (DTS) data, and compares three approaches to detecting and locating
them: a naive threshold-based baseline, a random forest, and a 1D
convolutional neural network.

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
- **`naive.py`**, **`ml.py`**, **`cnn.py`** - The three detection approaches
  compared below. Each is a pure collection of functions with no executable
  code of its own.
- **`main.py`** - The only file that actually runs anything. Generates the
  training/test data once and runs all three approaches against it in turn,
  printing a side-by-side comparison.

## Setup

```
pip install -r requirements.txt
```

## Running

```
python main.py
```

This generates 3000 training and 500 test scenarios, evaluates the naive
detector and random forest directly, trains and evaluates the CNN (which
takes a minute or so), and saves a preview plot (`dts_profiles_preview.png`)
of a few example temperature profiles.

## Algorithms compared

### 1. Naive threshold detector (`naive.py`)

Smooths each profile with a moving-average filter (`uniform_filter1d`) to
estimate the "expected" local baseline at every point, then flags an inflow
if any point's deviation from that baseline exceeds a fixed threshold. The
location estimate is simply the position of the single largest deviation
(`argmax`), read off at the full sensor resolution (500 points). No
training involved - it's a fixed rule.

### 2. Random forest (`ml.py`)

Two separate `sklearn` models:
- A **classifier** decides whether a profile contains an inflow at all,
  trained on a handful of summary statistics of the deviation curve (max
  deviation, std deviation, etc.) rather than the raw profile.
- A **regressor** estimates the inflow's position, trained directly on the
  full 500-point deviation curve (summary statistics would throw away the
  positional detail location needs).

### 3. 1D CNN (`cnn.py`)

A small 1D convolutional network (two `Conv1D` + `MaxPooling1D` blocks,
then dense layers) that regresses the inflow's position directly from the
deviation curve. A convolution's kernel is applied with the *same* learned
weights at every position along the pipe, giving the model translation
invariance for free - it only has to learn "what a bump looks like" once,
rather than relearning the same pattern separately at each of the 500
possible positions the way a tree-based model must.

## Results (from `python main.py`, 3000 train / 500 test scenarios)

| | Detection accuracy | Location error (median) | Location error (mean) |
|---|---|---|---|
| Naive threshold | 77.4% | **2.5 m** | 35.2 m |
| Random forest | **79.4%** | 130.4 m | - |
| 1D CNN | - | 34.2 m | 47.2 m |

(Location errors above are measured only on the scenarios the naive
detector actually flagged, for a fair comparison across all three; CNN
error over *all* inflow scenarios in the test set was 68.5 m mean / 40.9 m
median.)

**Detection is close to a three-way tie** - the random forest classifier
edges out the naive threshold slightly (79.4% vs 77.4%), since it combines
several summary statistics of the deviation curve instead of thresholding
on a single one. Not a dramatic difference either way.

**Location is where the naive detector wins decisively**, and by a large
margin. Why:

- **The naive detector reads its answer directly off the full-resolution
  signal.** `argmax(deviation)` picks a position out of the full 500-point
  profile with no compression or downsampling in between. There is no
  function-approximation error to speak of - whatever point has the
  biggest deviation *is* the answer, and physically, that point is almost
  always extremely close to the true inflow location, because that's
  exactly where the temperature bump the physics simulation generates is
  centred. For the typical case, this task is close to trivial for an
  `argmax` - which is why the naive detector's *median* location error is
  only 2.5 m.

- **The random forest regressor lacks translation invariance.** Even
  though it's trained on the full 500-point deviation curve, a tree-based
  model treats each of those 500 columns as an independent, fixed-position
  feature. It has no built-in notion of "a bump," only "column 214 is
  high" - so it has to more or less relearn the same rule separately for
  every possible position along the pipe. With only ~2,100 training
  examples spread across 900 metres of pipe, it never sees enough examples
  at or near most positions to learn this reliably, so its position
  estimates end up only loosely related to the true location - hence the
  130 m median error, actually *worse* than never having compressed the
  input at all.

- **The CNN fixes translation invariance, but still loses on precision.**
  Its convolutional layers do let it recognise "a bump" wherever it occurs,
  which is why it beats the random forest by a wide margin (34 m vs 130 m
  median). But it still loses to the naive rule by roughly an order of
  magnitude, for two structural reasons: (1) two `MaxPooling1D(pool_size=4)`
  layers downsample the 500-point input by a factor of 16 before the final
  dense layers ever see it, capping how precisely the network can localise
  a position in principle, regardless of training; and (2) with only
  ~2,100 positive training examples and a handful of epochs before early
  stopping, there simply isn't enough data for the network to learn a
  sub-pixel-precise mapping from curve shape to position - it learns to
  land in roughly the right neighbourhood, not to pinpoint it.

In short: this is a case where the "naive" rule isn't naive so much as
*close to the correct, near-optimal answer for the physics being
simulated* (a single, clean, isolated peak in a low-noise signal). Both
learned models have to spend some of their capacity and training data
either fighting the raw representation (random forest) or compressing
away spatial precision (CNN), while the simple threshold rule pays neither
cost. Learned models would be expected to pull ahead on messier,
higher-noise, or multi-event data where a fixed rule no longer holds up.
