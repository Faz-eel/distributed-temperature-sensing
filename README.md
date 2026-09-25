# Distributed Temperature Sensing (DTS) — Inflow Detection and Location

![Simulated DTS temperature profiles](dts_profiles_preview.png)

A small research-style project simulating a common water infrastructure
problem: a pipe fitted with a fibre-optic cable that measures temperature
continuously along its length (distributed temperature sensing). Water
entering the pipe at an unexpected point — groundwater infiltration, a
cross-connection, an illicit discharge — carries a different temperature
than the main flow, and shows up as a bump or dip in the temperature
profile. The task is to answer two separate questions from that profile
alone: **is** there an inflow, and if so, **where**?

Nothing here uses real sensor data. Every profile is generated from a
physics-based simulation (`physics.py`), so the project is a controlled
way to compare different detection methods against a known, correct
answer.

This README exists mainly to explain *why* the project ended up with the
methods it did — several approaches were tried, some failed in
informative ways, and each failure directly motivated the next attempt.
The numbers below all come from the same train/test data (3000 training
scenarios, 500 test scenarios, fixed random seeds), so they are directly
comparable to one another.

## Running it

```
pip install -r requirements.txt
python main.py
```

The full run takes a few minutes, mostly training the two neural
networks. Everything else in the project is a plain module of functions
with no executable code of its own — `main.py` is the only file that
actually runs anything. It also saves the plot shown above
(`dts_profiles_preview.png`): a normal profile, a cold inflow at 400 m
and a warm inflow at 650 m.

## Files

| File | What it does |
|---|---|
| `physics.py` | The physics: given an inflow's location, temperature, and rate, computes the resulting temperature profile along the pipe (energy-weighted mixing, then exponential decay back toward ground temperature). |
| `scenarios.py` | Generates many random scenarios (with and without an inflow) by calling `physics.py` repeatedly. Returns the temperature profiles (an array of shape `(n_scenarios, n_points)`), the hidden ground truth for each scenario (a table with `has_inflow`, `position_m`, etc.), and the position in metres of each sensor point. |
| `naive.py` | A simple threshold-based baseline: flag an inflow if the profile deviates from its own smoothed baseline by more than a fixed amount; locate it at the point of largest deviation. Only reports a location when it is confident enough to flag something at all. |
| `ml.py` | A random forest classifier (detection) and regressor (location), the first machine learning attempt. |
| `cnn.py` | A 1D convolutional neural network that predicts location as one regressed number. **Did not work well** — kept as a documented negative result. |
| `cnn_softmax.py` | A 1D convolutional neural network that predicts a probability for every position along the pipe. **This is the method that actually solved the location problem**, and its own probabilities turned out to double as a genuine detection signal too, so this single file does both jobs. |
| `main.py` | Runs every method above on the same data and prints a full comparison. |

## The chain of reasoning

### 1. Naive threshold detector — the baseline

Smooth the profile, look at how far the raw signal deviates from that
smoothed version (the *deviation curve*), and flag anything above a fixed
threshold. The location guess is simply the position of the single
largest deviation, reported only when the method is confident enough to
flag something.

Despite being the simplest method by far, this turned out to be very
hard to beat on location specifically: when a real inflow signal exists,
finding its peak directly is already close to the best possible answer,
because a single, cleanly shaped bump doesn't leave much room for a
cleverer method to improve on it.

### 2. Random forest — first machine learning attempt

A classifier decided whether an inflow was present, using a handful of
summary statistics of the deviation curve (maximum deviation, standard
deviation, etc.). A separate regressor tried to predict *where*, fed the
full 500-point deviation curve, since summary statistics would throw away
exactly the positional detail that location needs.

**This is where the first real, informative failure showed up.** Feeding
the model the curve position by position made location prediction
*worse*, not better — even worse than the naive method. The reason: each
of the 500 input columns corresponds to a *fixed* physical position
(column 340 always means "340 metres along the pipe"), and a tree-based
model has no notion that nearby columns are related. It has to learn a
completely separate rule for every possible position an inflow might
occur at, and with only a few thousand training examples, it cannot do
that reliably. An inflow seen near column 340 in training tells the model
almost nothing about one that occurs at column 120 in a different
scenario.

This result — position doesn't behave like an ordinary feature — is the
reason the project moved to convolutional neural networks next.

### 3. CNN regressor — expected to fix the random forest's problem, and didn't

A 1D convolution slides the *same* learned kernel across every position,
so in principle it should recognise "a bump" wherever it occurs, without
needing to relearn the pattern separately for each position — exactly
the property the random forest was missing.

**This is the second informative failure.** The convolutional layer does
successfully find the bump pattern, but the model's final layers
(`Flatten()` followed by `Dense` layers) destroy the direct link between
"a feature was found here" and "here is a physical position on the
pipe." Once the feature map is flattened into one long list of numbers,
the network has no explicit sense of position left, and has to relearn
that mapping from scratch.

Concretely: the convolutional layer is small (16 patterns of 15 numbers
each, 256 learned weights in total) and reuses those same weights at every
position, which is what lets it recognise a bump anywhere. But `Flatten()`
then lays its output out as one long list of 2,000 numbers (125 positions
× 16 pattern scores), and the `Dense` layer that follows gives every one
of those 2,000 slots its own separate weight — over 128,000 of the
model's roughly 128,400 learned numbers. Position survives only as a
slot's place in the list, and nothing tells the layer that neighbouring
slots are neighbouring places on the pipe. To output "about 40% of the
way along" it has to learn that a strong signal in slots 800–815 means
0.4, and then learn the equivalent separately for every other stretch of
the pipe. That is the same problem the random forest had, from the same
kind of limited data (about 2,100 examples spread across the whole
pipe), so it only ever learns it approximately. The result was a median
location error of roughly 55–65 m (naive's median error, when it detects
at all, is 2.5 m).

### 4. CNN with softmax over position — the actual fix

The regressor's failure raised an obvious question: if flattening away
position is the problem, why not build a model whose output has one
slot permanently assigned to each physical chainage, all the way through
to the final layer?

That is exactly what this model does. Instead of ending in a `Flatten()`
plus `Dense(1)`, it ends in a second `Conv1D` layer (with a 1-point
kernel, just to collapse multiple filter channels into a single score
per position) followed directly by a softmax. There is no point at which
position is ever converted into an unordered list of numbers — position
index 340 in the input is still position index 340 in the output. The
single highest-probability position becomes the location guess. Because
each output slot already *is* a position, there is no slot-to-position
mapping left for the network to learn.

This fixed the location problem outright: median location error of
about 1.2 m, better than naive's own 2.5 m.

**A second, unplanned result followed from this.** The model is only
ever trained on scenarios that genuinely have an inflow — there is no
target position to teach it for a normal profile. So the question arose:
what does it do when shown a profile with nothing in it at all? Tested
directly against held-out normal scenarios it never saw during training,
the answer turned out to be useful on its own: with no real bump to lock
onto, the model has nothing to confidently commit to, and spreads its
probability out thinly across all 500 positions instead. The peak
probability it reports averages around 0.5 when a genuine inflow is
present, versus under 0.1 when there is none — a clear, usable gap.
Thresholding that single number (peak probability > 0.1) turns out to
answer detection as well as a purpose-built classifier would, using the
exact same forward pass already computing location. That made a separate
classifier model unnecessary, and this file now answers both of the
project's original questions on its own.

## Summary of results

All figures from the same 500-scenario test set. The random forest and
naive results are identical on every run. The neural network results
vary slightly from run to run (their starting weights are not seeded), so
they are given as approximate figures.

**Detection accuracy (is there an inflow at all):**

| Method | Accuracy |
|---|---|
| Naive threshold | 77.4% |
| Random forest | 79.4% |
| Neural net (softmax model's own peak probability) | ~85–87% |

**Location accuracy (where is it), median error in metres:**

| Method | Median error |
|---|---|
| Naive threshold (only on scenarios it was confident enough to flag) | 2.5 m |
| Random forest (fed the full deviation curve) | 130.4 m |
| CNN regressor (forced to guess on every scenario) | ~55–65 m |
| **CNN, softmax over position** (all scenarios with a real inflow) | **1.2 m** |

Note that naive's location figure only covers the scenarios it was
confident enough to flag in the first place, matching how it is actually
meant to be used, while the other methods are scored on every scenario
that has a real inflow. Even so, the softmax model's median precision
matches or beats naive's.

Medians hide a heavy tail, though. The softmax model's *mean* error is
around 55 m, because roughly one in seven inflow scenarios is missed by
more than 50 m — presumably weak inflows whose bump is buried in sensor
noise, where there is no clear peak to lock onto. Naive shows the same
pattern (2.5 m median, 35 m mean). Both methods are very precise when
the signal is clear and can go badly wrong when it isn't.

## Honest limitations

Everything here is simulated, not measured from a real DTS deployment.
The physics is deliberately simplified — a single isolated inflow event
per scenario, no multiple overlapping events, no noise sources beyond a
fixed sensor error term. Real distributed sensing data is unlikely to be
this clean, and it remains an open question how far these results
generalise to messier, more realistic signals with several overlapping
events. The naive method's strong showing on location is itself a
finding about the *simplicity* of this particular simulated problem, not
a general claim that simple methods beat neural networks at this kind of
task. The softmax model's detection threshold (0.1) was chosen by
inspecting held-out data directly rather than derived from first
principles, and would need re-checking on any new dataset.
