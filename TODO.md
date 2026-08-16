# GraphBot roadmap

GraphBot's primary product is the local browser UI: **Click Mode** and **Draw Mode** are the normal workflows; **Trajectory Search** is an experimental evolutionary solver. This roadmap intentionally contains only future work. Completed milestones belong in Git history and the documentation, not here.

## Near-term product polish

- [ ] Produce the planned README showcase assets: hero workflow, one-drawing/multiple-answers comparison, MLP-training animation, and Trajectory Search evolution.
- [ ] Capture a small set of polished, reproducible UI examples for Click Mode, Draw Mode, and Trajectory Search.
- [ ] Keep the UI as the only public entry point; audit any new documentation or scripts so they do not reintroduce GraphBot.py as the default workflow.
- [ ] Decide whether the legacy research program should remain at the repository root, move to a clearly marked legacy area, or gain a reproducible research harness. Preserve its unique A*, polynomial, and symbolic-GA experiments before any relocation.

## Automatic trajectory search

Trajectory Search already has an increasing-x genome, configurable hit radius, straight/cubic-spline paths, bounds checks, a forbidden-mask penalty, and lexicographic ranking. The next work starts beyond that baseline.

- [ ] Detect enemy targets from a captured field, so manual target placement becomes optional.
- [ ] Distinguish the active player, teammates, and enemies without relying on left/right screen position alone.
- [ ] Estimate usable enemy hit radii from the image and feed them into trajectory scoring.
- [ ] Improve obstacle clearance from a collision penalty into configurable distance-aware routing.
- [ ] Compare several trajectory representations: control-point polylines, splines, and other compact genomes.
- [ ] Add alternative solvers beside the current evolutionary search, then compare planners on the same scene.
- [ ] Support objective trade-offs such as hit count, formula length, safety margin, and target ordering without hiding them behind one opaque score.
- [ ] Build reproducible search scenarios and regression tests for target hits, bounds, and forbidden-mask behavior.

## Approximation laboratory

Current Draw Mode includes linear segments, sigmoid networks, Taylor/polynomial features with optional MLPs, Fourier features with optional MLPs, cubic splines, B-splines, and eight hidden-layer activations.

- [ ] Add a benchmark view: one stroke → every implemented approximator → MSE, maximum error, formula length, and training time.
- [ ] Add hard target/anchor constraints for neural approximators where a curve must pass through designated points.
- [ ] Visualize MLP training over epochs in the UI or a deterministic showcase export.
- [ ] Explore trainable Fourier frequencies and phases.
- [ ] Explore radial-basis networks with Gaussian and Cauchy-style kernels.
- [ ] Explore SIREN-style sine networks.
- [ ] Evaluate wavelet, rational-network, and small KAN-like representations as research experiments.
- [ ] Keep formula complexity as a first-class constraint: a low-error model is not automatically a usable Graphwar expression.

## Text and raster experiments

A future research mode could turn rasterized shapes into high-frequency mathematical paths. It should begin as a standalone experiment, not silently complicate the normal Draw Mode workflow.

- [ ] Rasterize arbitrary text from system fonts, including Unicode, into a black/white mask.
- [ ] Scan masks with configurable vertical density and multiple passes through dark regions.
- [ ] Represent white gaps with near-vertical transitions while preserving a valid one-valued trajectory where possible.
- [ ] Prototype several generators: piecewise linear/spline, sine, sigmoid pairs, sigmoid(sin()) or square-like periodic waves, and Fourier-based paths.
- [ ] Measure export size, numerical stability, and game behavior for very high-frequency formulas.
- [ ] Generalize from text to arbitrary raster images only after the text pipeline is robust.

## Computer vision and tooling

The project already has quiet client-window capture, saved clean field crops, player/active-player calibration, obstacle calibration, and a forbidden-mask configurator.

- [ ] Build a unified calibration UI that consolidates the separate capture, player, active-marker, obstacle, and forbidden-mask tools.
- [ ] Curate saved field captures into a labeled regression dataset.
- [ ] Add detector regression tests against accumulated captures, including active-player position and forbidden-mask quality.
- [ ] Improve player removal from the forbidden mask so sprites and graph strokes are less likely to block valid routes.
- [ ] Derive or validate field bounds dynamically across window sizes, DPI configurations, and Graphwar layouts.
- [ ] Add explicit diagnostics when quiet window capture is unsupported by a particular game/window state.

## Research playground

These are useful directions, not promises of product features.

- [ ] Revisit the existing polynomial planner with reproducible scenes and formula-complexity metrics.
- [ ] Revisit the existing symbolic genetic algorithm with safety, target, and expression-size constraints.
- [ ] Investigate mixture-of-experts or hybrid planners that choose a representation per scene.
- [ ] Evaluate alternative evolutionary algorithms and local optimization after the baseline search is benchmarked.
- [ ] Create a small deterministic scenario suite for comparing all research planners.
- [ ] Keep research dependencies and media-generation tools optional; do not add heavyweight packages to GraphBot's runtime requirements.
