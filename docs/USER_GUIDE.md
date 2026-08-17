# GraphBot user guide

## Install and start

GraphBot's primary interface is a local browser UI on Windows.

```powershell
git clone https://github.com/KroSheChKa/GraphBot.git
cd GraphBot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools/approximator_server.py
```

The server opens `http://127.0.0.1:8765/`. If it does not, open that address manually. Keep the PowerShell window running while using the UI.

## Capture a field

1. Start Graphwar and make sure its window title is `Graphwar`.
2. In GraphBot, select **Capture field**.
3. GraphBot asks Windows to render Graphwar's client area into an off-screen bitmap, then crops the configured field. It does not focus or move the game window.

The capture becomes the canvas background. The API also attempts to detect the active player and construct a forbidden grid from dark obstacle pixels. A clean raw field crop is stored locally under `data/field_captures/` for calibration and regression work; that folder is ignored by Git.

If auto-detection chooses the wrong active player, disable **Auto-detect active player** and place the purple **A** marker yourself.

## Click Mode

Use Click Mode when you want direct control over the route.

1. Click the active soldier as the first point.
2. Click targets/waypoints in the order the path should follow.
3. Undo with right-click, Backspace, Ctrl+Z, or **Undo last click**.
4. Select **Copy y** and paste the expression into Graphwar.

The copied Click Mode formula has no `y=` prefix. This is intentional. The mode joins points using absolute-value segments. If you click left of the prior point, GraphBot creates a near-vertical transition because a single function cannot have a truly vertical segment.

## Draw Mode

Use Draw Mode to turn a sketched shape into an approximation.

1. Select **Draw Mode**.
2. Drag on the canvas to draw a target curve.
3. Choose an approximation method and tune its visible controls.
4. Use **Retrain** after changing neural-model parameters.
5. Compare the status MSE and press **Copy y** for the active method.

Available methods are:

- **Linear:** joins sampled points exactly with straight segments.
- **Sigmoid:** a shallow trainable sum of logistic steps.
- **Taylor:** power features, optionally followed by a feature MLP.
- **Fourier:** harmonic features, optionally followed by a feature MLP.
- **Cubic spline:** natural or clamped cubic interpolation; optionally use the B-spline basis fit.

For Taylor and Fourier, hidden layers set to zero produce a linear model on their feature basis. With one or more hidden layers, choose one of the supported activations: tanh, sigmoid, ReLU, Leaky ReLU, Softplus, Swish/SiLU, GELU, or Mish.

Enable **Prevent backward drawing** when your stroke should remain a function of $x$. The canvas will keep the greatest reached $x$ while still allowing vertical movement.

### Private training animation video

This control is intentionally hidden in the normal UI. Start the local server with `python tools/approximator_server.py --record-training` to enable it. Sigmoid, Taylor, and Fourier methods can then show their fit changing over time. After drawing a curve, set **Epochs per frame** and **Frame delay**, then select **Record training video**. The canvas plays a fresh training run, so each frame contains the approximation after the next group of epochs. Only the field canvas is recorded; controls and the rest of the desktop are excluded.

When the run finishes, GraphBot uploads the browser-produced WebM to the local server. It is saved as `outputs/recordings/training-*.webm` (an ignored generated-artifact directory). Use a current Chrome, Edge, or Firefox build with canvas recording support. Linear and spline fits have no iterative training epochs, so the recording control is not shown for them.

## Trajectory Search

Trajectory Search is the public name for the UI's evolutionary-search mode. It is an experimental solver.

1. Select **Trajectory Search**.
2. Click the active soldier first, then add enemy targets in any order.
3. Pick straight or cubic-spline trajectories, then adjust population, control points, mutation scale, and hit radius if needed.
4. Leave **Avoid detected black zones** enabled after a successful field capture when the obstacle mask looks reasonable.
5. Press **Start evolution**. The blue lines are candidates; green is the champion.
6. Stop when satisfied, then use **Copy y**.

The solver moves only to increasing $x$. It marks a target left of the active soldier as unreachable. Its collision grid comes from captured image pixels, so inspect the red forbidden-mask overlay before trusting an obstacle-sensitive route.

When the server was started with `--record-training`, Trajectory Search also shows **Generations to record** and **Record generations**. It restarts the evolutionary run, records the specified number of generations of the canvas, then stops on the final champion and saves `outputs/recordings/trajectory-*.webm`. The blue population and the green best-ever champion are both included in the recording.

## Calibration

Capture is configured with client-relative margins in `config/capture_config.json`. Use these focused utilities when detection needs adjustment:

```powershell
python tools/preview_capture.py
python tools/calibrate_active.py
python tools/calibrate_players.py
python tools/calibrate_forbidden_mask.py
```

`preview_capture.py` adjusts the field crop. `calibrate_active.py` reviews the active-player estimate. `calibrate_players.py` and `calibrate_obstacles.py` target their respective circle detectors. `calibrate_forbidden_mask.py` previews the exact mask/grid used by Trajectory Search and saves `config/forbidden_config.json` with **S**.

## Troubleshooting

### “Window «Graphwar» not found”

Run Graphwar and check that its title is exactly `Graphwar`.

### “Could not capture Graphwar quietly”

Windows or the game declined off-screen rendering. Make sure the window exists and is not in a state that prevents drawing, then retry. The UI intentionally does not fall back to capturing arbitrary desktop pixels, because that could silently capture another app instead of Graphwar.

### The crop is offset or includes UI chrome

Run `python tools/preview_capture.py`, tune the saved margins, and capture again.

### The active-player marker is wrong

Disable auto detection and set **A** manually, or tune it with `calibrate_active.py`.

### The search drives through an obstacle or refuses a clear route

Show the forbidden overlay. The image mask may need calibration, or may be filtering player/graph pixels imperfectly. Re-run `calibrate_forbidden_mask.py`; alternatively disable avoidance for a visual experiment and review the candidate route yourself.

### A copied expression is too complex or behaves badly

Use fewer neural layers/neurons, fewer Fourier harmonics, fewer spline control points, or a simpler representation. Avoid division by values near zero and overly rapid oscillation. See [GAME_RULES.md](../GAME_RULES.md) for Graphwar constraints.

## Formula behavior

GraphBot copies expression text only. It does not type, paste, or fire into Graphwar. In normal-function mode, Graphwar vertically translates the curve through the soldier; constants therefore do not change the resulting path. See [the rules reference](../GAME_RULES.md) before using differential-equation modes.
