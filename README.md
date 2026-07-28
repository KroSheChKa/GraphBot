# GraphBot

> A passion project for [Graphwar](https://github.com/catabriga/graphwar) — turn what you see on the battlefield into a paste-ready mathematical function.

GraphBot watches the game field and builds Graphwar-compatible formulas. The **recommended workflow** is the local **web UI** (`approximator_server.py`) — **Click mode** (default), **Draw mode** with four approximation methods, and an animated genetic **Dot mode**. **`GraphBot.py`** still offers click mode (OpenCV overlay) and an **automatic mode** prototype that is **not production-ready yet**.

> **Project status:** web UI click + draw = ready to use · Dot mode = first playable GA version · `GraphBot.py` auto mode = work in progress (see [Auto mode](#auto-mode-work-in-progress))

> **How GraphBot touches Graphwar:** GraphBot is an **external helper** — it does **not** modify game files, inject into the game process, read game memory, or automate gameplay. The only direct interaction with the Graphwar window is **moving it to a fixed corner** so screen capture aligns with the configured field region. Everything else is: screenshot → math → copy a formula to your clipboard. **You** paste it into Graphwar yourself.

https://github.com/user-attachments/assets/95afd94f-aecd-4682-b958-3359238795a6

<p align="center">
  <img src="https://github.com/user-attachments/assets/7ee4f917-a18f-490c-a105-48a06fc8f43e" alt="GraphBot preview overlay on the Graphwar field" width="720" />
</p>

---

## Table of contents

- [What is Graphwar?](#what-is-graphwar)
- [Getting started](#getting-started)
- [Web UI (recommended)](#web-ui-recommended)
- [How GraphBot works](#how-graphbot-works)
  - [External tool only (no game tampering)](#external-tool-only-no-game-tampering)
- [Click mode](#click-mode)
- [Draw mode](#draw-mode)
  - [From stroke to dataset](#from-stroke-to-dataset)
  - [Linear segments (same core as click mode)](#1-linear-segments-same-core-as-click-mode)
  - [Sigmoid network (universal approximation)](#2-sigmoid-network-universal-approximation)
  - [Taylor features (polynomial / beta)](#3-taylor-features-polynomial--beta)
  - [Fourier features (harmonics)](#4-fourier-features-harmonics)
  - [Activation functions (Taylor / Fourier MLP)](#activation-functions-taylor--fourier-mlp)
- [Dot mode](#dot-mode)
- [Auto mode (work in progress)](#auto-mode-work-in-progress)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [More to come](#more-to-come)
- [Feedback & issues](#feedback--issues)
- [License](#license)

---

## What is Graphwar?

Graphwar is an artillery game on a Cartesian plane. You type a function; the game fires along that curve (with a vertical shift so the shot passes through your soldier). Hit enemies, avoid teammates and black obstacle circles.

GraphBot does not replace the game — it helps you **derive** functions faster. See [`GAME_RULES.md`](GAME_RULES.md) for full Graphwar rules and syntax.

**Field limits (approx.):** `x ∈ [-25, 25]`, `y ∈ [-15, 15]`.

---

## Getting started

### Requirements

| Requirement | Notes |
|-------------|-------|
| **Windows** | Screen capture and window APIs are Win32-specific (`pywin32`). |
| **Python 3.10+** | Tested with dependencies in `requirements.txt`. |
| **Graphwar** | Window title must be `Graphwar`. Keep it visible while the bot runs. |

### Install

```powershell
git clone https://github.com/KroSheChKa/GraphBot.git
cd GraphBot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run `GraphBot.py` (CLI overlay)

Legacy / alternative entry point — click mode on the live game window with F-keys:

```powershell
python GraphBot.py
```

1. Choose **`1`** (click mode) or **`0`** (auto — experimental).
2. Press **F1** to start, **F2** to quit.
3. In click mode: **F3** start recording clicks, **F4** finish.

See [Click mode → `GraphBot.py`](#graphbotpy-cli) for how formula building differs from the web UI.

**Tip:** If detection looks wrong, tune capture with `python tools/preview_capture.py` and calibration tools under `tools/`.

---

## Web UI (recommended)

The main tool is a local p5.js app served by Python:

```powershell
python tools/approximator_server.py
```

Open **[http://127.0.0.1:8765/](http://127.0.0.1:8765/)** in your browser.

| Mode | What it does |
|------|----------------|
| **1. Click mode** *(default)* | Place waypoints on the canvas; get a piecewise `direct_line` formula |
| **2. Draw mode** | Sketch a curve, resample to a dataset, approximate with 4 methods |
| **3. Dot mode** | Place the active soldier and unordered enemies; evolve a population of left-to-right trajectories |

### Shared controls

| Action | Control |
|--------|---------|
| Capture Graphwar field as background | **Capture field** (Graphwar must be running; window moved to corner for alignment) |
| Clear current path / stroke | **C** *(background screenshot stays)* |
| Copy formula | **Copy y** |
| Reset sliders & canvas state | **Reset** |

**After «Capture field»:** any previous clicks, drawn curve, or Dot-mode population are cleared automatically — you start fresh on the new screenshot.

### Click mode (web UI)

| Action | Control |
|--------|---------|
| Place active soldier | **1st click** — purple marker **A** |
| Place targets | **2nd, 3rd… clicks** — orange markers **2**, **3**… |
| Undo last click | **Right-click**, **Backspace**, or **Undo last click** |

Formula output: **expression only, no `y=` prefix** — paste into Graphwar as-is.

### Draw mode (web UI)

Switch to **2. Draw mode** in the side panel, then pick an approximation method:

| Method | Idea |
|--------|------|
| **2.1 Linear (segments)** | Exact piecewise lines through the dataset |
| **2.2 Sigmoid network** | Sum of shifted sigmoids (universal approximation) |
| **2.3 Taylor (polynomial)** | Polynomial features ± MLP *(beta)* |
| **2.4 Fourier (harmonics)** | Harmonic features ± MLP |

| Action | Control |
|--------|---------|
| Draw target curve | Click and drag on the canvas |
| Adjust dataset density | **Dataset step** slider |
| Hidden-layer activation (Taylor / Fourier) | **Hidden-layer activation** — only when hidden layers ≥ 1 |
| Retrain after parameter change | **Retrain** |

Formula output: **`y=...`** (Graphwar syntax). Compare MSE in the status line before copying.

Taylor and Fourier methods with **≥ 1 hidden layer** use a small MLP on feature vector $\varphi(t)$. Pick the activation in the side panel — see [Activation functions](#activation-functions-taylor--fourier-mlp).

---

## How GraphBot works

```mermaid
flowchart LR
  GW[Graphwar window] --> CAP[Screen capture]
  CAP --> DET[Player / obstacle detection]
  DET --> PATH[Waypoints, drawn curve, or evolved population]
  PATH --> FMT[Graphwar formula]
  FMT --> CLIP[Clipboard]
```

1. **Capture** — crop the game field via Win32 window rect + margins from `config/capture_config.json`.
2. **Detect** — find allies, enemies, active player (red glow), and black obstacles (OpenCV + Hough) *on the screenshot only*.
3. **Plan** — build waypoints (click), freehand draw + resample (draw), or evolve left-to-right control-point paths (dot). *(Auto planners in `GraphBot.py` — A*, polynomial search, symbolic GA — are experimental.)*
4. **Encode** — convert segments or approximations into Graphwar syntax and copy to clipboard.

### External tool only (no game tampering)

GraphBot stays **outside** Graphwar:

| GraphBot does | GraphBot does **not** |
|---------------|------------------------|
| Take a **screenshot** of the visible game field | Edit, patch, or replace any **game files** |
| **Move the Graphwar window** to a known screen position for consistent capture | Inject DLLs, hooks, or code into the **game process** |
| Run **OpenCV** on the captured image | Send keystrokes/clicks **into the game** to play for you |
| Copy a formula to the **clipboard** | Read **game memory** or network traffic |

There is no autopilot that fires shots or submits functions. You still aim by typing (or pasting) the formula in Graphwar’s own UI — GraphBot only helps you **derive** that formula faster.

The piecewise building block is shared between click mode and draw mode (linear segments):

```409:417:GraphBot.py
def direct_line(p1, p2):
    x1, y1 = fmt_game(p1[0]), fmt_game(p1[1])
    x2, y2 = fmt_game(p2[0]), fmt_game(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-12:
        dx = fmt_game(vertical_eps(y1, y2)) if y1 != y2 else VERTICAL_MIN_EPS
        x2 = fmt_game(x1 + dx)
    dist = fmt_game(-((y1 - y2) / 2) / dx)
    return f"{dist}*(abs(x - {x1}) - abs(x - {x2}))"
```

Each segment is a **V-shaped absolute-value line** between two points. A full path is the sum of segments.

---

## Click mode

Both click workflows build a path from **`direct_line` segments** — V-shaped absolute-value pieces between waypoints. The core formula for one segment:

For endpoints $(x_1, y_1)$ and $(x_2, y_2)$:

$$
d = -\frac{y_1 - y_2}{2\,(x_2 - x_1)}, \qquad
\text{segment}(x) = d \cdot \bigl(|x - x_1| - |x - x_2|\bigr)
$$

Full path:

$$
f(x) = \sum_i \text{segment}_i(x)
$$

**Vertical segments:** if the next waypoint has $x$ **to the left** of the previous one (Graphwar expects forward motion), GraphBot inserts a near-vertical step using a tiny $\Delta x$ — same logic in the web UI and in `GraphBot.py`'s `process_clicks_to_waypoints`.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e94bcb04-1525-41aa-baf6-3bdedf8124d5" alt="Click mode — placing waypoints on the field" width="720" />
</p>

### Web UI *(recommended)*

1. Run [the web UI](#web-ui-recommended), optionally **Capture field**.
2. **1st click** — your active soldier (purple **A**). You choose the position manually on the screenshot.
3. **Next clicks** — targets (enemies, detour points) in **click order**.
4. If a click lands **left of the previous waypoint** → vertical segment is inserted automatically.
5. **Copy y** — copies the expression **without `y=`**, e.g.:

```
-1.2*(abs(x - -18.5) - abs(x - -5.2)) + 0.8*(abs(x - -5.2) - abs(x - 12.1))
```

Paste into Graphwar. In normal mode the game still adds its own vertical shift (`+c`) so the shot passes through your soldier.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0ef7b1f7-0342-4c67-96f0-3e5dab0feb5d" alt="Piecewise linear path built from V-shaped segments" width="720" />
</p>

### `GraphBot.py` (CLI)

1. Start `GraphBot.py`, choose mode **`1`**, press **F1**.
2. **F3** — start recording clicks on the **live game field**; **F4** — done.
3. Click **targets only** on the field (clicks outside the capture region are ignored).
4. GraphBot **auto-detects** the active soldier (red glow + OpenCV), **sorts targets by `x`**, builds `soldier → target₁ → target₂ → …`.
5. Formula copied to clipboard — **no `y=` prefix**.

If active-player detection fails, tune `tools/calibrate_active.py`.

| | Web UI click mode | `GraphBot.py` click mode |
|--|-------------------|--------------------------|
| Where you click | Canvas (after screenshot) | Live Graphwar window |
| Active soldier | **Manual** 1st click (**A**) | **Auto-detected** from screenshot |
| Target order | Click order + vertical-left rule | Sorted by `x` |
| Formula prefix | none | none |

---

## Draw mode

Draw mode lives in the [web UI](#web-ui-recommended) only. Sketch a curve, sample it into a dataset, and approximate with one of four methods.

<p align="center">
  <img src="docs/images/draw-mode-example.png" alt="Draw mode building a Fourier approximation from sampled curve points" width="900" />
</p>

<p align="center"><em>Draw mode — sampled curve points and the resulting Fourier approximation on a captured Graphwar field.</em></p>

### From stroke to dataset

```mermaid
flowchart TD
  A[Mouse draw on canvas] --> B[Merge points with same x]
  B --> C[Uniform resample with sample step]
  C --> D[Training dataset blue points]
  D --> E{Approximation method}
  E --> L[Linear segments]
  E --> S[Sigmoid network]
  E --> T[Taylor MLP]
  E --> F[Fourier MLP]
```

1. **Draw** — freehand stroke in game coordinates (`x: -25…25`, `y: -15…15`).
2. **Merge** — points with nearly equal `x` are averaged (stable vertical strokes).
3. **Resample** — uniform steps along `x` controlled by **dataset step** (`sampleStep`). More points → more linear segments; smoother target for neural approximators.
4. **Approximate** — pick a method; compare MSE in the panel; copy the winning formula.

The red curve is your intent; blue dots are the dataset; green is the approximation.

---

### 1. Linear segments (same core as click mode)

Connect consecutive **dataset points** with the same `direct_line` formula as click mode. Segment count ≈ `dataset points − 1`.

**When to use:** You want an exact piecewise path through the samples — same math as click mode, but waypoints come from drawing instead of clicking.

$$
y = \sum_{k=1}^{N-1} d_k \cdot \bigl(|x - x_k| - |x - x_{k+1}|\bigr)
$$

<p align="center">
  <img src="docs/images/draw-mode-linear.png" alt="Linear segment approximation (add screenshot here)" width="720" />
</p>

---

### 2. Sigmoid network (universal approximation)

A shallow network of shifted sigmoids — inspired by the **universal approximation theorem**: a sum of sigmoids can approximate wide classes of curves.

**Model:**

$$
y(x) = b + \sum_{i=1}^{N} w_i \cdot \sigma\!\bigl(k \cdot (x - x_{0,i})\bigr),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Graphwar export** uses the logistic form:

$$
y = b + \sum_i \frac{w_i}{1 + \exp\!\bigl(-k\,(x - x_{0,i})\bigr)}
$$

| Parameter | Role |
|-------------|------|
| `numNeurons` | Number of sigmoid steps |
| `sigmoidK` | Sharpness of each step |
| `stepHeights` | Initialize $w_i$ from target height jumps at each $x_{0,i}$ |
| `freezeX0` | Keep uniform neuron positions while training weights |

<p align="center">
  <img src="docs/images/draw-mode-sigmoid.png" alt="Sigmoid approximation (add screenshot here)" width="720" />
</p>

<!-- Optional: diagram of stacked sigmoids -->
<!-- <img src="docs/images/sigmoid-universal-approximation-diagram.png" alt="Universal approximation with sigmoids" width="640" /> -->

---

### 3. Taylor features (polynomial / beta)

Polynomial features around a scaled origin — related to a **Taylor expansion** mindset: local behavior encoded by powers of $t$.

**Features:**

$$
t = \frac{x - c}{s}, \qquad
\varphi(t) = \bigl[1,\; t,\; t^2,\; \ldots,\; t^n\bigr]
$$

**With hidden layers:** $\varphi(t) \rightarrow \text{MLP with chosen activation} \rightarrow y$ (see [Activation functions](#activation-functions-taylor--fourier-mlp))

**With 0 hidden layers:** pure polynomial in $t$ (expanded to powers of $x$ for Graphwar export).

| Parameter | Role |
|-------------|------|
| `taylorOrder` | Highest power $n$ |
| `taylorHiddenLayers` | `0` = pure polynomial; `>0` = MLP on features |
| `taylorHiddenSize` | Width of hidden layers |
| `mlpActivation` | Nonlinearity between hidden layers (`tanh`, `ReLU`, `Swish`, …) |

<p align="center">
  <img src="docs/images/draw-mode-taylor.png" alt="Taylor feature approximation (add screenshot here)" width="720" />
</p>

---

### 4. Fourier features (harmonics)

Trigonometric basis — same spirit as a **Fourier series** on a normalized interval:

$$
t = \frac{x - c}{s}, \qquad
\varphi(t) = \bigl[1,\; \cos(\pi t),\; \sin(\pi t),\; \cos(2\pi t),\; \sin(2\pi t),\; \ldots\bigr]
$$

**With 0 hidden layers:** linear combination of harmonics (Fourier-like sum).

**With hidden layers:** richer expressivity via MLP on $\varphi(t)$ and a configurable activation (same list as Taylor — [below](#activation-functions-taylor--fourier-mlp)).

| Parameter | Role |
|-------------|------|
| `fourierHarmonics` | Number of harmonic pairs $K$ |
| `fourierHiddenLayers` | `0` = pure harmonic sum |
| `fourierHiddenSize` | Hidden layer width when MLP is used |
| `mlpActivation` | Nonlinearity between hidden layers |

<p align="center">
  <img src="docs/images/draw-mode-fourier.png" alt="Fourier feature approximation (add screenshot here)" width="720" />
</p>

---

### Activation functions (Taylor / Fourier MLP)

When **Taylor** or **Fourier** uses **≥ 1 hidden layer**, the web UI trains a small MLP on feature vector $\varphi(t)$. Choose the nonlinearity in **Hidden-layer activation** (disabled at 0 hidden layers — then the model is purely linear on $\varphi$).

| Activation | Notes |
|------------|--------|
| **tanh** | Default; smooth, bounded |
| **sigmoid (σ)** | Classic logistic |
| **ReLU** | Common in modern nets; see Graphwar export below |
| **Leaky ReLU** | Small slope on $x < 0$ |
| **Softplus** | Smooth ReLU-like: $\ln(1 + e^x)$ |
| **Swish / SiLU** | $x \cdot \sigma(x)$ — popular in EfficientNet-style models |
| **GELU (approx)** | Transformer-style nonlinearity |
| **Mish** | $x \cdot \tanh(\ln(1 + e^x))$ — used in many modern detectors |

All exported formulas use **only** Graphwar builtins: `+`, `-`, `*`, `/`, `^`, `sqrt`, `log`, `ln`, `abs`, `sin`, `cos`, `tan`, `exp`. There is **no** `max()` or `min()` in the game ([`GAME_RULES.md`](GAME_RULES.md)).

#### ReLU without `max()`

Standard ReLU is $\mathrm{ReLU}(x) = \max(0, x)$. Graphwar cannot parse `max`, so GraphBot exports the equivalent form:

$$
\max(0, x) = \frac{x + |x|}{2}
$$

For $x \ge 0$: $|x| = x$ → $(x + x)/2 = x$. For $x < 0$: $|x| = -x$ → $(x - x)/2 = 0$.

<p align="center">
  <img src="docs/images/relu-graphwar-equivalence.svg" alt="ReLU: max(0,x) and (x+|x|)/2 are the same curve" width="640" />
</p>

*Solid blue:* $\max(0, x)$ (used internally while training). *Dashed red:* $(x + |x|)/2$ (what you paste into Graphwar). Same graph.

**Example in a formula** (hidden pre-activation $z$):

```
((z)+abs(z))/2
```

Leaky ReLU uses the same trick: $\max(0,z) + \alpha\min(0,z)$ is written with `abs(z)` only — no `max`/`min`.

---

## Dot mode

Dot mode is the animated genetic-search workflow in the web UI:

1. Switch to **3. Dot mode**.
2. Click the active soldier first (**A**), then click enemy targets in any order.
3. Press **Start evolution** and watch each population grow from A across the field.
4. Stop when satisfied and use **Copy y** to copy the best agent as a Graphwar piecewise-line expression.

Each genome stores `y` values at a fixed, increasing sequence of `x` control points. The first gene is locked to the active soldier and every other value is clamped to `[-15, 15]`, so agents remain left-to-right mathematical functions. Selection is lexicographic: maximize target hits, minimize distance to missed targets, avoid the optional outer edge strips, then prefer smoother and shorter curves. Targets use the configurable **Hit radius** rather than exact point equality.

| Control | Effect |
|---------|--------|
| **Population** | Number of visible agents per generation |
| **Control points** | Genome/path resolution and exported segment count |
| **Hit radius** | Circle around each enemy that counts as a hit |
| **Mutation scale** | Size of random changes between generations |
| **Edge penalty offset** | Places neutral-zone lines inward from `y = ±15`; only trajectory samples beyond them are penalized |
| **Generation time** | How long one animated generation remains on screen |

Blue trails are the current population; the green trail is the current champion. Right-click, Backspace, or Ctrl+Z removes the last point; Space toggles evolution. Targets left of A are marked as unreachable because this first version never moves backward in `x`.

### Dot mode in action

<p align="center">
  <img src="docs/images/dot-mode-example.png" alt="Dot mode genetic algorithm planning a safe trajectory through all targets" width="900" />
</p>

<p align="center"><em>1. Planner result — the evolved champion hits all targets while avoiding the detected forbidden mask.</em></p>

<p align="center">
  <img src="docs/images/GA-ingame.jpg" alt="Dot mode trajectory fired in Graphwar" width="760" />
</p>

<p align="center"><em>2. In-game result — the exported piecewise function reproduced the planned trajectory in Graphwar.</em></p>

After **Capture field**, Python extracts a raster forbidden-mask from black pixels, removes detected players and thin graph strokes, adds a safety margin, and sends a compact occupancy grid to the browser. Dot mode shows it as a translucent red overlay. Safe agents lexicographically outrank every colliding agent; hit count and missed-target distance still outrank the edge-strip penalty, so a necessary border route remains available.

### Forbidden-mask configurator

Tune and inspect the exact data used by Dot mode:

```powershell
python tools/calibrate_forbidden_mask.py
```

You can also open a saved field image:

```powershell
python tools/calibrate_forbidden_mask.py path\to\field.png
```

The dashboard keeps four views together:

1. Original field with final forbidden area in red.
2. Raw pixels accepted by the black threshold.
3. Clean connected areas in green plus the safety expansion in red.
4. The exact occupancy grid transferred to JavaScript.

| Key | Action |
|-----|--------|
| **Space** | Freeze/unfreeze the current live field |
| **F** | Toggle removal of detected players |
| **S** | Save `config/forbidden_config.json` |
| **D** | Export source, intermediate masks, dashboard and JSON report to `outputs/` |
| **R** | Restore default mask parameters |

The raster mask is the collision source of truth: overlapping or nested circles may merge into one connected area without losing their forbidden pixels. Hough circle reconstruction is not used by Dot mode.

> **Current boundary:** player filtering still depends partly on the existing player-circle detector. Dot mode additionally ignores a small area around manually clicked A/enemy points so imperfect player removal does not make valid hits impossible.

---

## Auto mode (work in progress)

> **Status: in development.** Auto mode is not the main focus of the project yet. Core pieces exist (screen capture, player detection, preview overlay, prototype planners), but gameplay-critical behavior is still missing or unreliable — teammate filtering, accurate enemy radius, black-circle avoidance, and stable active-player detection are all on the [roadmap](#roadmap).

Automatic mode (`0` at startup) tries to detect enemies and build formulas without manual input. Treat it as a **preview of what's coming**, not a finished autopilot.

<p align="center">
  <img src="https://github.com/user-attachments/assets/16caabb1-507c-4a7a-bde9-aedb832485d9" alt="Player detection overlay" width="720" />
</p>

### Known limitations (today)

| Area | Current state |
|------|----------------|
| Teammates | Left/right split only — may route through allies |
| Enemies | Aims at circle **centers**, not full hit radius |
| Black obstacles | Detection exists but auto routing is **not fully wired** |
| Active player | Fallback heuristics; red-glow detection still being tuned |
| UX | Busy-wait on F-keys; formula loop is rough around the edges |

### Planners (experimental)

| Planner | Description | Maturity |
|---------|-------------|----------|
| **A* chain** | Path through enemy centers; obstacle avoidance partially implemented | Prototype |
| **Polynomial search** | Sample and mutate polynomials anchored at your soldier; score by hits and penalties | Experimental |
| **Symbolic GA** | Evolve Graphwar-like expressions on live scene data | Experimental |

Polynomial candidate form:

$$
y = y_0 + a_1(x - x_0) + a_2(x - x_0)^2 + a_3(x - x_0)^3 + a_4(x - x_0)^4
$$

Updates roughly every second while Graphwar is visible. Press **F2** to quit.

**For reliable results right now, use the [web UI](#web-ui-recommended) or [`GraphBot.py` click mode](#graphbotpy-cli).**

---

## Project layout

```
GraphBot/
├── GraphBot.py              # Main bot (auto + click modes)
├── core/                    # Capture, detection, pathfinding, planners
├── config/                  # JSON configs (capture, players, obstacles)
├── tools/
│   ├── approximator_server.py   # Web UI server (click + draw + dot modes)
│   ├── calibrate_forbidden_mask.py # Raster forbidden-area dashboard
│   ├── preview_capture.py       # Debug capture region
│   └── calibrate_*.py           # Tune detection parameters
├── Visuals in p5.js/
│   └── universal-approximator/  # Web UI (p5.js + training + Dot-mode GA)
├── docs/images/             # README screenshots (add yours here)
├── GAME_RULES.md            # Graphwar rules reference
├── TODO.md                  # Detailed dev notes
└── outputs/                 # Local logs / temp artifacts (gitignored)
```

---

## Roadmap

High-level checklist distilled from [`TODO.md`](TODO.md). Detailed notes stay in that file.

> **Focus:** most open items below are **auto mode** blockers. Click mode and draw mode are usable today; Dot mode is an evolving first version; auto mode should not be expected to play rounds reliably until these land.

### Auto mode (in development)

- [ ] **Teammate avoidance (auto)** — distinguish allies from enemies beyond left/right split; never route through teammates.
- [ ] **Enemy as a circle** — use radius from Hough, not just center; one segment may hit multiple nearby enemies.
- [ ] **Black obstacle avoidance (auto)** — enable `detect_black_circles()` in auto mode; pathfind around lethal circles.
- [ ] **Active player detection** — prioritize red glow outline over “largest circle” heuristic.

### UX & tooling

- [ ] **Keyboard UX** — replace F1/F3/F4 busy-wait with OpenCV `waitKey`; stay alive after click-mode formula instead of exiting.
- [ ] **Calibration suite** — sliders for Hough thresholds, glow mask, field margins; export JSON for `GraphBot.py`.
- [ ] **Dynamic field bounds** — derive capture rect from window size instead of hard-coded margins.

### Done recently

- [x] **Dot mode v1:** animated populations, lexicographic fitness, start/stop controls, and champion formula export
- [x] **Dot obstacle avoidance:** calibrated raster mask, compact grid transfer, safety-first GA fitness
- [x] **Draw mode:** activation picker for Taylor / Fourier MLP (`tanh`, `ReLU`, `Swish`, `GELU`, `Mish`, …)
- [x] **Graphwar-safe ReLU export** — `max(0,x)` → `(x+|x|)/2` in copied formulas
- [x] Web UI with **Click mode** (default) + **Draw mode** (4 approximation methods) + **Dot mode**
- [x] Click mode: manual soldier (**A**), vertical segments on left-click, formula without `y=`
- [x] Field capture resets previous clicks / strokes in the web UI
- [x] Graceful handling when no players are detected (`GraphBot.py`)
- [x] Click-mode vertical segments in `GraphBot.py` (`process_clicks_to_waypoints`)
- [x] Win32 field capture + `capture_config.json`
- [x] Partial calibration tools (`preview_capture`, `calibrate_active`, `calibrate_players`)

---

## More to come

This repo is actively evolving — a **pet project** built for fun and learning, not a finished product.

The biggest active effort is **auto mode** — obstacle routing, teammate logic, and trustworthy detection. Draw mode and click mode will keep improving too. If you have ideas (especially for auto planners), I'd love to hear them.

---

## Feedback & issues

Something broken? Open an [**Issue**](https://github.com/KroSheChKa/GraphBot/issues) with steps to reproduce, your Windows version, and a screenshot if possible.

Have a feature idea or math trick worth adding? Same place — [**Issues**](https://github.com/KroSheChKa/GraphBot/issues) or a PR. All constructive feedback welcome.

---

## License

MIT — see [`LICENSE`](LICENSE).

Built with curiosity for Graphwar, OpenCV, and a bit of approximation theory.
