# GraphBot agent context

## Purpose and entry point

GraphBot is a Windows-local helper for Graphwar. Its primary product is the browser UI served by:

```powershell
python tools/approximator_server.py
```

The server serves `Visuals in p5.js/universal-approximator/` and exposes `POST /api/capture`. Do not present `GraphBot.py` as the normal user workflow: it is a retained legacy/research OpenCV console program with experimental A*, polynomial, and symbolic-GA planners.

## Important invariants

- Field coordinates are approximately `x=-25..25`, `y=-15..15`; authoritative transform: `core/field_geometry.py`.
- GraphBot produces expression text; it must not inject into the game process, read game memory, or send gameplay input.
- Web capture must stay quiet: `core/game_capture.py` renders the Graphwar client area off-screen and must never silently substitute arbitrary desktop pixels.
- Current UI internal mode values are `click`, `draw`, and `dot`. The public UI label for `dot` is **Trajectory Search**; do not rename internal identifiers casually.
- The current UI search intentionally moves right only. Targets left of its start are unreachable.
- Formula-export details must respect [GAME_RULES.md](GAME_RULES.md). Keep direct-function text conventions intact: Click Mode exports no `y=`, Draw/Search export `y=...`.

## Key locations

- `Visuals in p5.js/universal-approximator/sketch.js`: UI state, drawing, controls, formula export.
- `network.js`: sigmoid and Taylor/Fourier feature MLPs plus activation export.
- `spline.js`: cubic spline/B-spline fitting and piecewise formula export.
- `dot-mode.js`: evolutionary Trajectory Search (genome, fitness, reproduction).
- `core/game_capture.py`: quiet window capture, detection payload, archive.
- `core/detection.py`: player, active marker, and obstacle detection.
- `core/forbidden_mask.py`: dark-pixel mask and compressed occupancy grid.
- `tools/approximator_server.py`: local server.

## Development checks

Run targeted checks first. The ordinary suite is:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use calibration tools only when working on capture/detection. Do not launch Graphwar, interactive UI sessions, broad recursive scans of generated captures, or heavyweight media generation unless the task requires it.

## Documentation

- Public landing page: [README.md](README.md)
- Mathematics: [docs/MATH.md](docs/MATH.md)
- UI operation: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- Graphwar constraints: [GAME_RULES.md](GAME_RULES.md)
- Future README media: [docs/media/README.md](docs/media/README.md)

Keep public documentation in English. Do not claim a feature exists unless the code supports it. Do not add broken media links; reserve planned assets in the media manifest instead.
