# Dynamic UI Graph Sizing Plan

## Goal
Size the UI canvas container to match the natural dimensions of the captured graph image, eliminating scaling/truncation to a fixed 1000×600 container. The graph fills its container at native resolution; canvas scales to fit the graph, not vice versa.

## Decisions (resolved)
1. **Backend returns raw crop at natural dimensions** — Python stops applying `resize_to_grid` for output sizing; reports `raw_width`/`raw_height` in `/api/capture` response.
2. **Natural aspect ratio** — No 5:3 enforcement. `pixel_to_game`/`worldToScreen` scale each axis independently over its own game range; game-coordinate math unchanged.

## Affected files
- `core/game_capture.py` — capture pipeline output sizing
- `core/field_capture_archive.py` — archive stores raw crop (no change needed, already stores `width`/`height`)
- `Visuals in p5.js/universal-approximator/sketch.js` — canvas sizing, bgImage draw, capture response handling
- `Visuals in p5.js/universal-approximator/style.css` — container/layout CSS
- `tools/approximator_server.py` — verify `/api/capture` passes through width/height (check)

## Task list

### 1. Python backend: expose natural graph dimensions
In `core/game_capture.py` `capture_game_field`:
- Keep `auto_crop_white_grid` + `crop_client_field` as-is (they isolate the graph area).
- Do **not** call `resize_to_grid` on the image sent to the browser. Send the raw auto-cropped field.
- Set response `width`/`height` = actual shape of `grid_bgr` (after auto-crop).
- Add `raw_width`/`raw_height` aliases equal to `width`/`height` for UI clarity.
- Keep `resize_to_grid` function defined but unused by the capture path (or repurposed as optional). Do not delete—tests reference it.
- Add size caps: clamp natural dims to `[MIN_W, MIN_H] = [400, 240]` and `[MAX_W, MAX_H] = [1600, 960]`. If outside caps, center-crop/pad to cap before encoding (protects fill-rate and extreme layouts).
- Preserve `field` metadata (`left`/`top` = 0 since crop starts at graph origin).

### 2. Server: verify `/api/capture` response passthrough
Check `tools/approximator_server.py` passes `width`/`height`/`raw_width`/`raw_height` through to JSON. Add fields if missing.

### 3. UI: dynamic canvas sizing
In `sketch.js`:
- Remove fixed `CANVAS_WIDTH`/`CANVAS_HEIGHT` constants (or keep as fallback defaults).
- In `captureGameField()`, after `bgImage` loads:
  - Read `data.raw_width` / `data.raw_height` (fallback to `data.width`/`data.height`, then `CANVAS_WIDTH`/`CANVAS_HEIGHT` defaults).
  - Call `resizeCanvas(rawWidth, rawHeight)` to resize the p5 canvas to match the graph.
  - `width`/`height` globals update automatically; `worldToScreen`/`screenToWorld` remain proportionally correct.
- Before first capture: create canvas at fallback default size (e.g., 1000×600) in `setup()`.
- `draw()` `image(bgImage, 0, 0, width, height)` already stretches to fill canvas — works at any size.
- `drawGrid()`, `drawAxes()` use `lineWorld`/`worldToScreen` — already axis-independent.
- `drawForbiddenGridOverlay()` already scales off stored `image_width`/`image_height` vs live `width`/`height` — no change needed.
- `dotTargetRadius` uses `(params.dotTargetRadius * width) / (X_MAX - X_MIN)` — adapts to canvas width.

### 4. CSS/layout: dynamic container
In `style.css`:
- `#canvas-container`: currently `flex: 0 0 auto` (no explicit size) — okay, but ensure it doesn't constrain canvas. Keep `flex: 0 0 auto`.
- `canvas`: `display: block` already set. Add `max-width: 100vw; max-height: calc(100vh - ...)`? Optional guard against oversized canvas overflowing viewport. Recommend: leave unconstrained (user said "container dimensions automatically adjust"); add overflow handling only if viewport overflow is observed.
- `#controls-panel`: fixed `width: 280px`. At small canvas widths (< ~500px), sidebar may crowd. Recommend: add `@media (max-width: 700px)` stack controls below canvas (flex-direction column on `.workspace`).

### 5. Tests
Update `tests/test_game_capture.py`:
- `ResizeToGridTests`: keep existing tests (function still exists). Add test for `resize_to_grid` being bypassed in capture path? Capture path isn't unit-testable without a window; test `auto_crop_white_grid` returns expected box dims instead.
- Add test verifying capture response `width`/`height` match auto-crop box dims (mock `grab_client_bgr`/`find_game_window`).

## Failure modes & rollout
- **No capture yet**: canvas falls back to default size; draws empty background grid. No regression.
- **Capture fails**: canvas retains previous size/image; error alert shown (existing behavior).
- **Extreme aspect (very wide/tall)**: capped at MAX_W/MAX_H; center-cropped to cap if raw exceeds cap. Graph edges may be trimmed at cap boundary (acceptable safety valve).
- **Raw crop smaller than MIN**: padded to MIN with white; graph centered.
- **Server older than UI**: UI falls back to `data.width`/`data.height`, then to default 1000×600. Backward compatible.

## Validation
- Run `python -m unittest discover -s tests` — all existing tests must pass.
- Manual: capture a field, verify canvas dimensions equal returned `raw_width`/`raw_height`, verify gridlines/axes align with graph edges, verify no white padding at canvas borders.
- Verify `drawForbiddenGridOverlay` still aligns to graph cells after resize.

## Open questions (mark if out of scope)
- None blocking. Size-cap values [400×240 min / 1600×960 max] are recommendations; adjust if needed.