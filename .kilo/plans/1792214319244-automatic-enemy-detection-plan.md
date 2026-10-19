# Automatic Enemy Detection and UI Toggle Plan

## Goal

Expose the already-computed enemy player list in capture responses, convert it into ordered trajectory-search targets, and add a UI toggle that behaves consistently with `autoDetectActive`.

## Decisions

- Default toggle value: `autoDetectEnemies: true`, mirroring `autoDetectActive`.
- Seeding applies to Dot mode only. Click mode keeps manual target order; Draw mode is unaffected.
- Enemies are seeded sorted by game x ascending (leftmost to rightmost).
- All detected enemies are included; the solver already treats targets left of the active soldier as unreachable.
- Detection failures are non-fatal and yield an empty list.

## Affected Files

- `core/detection.py`: reuse existing `find_all_players()`; no algorithm changes required.
- `core/game_capture.py`: call `find_all_players()`, convert enemy pixels to game coordinates, add `detected_enemies` to response.
- `Visuals in p5.js/universal-approximator/sketch.js`: add state, default param, checkbox, event branch, preference application, capture-time seeding, reset handling, and manual re-add button.

## Backend Implementation

1. Import `find_all_players` in `core/game_capture.py`.
2. After `active_result`, compute:
   ```python
   enemy_result = find_all_players(bgr, field["width"])
   enemies = enemy_result.get("enemies", [])
   ```
3. Build sorted payload:
   ```python
   detected_enemies = []
   for cx, cy, radius in sorted(enemies, key=lambda item: item[0]):
       gx, gy = field_to_game(cx, cy, field["width"], field["height"])
       detected_enemies.append({
           "pixel": {"x": round(float(cx), 4), "y": round(float(cy), 4), "radius": round(float(radius), 4)},
           "game": {"x": gx, "y": gy},
       })
   ```
4. Add `"detected_enemies": detected_enemies` to the response.
5. Wrap detection in try/except so capture still succeeds with an empty list on failure.

## Frontend Implementation

1. Add `autoDetectEnemies: true` to `DEFAULT_PARAMS`.
2. Add global `let capturedEnemyTargets = [];`.
3. Add checkbox next to `autoDetectActive`:
   ```js
   ${controlCheckbox("autoDetectEnemies", "Auto-detect enemy targets", params.autoDetectEnemies,
     "After capture, adds detected right-side enemies as targets, ordered left to right.")}
   ```
4. Add control-change branch:
   ```js
   if (changedKey === "autoDetectEnemies") {
     applyEnemyDetectionPreference();
     updateStatusPanel();
     updateCopyButton();
     return;
   }
   ```
5. Implement `applyEnemyDetectionPreference()`:
   - Rebuild `dotPoints` from the active anchor when present.
   - If enabled and `capturedEnemyTargets` is non-empty, convert game coordinates, sort by x, and append to `dotPoints`.
   - Update status, copy, and dot buttons.
6. Capture handler:
   ```js
   capturedEnemyTargets = Array.isArray(data.detected_enemies) ? data.detected_enemies : [];
   if (params.autoDetectEnemies) applyEnemyDetectionPreference();
   ```
7. `resetParams()` should call `applyEnemyDetectionPreference()` when enabled and data exists.
8. `clearWorkspaceState()` must retain `capturedEnemyTargets`.
9. Add a manual "Add detected enemies" button that calls `applyEnemyDetectionPreference()`.

## Edge Cases

- No enemies: payload is `[]`; no targets added.
- Active not detected: store enemies; seed only after an anchor exists.
- Enemies left of active: included in payload; solver already marks them unreachable.
- Duplicate enemy circles: dedupe by center distance before conversion.
- Side jitter near x=0: existing `classify_side()` radius handling remains authoritative.
- Capture detection failure: non-fatal; empty list returned.

## Validation

- Backend: add a synthetic-scene test asserting `find_all_players(...).enemies` is non-empty and sorted left-to-right.
- Backend: verify payload includes `detected_enemies`.
- Frontend manual checks:
  1. Capture with multiple enemies and confirm Dot targets appear left-to-right.
  2. Toggle off; confirm manual targets are not wiped.
  3. Clear targets, then use "Add detected enemies"; confirm re-population.
  4. Confirm Click and Draw modes are unaffected.
- Run existing tests: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

## Risks

- Enemy detection lacks a unique visual marker, so false positives are possible.
- Auto-seeding only Dot mode avoids overwriting Click mode's ordered segments.
- Enemy detection remains subordinate to active detection; if the active anchor is wrong, seeded targets may be misleading.
