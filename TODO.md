# GraphBot — TODO

Development plan. Tackle one item at a time, not everything at once.

---

## 1. Teammate avoidance (auto mode)

**Status:** not started

**Current:** in `separate()`, the left half of the field (`x < width/2`) is treated as allies (`good`), the right as enemies (`bad`). Teammates do not appear in the auto formula but are not filtered explicitly either.

**Needed:**
- [ ] Explicitly separate the active player from teammates (not just left/right split).
- [ ] Do not route segments through or toward teammates.
- [ ] Decide how to tell a teammate from an enemy (color, position, size, team).

---

## 2. Enemy as a circle, not a point (auto mode)

**Status:** not started

**Current:** `direct_line()` aims at the circle **center**. Enemy radius (`i[2]` from Hough) is unused.

**Important:** a hit counts anywhere on the enemy **circle**. If radius is off by even a pixel, the trajectory may miss. Need an **accurate** radius estimate, not just the center.

**Idea:** when enemies are close in X, one segment can take several — the line passes through their circles along the way and the formula stays shorter.

**Needed:**
- [ ] Measure each enemy radius accurately (Hough + validation/refinement).
- [ ] Build trajectories using radius, not center only.
- [ ] When enemies are close in X, check whether one segment can clip multiple targets.
- [ ] Optimize target order/set for the shortest formula.

---

## 3. Automatic obstacle avoidance (black circles, auto mode)

**Status:** not started (partial groundwork)

**Current:**
- `detect_black_circles()` exists but is **commented out** in `main()` (mode 0).
- Prototype A* lives in `Visuals in p5.js/astar-pathfinding/` — **not wired** to Python. **Leave p5.js alone for now** — reference only.

**Important:** hitting a black circle ends the round (effective “death”). Avoidance is mandatory in auto mode.

**Needed:**
- [ ] Enable black-circle detection in auto mode.
- [ ] Avoidance algorithm (new or adapted from p5.js ideas) → waypoint chain → `direct_line()` segments.
- [ ] Enforce “movement only to the right” (as in the game).

---

## 4. Better active-player detection (auto mode)

**Status:** ✅ implemented for capture and web UI

**Current:** the red outline identifies the active-player candidate; the Hough player circle and red-ring geometry refine its center. The capture API returns confidence and measured uncertainty, and the web UI uses the result as a persistent anchor.

**Game hint:** the active character has a **slight red outline**. Can be detected with a color mask / contour.

**Needed:**
- [x] Mask for the active player’s red outline (preferred approach).
- [x] Fallback heuristics: size, position, difference from teammates/enemies.
- [x] Tie into the calibration utility (#11) for threshold tuning.

The anchor is retained through mode changes, `C`, and Undo; it changes only on a new capture or manual drag.

---

## 5. Manual mode: click “to the left” = straight down

**Status:** deferred (reverted to simple clicks + sort by X)

**Current in GraphBot.py:** classic mode — clicks sorted by X, `direct_line` segments. Vertical segments are a separate task later.

---

## 6. Graceful “no players found” handling

**Status:** ✅ done

**Implementation (`GraphBot.py`):**
- `warn_no_players()` — clear message + calibration hints
- Auto mode: `continue` instead of crash; separate branch when no enemies on the right
- Click mode: `continue` when no players; no crash when there are no clicks either

---

## 7. UX: fewer F-keys, don’t exit after formula

**Status:** not started (discussion)

**Current:**
- F1 — start after mode selection.
- F2 — quit.
- F3/F4 — begin/end click collection.
- Click mode calls `safe_exit(0)` after the formula — program exits.
- Busy-wait on keys wastes CPU.

**Desired behavior:**
- Fewer required key presses.
- After printing a formula the program **keeps running** (recalculate, new round).
- No spinning `while not key: pass` loops.

**Options (pick when implementing):**

| Option | Auto mode | Manual mode |
|--------|-----------|-------------|
| **A. Right after mode pick** | Start without F1; formula refresh loop | Wait for clicks immediately |
| **B. OpenCV window** | `cv2.waitKey()` instead of busy-wait; `q` quit, `r` refresh | Clicks + RMB or Enter = done |
| **C. Single key** | Space = refresh now, Esc = quit | Space = build formula, Esc = quit |
| **D. Focus-based auto** | When Graphwar is focused — refresh every N ms | — |

**Recommendation:** option **B** — preview window already exists (`cv2.imshow`), natural place for controls; in manual mode **RMB or Enter** instead of F4; **Esc** instead of F2; drop F1/F3.

**Needed:**
- [ ] Agree on control scheme with the user.
- [ ] Replace busy-wait with `waitKey` / timer.
- [ ] Remove `safe_exit(0)` after manual mode — return to loop or wait for next command.
- [ ] Console hints for active keys/actions.

---

## 8. Hard-coded field and window constants

**Status:** not started (low priority)

**Current:** `field = {left: 14, top: 52, width: 772, height: 452}`, window `(-7, 0)`.

**Context:** the game is always one resolution — usually fine. But `left`/`top` margins may differ slightly (Windows 10 vs 11, window frame).

**Needed (optional):**
- [ ] Read Graphwar window size via `GetWindowRect` and derive the field relative to it.
- [ ] Or save margins once from the calibration utility (#11).

---

## 9. Calibration utility (standalone program)

**Status:** not started

**Goal:** small tool to **tune detection coefficients** — adjust sliders/parameters and see results on a game screenshot immediately.

**Show on preview:**
- [ ] Detected enemies (circle + center + radius).
- [ ] Active character (red outline / mask).
- [ ] Teammates (if distinguishable).
- [ ] Black obstacle circles.
- [ ] Captured field bounds (`field`).

**Tune:**
- [ ] Grayscale thresholds for player mask (`lower_bound`, `upper_bound`, hole 169–171).
- [ ] Hough parameters (`param1`, `param2`, `minRadius`, `maxRadius`, blur).
- [ ] Black-circle thresholds.
- [ ] Red-outline thresholds/color for active player.
- [ ] Field margins `left`, `top`, `width`, `height`.

**Format:** separate file, e.g. `calibrate.py` or `GraphBot_calibrate.py`. Save JSON config read by main `GraphBot.py`.

---

## Known issues (reference)

| # | Issue | Status |
|---|-------|--------|
| A | Crash when `detect_players() == None` | ✅ **#6** |
| B | `separate()` — fragile good/bad/active logic | → **#1**, **#4** |
| C | Hard-coded field constants | → **#8**, **#11** |
| D | Windows only | by design |
| E | Busy-wait F1/F3/F4 | → task **#7** |
| F | A* in p5.js — prototype bugs | **do not touch** |

---

- [ ] **#8 (partial):** Win32 field capture + `capture_config.json` — done
- [ ] **#11 (partial):** `calibrate_active.py` — active player calibration via red glow

1. ~~**#6**~~ — done
2. **#11** — calibration (partial: preview + calibrate_active + calibrate_players)
3. **#4** — active player via red outline (partial via calibrate_active)
4. **#7** — UX without F-keys
5. ~~**#5**~~ — vertical “down” in manual mode — done
6. **#1** — teammates
7. **#2** — enemy radius
8. **#3** — black-circle avoidance
9. **#8** — as needed

---

*Last updated: 2025-06-14 (voice-review clarifications)*
