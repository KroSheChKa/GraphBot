# README media plan

This directory reserves stable, GitHub-friendly showcase filenames. Do not add placeholder images merely to fill a README slot. Produce polished captures or animations, then place them at the exact paths below; the README structure will not need to change.

GIF is the preferred README animation format. Keep captions and critical contrast readable in both GitHub themes. Source projects and large recording intermediates should stay outside the application runtime dependencies.

## `hero-demo.gif`

- **README section:** immediately below the hero text.
- **Format:** looping GIF, 1440×810 or 1280×720, 6–8 seconds, seamless if practical.
- **Purpose:** explain the whole product without narration.
- **Storyboard:** Graphwar field → Capture field → choose Draw or Click → trajectory preview → formula appears → Copy y.
- **Style:** clean UI capture, deliberate cursor movement, no tiny unreadable controls.

## `approximators-comparison.gif`

- **README section:** Draw Mode / Function laboratory.
- **Format:** GIF or a high-resolution PNG; 1440×810 preferred. Animation: 8–12 seconds.
- **Purpose:** one target drawing receives several real GraphBot representations.
- **Storyboard:** target curve remains visible; reveal Linear, Sigmoid, Taylor, Fourier, and Spline/B-spline one at a time; show method name and a compact MSE label.
- **Rule:** use actual model output from the UI, not hand-drawn substitutes.

## `mlp-training.gif`

- **README section:** Neural approximation.
- **Format:** looping GIF, 1280×720, 6–10 seconds.
- **Purpose:** make epochs, feature inputs, hidden layers, and convergence visually intuitive.
- **Storyboard:** target curve + initially poor approximation → several meaningful training stages → final curve; keep the network schematic minimal.

## `automatic-search.gif`

- **README section:** Trajectory Search.
- **Format:** looping GIF, 1280×720, 6–10 seconds.
- **Purpose:** make population, mutation, selection, and champion convergence obvious.
- **Storyboard:** blue candidates fan out → generations update → target hits improve → green champion remains; include the forbidden-mask overlay only if it is clearly legible.

## `click-mode.gif`

- **README section:** Click Mode.
- **Format:** GIF, 1280×720, 4–6 seconds.
- **Purpose:** first point, waypoints, absolute-value route, copied formula.

## `draw-mode.gif`

- **README section:** Draw Mode.
- **Format:** GIF, 1280×720, 5–8 seconds.
- **Purpose:** draw → select approximator → retrain/preview → formula.

## Optional production workflow

Use a separate workspace for animation sources. Manim is a strong optional choice for equation-to-curve transitions, Fourier components, MLP training sketches, and evolutionary diagrams; it must **not** become a GraphBot runtime dependency. For UI recordings, capture a controlled browser session first, then use an editor or dedicated encoder to crop, annotate, and export an optimized GIF.

Keep source files, fonts, and large intermediate videos out of this repository unless a future media workflow explicitly introduces them.
