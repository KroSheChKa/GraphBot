# GraphBot

A bot to smash players in the game Graphwar


https://github.com/user-attachments/assets/95afd94f-aecd-4682-b958-3359238795a6


![image](https://github.com/user-attachments/assets/7ee4f917-a18f-490c-a105-48a06fc8f43e)

"Fake" piecewise function
![image](https://github.com/user-attachments/assets/0ef7b1f7-0342-4c67-96f0-3e5dab0feb5d)

**Click mode**

![20241229000123_1](https://github.com/user-attachments/assets/e94bcb04-1525-41aa-baf6-3bdedf8124d5)

Players detection

![image](https://github.com/user-attachments/assets/16caabb1-507c-4a7a-bde9-aedb832485d9)

## Local workspace hygiene

- `outputs/` — local outputs, logs, screenshots and temporary artifacts.
- `sandbox/tester_programs.py` — single file for quick test scripts/experiments.
- `tools/` — helper calibration/preview scripts (`preview_capture.py`, `calibrate_*.py`).
- `core/` — main bot modules (`detection`, `pathfinding`, `window_capture`, `avoidance`, `polynomial_planner`, `symbolic_ga_planner`).
- `config/` — JSON configs (`capture_config.json`, `players_config.json`, `obstacles_config.json`, `active_config.json`).

## Auto planners

In `GraphBot.py`:
- choose mode `0` (automatic);
- choose planner:
  - `0` — `A* chain` (existing planner);
  - `1` — `Polynomial Search` (closed-form polynomial fitting);
  - `2` — `Symbolic GA` (Graphwar-like expression evolution on OpenCV scene data).

`Polynomial Search` builds candidates of this form:

`y = y0 + a1(x - x0) + a2(x - x0)^2 + a3(x - x0)^3 + a4(x - x0)^4`

What it does:
- samples many candidate curves and scores them (`hits * 1000 - penalties`);
- rejects curves that hit danger zones or leave the field;
- penalizes ally hits;
- mutates best candidates to improve score;
- overlays current/best curve and search stats in preview window.
