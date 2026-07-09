"""
A* pathfinding on the Graphwar game field.

Grid matches game bounds (x: -25..25, y: -15..15). Movement never decreases x
(same rule as the p5.js prototype). Output is simplified game-coordinate
waypoints for piecewise ``direct_line`` formulas.
"""

import heapq
import math

import cv2

from .avoidance import DEFAULT_CLEARANCE, fmt_game, segment_intersects_circle

X_MIN, X_MAX = -25.0, 25.0
Y_MIN, Y_MAX = -15.0, 15.0
COLS, ROWS = 100, 60

NEIGHBORS = (
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 1),
    (0, -1),
)

NEIGHBOR_COST = {
    (1, 0): 1.0,
    (1, 1): math.sqrt(2),
    (1, -1): math.sqrt(2),
    (0, 1): 1.0,
    (0, -1): 1.0,
}


def game_to_cell(x, y):
    gi = round((float(x) - X_MIN) / (X_MAX - X_MIN) * (COLS - 1))
    gj = round((Y_MAX - float(y)) / (Y_MAX - Y_MIN) * (ROWS - 1))
    gi = max(0, min(COLS - 1, gi))
    gj = max(0, min(ROWS - 1, gj))
    return gi, gj


def cell_to_game(gi, gj):
    x = X_MIN + gi / (COLS - 1) * (X_MAX - X_MIN)
    y = Y_MAX - gj / (ROWS - 1) * (Y_MAX - Y_MIN)
    return fmt_game(x), fmt_game(y)


def _cell_blocked(gi, gj, obstacles, margin):
    cx, cy = cell_to_game(gi, gj)
    for ox, oy, r in obstacles:
        if math.hypot(cx - ox, cy - oy) <= r + margin:
            return True
    return False


def _build_blocked_grid(obstacles, margin):
    return [
        [_cell_blocked(gi, gj, obstacles, margin) for gj in range(ROWS)]
        for gi in range(COLS)
    ]


def _nearest_free(gi, gj, blocked, prefer_right=False):
    if not blocked[gi][gj]:
        return gi, gj

    best = None
    best_score = float("inf")
    for di in range(-10, 11):
        for dj in range(-10, 11):
            ni, nj = gi + di, gj + dj
            if not (0 <= ni < COLS and 0 <= nj < ROWS):
                continue
            if blocked[ni][nj]:
                continue
            if prefer_right and ni < gi:
                continue
            score = di * di + dj * dj
            if score < best_score:
                best_score = score
                best = (ni, nj)
    return best


def _heuristic(gi, gj, goal_i, goal_j, obstacles, margin):
    gx, gy = cell_to_game(gi, gj)
    tx, ty = cell_to_game(goal_i, goal_j)
    dist = math.hypot(gx - tx, gy - ty)

    penalty = 0.0
    for ox, oy, r in obstacles:
        d = math.hypot(gx - ox, gy - oy)
        pad = r + margin + 0.4
        if d < pad:
            penalty += (pad - d) * 1.8
    return dist + penalty


def astar_game(start, goal, obstacles, clearance=DEFAULT_CLEARANCE):
    """
    Find a path between game-coordinate points. Returns list of (x, y) or None.
    """
    margin = clearance
    blocked = _build_blocked_grid(obstacles, margin)

    start_i, start_j = game_to_cell(start[0], start[1])
    goal_i, goal_j = game_to_cell(goal[0], goal[1])

    start_free = _nearest_free(start_i, start_j, blocked, prefer_right=False)
    goal_free = _nearest_free(goal_i, goal_j, blocked, prefer_right=False)
    if start_free is None or goal_free is None:
        return None

    start_i, start_j = start_free
    goal_i, goal_j = goal_free

    if start_i > goal_i:
        return None

    start_key = (start_i, start_j)
    goal_key = (goal_i, goal_j)

    open_heap = [(0.0, start_key)]
    g_score = {start_key: 0.0}
    came_from = {}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_key:
            break

        ci, cj = current
        for di, dj in NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < COLS and 0 <= nj < ROWS):
                continue
            if blocked[ni][nj]:
                continue

            neighbor = (ni, nj)
            step = NEIGHBOR_COST[(di, dj)]
            tentative = g_score[current] + step
            if tentative >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative
            f = tentative + _heuristic(ni, nj, goal_i, goal_j, obstacles, margin)
            heapq.heappush(open_heap, (f, neighbor))
    else:
        return None

    if goal_key not in came_from and goal_key != start_key:
        return None

    cells = [goal_key]
    while cells[-1] != start_key:
        cells.append(came_from[cells[-1]])
    cells.reverse()

    return [cell_to_game(gi, gj) for gi, gj in cells]


def _segment_hits_any(p1, p2, obstacles, margin):
    return any(segment_intersects_circle(p1, p2, o, margin) for o in obstacles)


def simplify_path(points, obstacles, margin):
    """String-pull: keep only corners needed to avoid obstacles."""
    if len(points) <= 2:
        return points

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1:
            if not _segment_hits_any(points[i], points[j], obstacles, margin):
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def build_enemy_chain_astar(
    enemies,
    obstacles,
    clearance=DEFAULT_CLEARANCE,
):
    """
    Chain enemies with A* segments. First enemy = formula anchor (no active player).
    Returns (path_waypoints, hit_enemies, skipped_enemies).
    """
    if not enemies:
        return [], [], []

    margin = clearance
    first = (fmt_game(enemies[0][0]), fmt_game(enemies[0][1]))
    path = [list(first)]
    hit = [list(first)]
    skipped = []

    for enemy in enemies[1:]:
        goal = (fmt_game(enemy[0]), fmt_game(enemy[1]))
        if goal[0] + 1e-6 < path[-1][0]:
            skipped.append(list(goal))
            continue

        raw = astar_game(tuple(path[-1]), goal, obstacles, clearance=clearance)
        if raw is None:
            skipped.append(list(goal))
            continue

        simplified = simplify_path(raw, obstacles, margin)
        for pt in simplified[1:]:
            path.append([pt[0], pt[1]])
        hit.append(list(goal))

    return path, hit, skipped


def game_to_field_px(gx, gy, field_width):
    fx = int((gx + 25) * field_width / 50)
    fy = int((15 - gy) * field_width / 50)
    return fx, fy


def draw_path_on_field(bgr, path_waypoints, field_width):
    if len(path_waypoints) < 2:
        return bgr
    out = bgr.copy()
    pts = [game_to_field_px(p[0], p[1], field_width) for p in path_waypoints]
    for i in range(len(pts) - 1):
        cv2.line(out, pts[i], pts[i + 1], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(out, pt, 3, (0, 255, 0), -1)
    return out
