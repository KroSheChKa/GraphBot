"""
Lightweight obstacle avoidance for Graphwar auto paths.

Strategy (#4): keep piecewise straight ``direct_line`` segments; when a segment
hits a black circle, insert a small detour (up/down, then past the circle).
Output is still a list of game-coordinate waypoints for ``waypoints_to_formula``.
"""

import math

GAME_PRECISION = 5
VERTICAL_MAX_COEFF = 999
VERTICAL_MIN_EPS = 0.001
DEFAULT_CLEARANCE = 0.35


def fmt_game(value):
    return round(float(value), GAME_PRECISION)


def vertical_eps(y_from, y_to, max_coeff=VERTICAL_MAX_COEFF):
    dy = abs(y_to - y_from)
    if dy < 1e-9:
        return VERTICAL_MIN_EPS
    return max(VERTICAL_MIN_EPS, dy / (2 * max_coeff))


def field_obstacles_to_game(obstacles_field, field_width):
    """Convert field-pixel circles (cx, cy, r) to game coords."""
    game = []
    for cx, cy, r in obstacles_field:
        gx = -25 + cx * 50 / field_width
        gy = 15 - cy * 50 / field_width
        gr = r * 50 / field_width
        game.append((fmt_game(gx), fmt_game(gy), fmt_game(gr)))
    return game


def segment_intersects_circle(p1, p2, circle, margin=0):
    cx, cy, r = circle
    r = r + margin
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(x1 - cx, y1 - cy) <= r

    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False

    sqrt_disc = math.sqrt(disc)
    for t in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
        if -1e-9 <= t <= 1 + 1e-9:
            return True
    return False


def _segment_hits_any(p1, p2, obstacles, margin):
    return any(segment_intersects_circle(p1, p2, o, margin) for o in obstacles)


def _path_length(points):
    total = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _append_vertical(points, x_col, y_target):
    """Add a near-vertical step at column x_col (Graphwar needs distinct x)."""
    x_col = fmt_game(x_col)
    y_target = fmt_game(y_target)
    if not points:
        points.append([x_col, y_target])
        return

    last_x, last_y = points[-1]
    if abs(last_y - y_target) < 1e-6:
        return

    eps = fmt_game(vertical_eps(last_y, y_target))
    end_x = fmt_game(x_col + eps)
    if abs(last_x - end_x) > 1e-6 or abs(last_y - y_target) > 1e-6:
        points.append([end_x, y_target])


def _detour_step(p1, circle, obstacles, margin, clearance, y_bounds):
    """One bypass step: vertical to clear height, then past the circle's right edge."""
    cx, cy, r = circle
    R = r + margin + clearance
    y_min, y_max = y_bounds
    x_out = fmt_game(cx + R)

    candidates = []
    for y_clear in (fmt_game(min(y_max, cy + R)), fmt_game(max(y_min, cy - R))):
        if not (y_min - 1e-6 <= y_clear <= y_max + 1e-6):
            continue

        route = [list(p1)]
        _append_vertical(route, p1[0], y_clear)
        if x_out > route[-1][0] + 1e-6:
            route.append([x_out, y_clear])
        if len(route) < 2:
            continue

        ok = True
        for i in range(len(route) - 1):
            a = tuple(route[i])
            b = tuple(route[i + 1])
            if b[0] + 1e-6 < a[0]:
                ok = False
                break
            if _segment_hits_any(a, b, obstacles, margin):
                ok = False
                break
        if ok:
            candidates.append(route)

    if not candidates:
        return None
    return min(candidates, key=_path_length)[1:]


def resolve_segment(p1, p2, obstacles, margin=0, clearance=DEFAULT_CLEARANCE, y_bounds=(-15, 15)):
    """
    Return intermediate waypoints (excluding p1, including p2) to reach p2 from p1
    without crossing obstacles. None if no detour found within iteration budget.
    """
    p1 = (fmt_game(p1[0]), fmt_game(p1[1]))
    p2 = (fmt_game(p2[0]), fmt_game(p2[1]))
    hit_margin = margin + clearance

    if p2[0] + 1e-6 < p1[0]:
        return None

    if not _segment_hits_any(p1, p2, obstacles, hit_margin):
        return [list(p2)]

    current = p1
    mids = []
    for _ in range(24):
        if not _segment_hits_any(current, p2, obstacles, hit_margin):
            mids.append(list(p2))
            return mids

        blocking = [
            o for o in obstacles if segment_intersects_circle(current, p2, o, hit_margin)
        ]
        if not blocking:
            mids.append(list(p2))
            return mids

        obstacle = min(blocking, key=lambda o: o[0])
        step = _detour_step(current, obstacle, obstacles, hit_margin, clearance, y_bounds)
        if step is None:
            return None

        for pt in step:
            mids.append(list(pt))
        current = tuple(mids[-1])

    return None


def build_enemy_chain(
    enemies,
    obstacles,
    margin=0,
    clearance=DEFAULT_CLEARANCE,
    y_bounds=(-15, 15),
):
    """
    Build formula waypoints enemy-to-enemy. First enemy = formula anchor
    (where the graph must already be). Active player is NOT included.
    """
    if not enemies:
        return [], [], []

    first = [fmt_game(enemies[0][0]), fmt_game(enemies[0][1])]
    path = [first]
    hit = [first]
    skipped = []

    for enemy in enemies[1:]:
        enemy_pt = [fmt_game(enemy[0]), fmt_game(enemy[1])]
        segment = resolve_segment(
            tuple(path[-1]),
            tuple(enemy_pt),
            obstacles,
            margin=margin,
            clearance=clearance,
            y_bounds=y_bounds,
        )
        if segment is None:
            skipped.append(enemy_pt)
            continue
        path.extend(segment)
        hit.append(enemy_pt)

    return path, hit, skipped


def build_greedy_enemy_path(
    active,
    enemies,
    obstacles,
    margin=0,
    clearance=DEFAULT_CLEARANCE,
    y_bounds=(-15, 15),
):
    """Legacy planner from active — use build_enemy_chain for formula output."""
    if not enemies:
        return [list(active)], [], list(enemies)

    path, hit, skipped = build_enemy_chain(enemies, obstacles, margin, clearance, y_bounds)
    if not path:
        return [list(active)], [], list(enemies)

    if resolve_segment(
        tuple(active), tuple(path[0]), obstacles, margin, clearance, y_bounds
    ) is None:
        skipped = [path[0]] + skipped

    return path, hit, skipped
