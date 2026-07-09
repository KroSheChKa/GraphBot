"""
Polynomial regression planner (JS-like) for Graphwar auto mode.

Approach:
- fit polynomial coefficients with closed-form ridge regression;
- evaluate resulting trajectory against enemies/dangers/allies;
- keep the best-scoring curve.
"""

import random
import time

import cv2
import numpy as np

from .avoidance import fmt_game, segment_intersects_circle

X_MIN, X_MAX = -25.0, 25.0
Y_MIN, Y_MAX = -15.0, 15.0

# More expressive than debug mode, still numerically stable with dx in [0..1].
COEFS_COUNT = 16

REJECT_SCORE = -1_000_000_000.0
DEFAULT_SCORE_PER_HIT = 1000.0
DEFAULT_ALLY_PENALTY = 450.0
DEFAULT_COEFF_PENALTY = 4.0


def _coeff_limit(power):
    # Looser limits so curve can bend earlier (not only near x=right edge).
    return max(0.002, 40.0 / (power ** 0.75))


def x_scale_for_shot(x0):
    """Scale x so dx goes from 0..1 over current shot range."""
    return max(1.0, X_MAX - float(x0))


def x_unit(x, x0):
    """Map shot x-range [x0..X_MAX] to unit range [-1..1]."""
    return 2.0 * (float(x) - float(x0)) / x_scale_for_shot(x0) - 1.0


def _clip_coeffs(coeffs, count=COEFS_COUNT):
    coeffs = list(float(c) for c in coeffs[:count])
    while len(coeffs) < count:
        coeffs.append(0.0)
    out = []
    for idx, coeff in enumerate(coeffs):
        limit = _coeff_limit(idx + 1)
        out.append(max(-limit, min(limit, coeff)))
    return tuple(out)


def _normalize_coeffs(coeffs, count=COEFS_COUNT):
    if coeffs is None:
        return None
    return _clip_coeffs(coeffs, count=count)


def polynomial_y(x, x0, y0, coeffs):
    u = x_unit(x, x0)
    u0 = -1.0  # x == x0

    y = float(y0)
    for power, coeff in enumerate(coeffs, start=1):
        y += coeff * ((u ** power) - (u0 ** power))
    return y


def sample_polynomial_points(x0, y0, coeffs, x_max=X_MAX, step=0.05):
    x_start = max(X_MIN, min(X_MAX, float(x0)))
    x_end = max(x_start, min(X_MAX, float(x_max)))
    if step <= 0:
        step = 0.05

    points = [(x_start, polynomial_y(x_start, x0, y0, coeffs))]
    x = x_start
    while x < x_end - 1e-12:
        x = min(x_end, x + step)
        y = polynomial_y(x, x0, y0, coeffs)
        points.append((x, y))
    return points


def _segment_out_of_bounds(p1, p2, bounds):
    x_min, x_max, y_min, y_max = bounds
    x1, y1 = p1
    x2, y2 = p2
    if x1 < x_min - 1e-9 or x1 > x_max + 1e-9:
        return True
    if x2 < x_min - 1e-9 or x2 > x_max + 1e-9:
        return True
    if min(y1, y2) < y_min - 1e-9:
        return True
    if max(y1, y2) > y_max + 1e-9:
        return True
    return False


def _point_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def evaluate_polynomial(
    coeffs,
    x0,
    y0,
    enemies,
    dangers,
    allies,
    bounds=(X_MIN, X_MAX, Y_MIN, Y_MAX),
    step=0.05,
    score_per_hit=DEFAULT_SCORE_PER_HIT,
    ally_penalty=DEFAULT_ALLY_PENALTY,
    coeff_penalty=DEFAULT_COEFF_PENALTY,
):
    points = sample_polynomial_points(x0, y0, coeffs, x_max=bounds[1], step=step)
    hit_enemy_ids = set()
    hit_ally_ids = set()
    enemy_min_gap = [float("inf")] * len(enemies)

    out_of_bounds = False
    hit_danger = False
    victory_reached = False

    if enemies:
        max_enemy_center_x = max(enemy[0] for enemy in enemies)
        rightmost_enemy_ids = {
            idx for idx, enemy in enumerate(enemies) if abs(enemy[0] - max_enemy_center_x) <= 1e-6
        }
        finish_x = max(enemy[0] + enemy[2] for enemy in enemies)
    else:
        rightmost_enemy_ids = set()
        finish_x = bounds[1]

    for idx in range(len(points) - 1):
        p1 = points[idx]
        p2 = points[idx + 1]

        for enemy_idx, enemy in enumerate(enemies):
            ex, ey, radius = enemy
            seg_dist = _point_segment_distance(ex, ey, p1[0], p1[1], p2[0], p2[1])
            enemy_min_gap[enemy_idx] = min(enemy_min_gap[enemy_idx], seg_dist - radius)
            if enemy_idx in hit_enemy_ids:
                continue
            if segment_intersects_circle(p1, p2, enemy, margin=0):
                hit_enemy_ids.add(enemy_idx)

        all_enemies_killed = len(hit_enemy_ids) == len(enemies) and len(enemies) > 0
        rightmost_killed = bool(rightmost_enemy_ids.intersection(hit_enemy_ids))
        passed_finish_line = p2[0] >= finish_x - 1e-9

        # Victory rule:
        # 1) all enemies are hit, OR
        # 2) rightmost enemy is hit and trajectory reached rightmost edge (cx + r).
        # After this point, later death zones/bounds do not matter for scoring.
        if all_enemies_killed or (rightmost_killed and passed_finish_line):
            victory_reached = True
            break

        # TEMP DEBUG (requested): bounds and danger checks are disabled.
        # if _segment_out_of_bounds(p1, p2, bounds):
        #     out_of_bounds = True
        #     break
        #
        # for danger in dangers:
        #     if segment_intersects_circle(p1, p2, danger, margin=0):
        #         hit_danger = True
        #         break
        # if hit_danger:
        #     break

        for ally_idx, ally in enumerate(allies):
            if ally_idx in hit_ally_ids:
                continue
            if segment_intersects_circle(p1, p2, ally, margin=0):
                hit_ally_ids.add(ally_idx)

    if (out_of_bounds or hit_danger) and not victory_reached:
        return {
            "valid": False,
            "score": REJECT_SCORE,
            "points": points,
            "hit_enemy_ids": sorted(hit_enemy_ids),
            "out_of_bounds": out_of_bounds,
            "hit_danger": hit_danger,
            "ally_hits": len(hit_ally_ids),
            "victory_reached": False,
        }

    hits = len(hit_enemy_ids)
    ally_hits = len(hit_ally_ids)
    coeff_norm = sum(abs(c) for c in coeffs)
    # Keep near-miss bonus weak: hits must dominate.
    proximity_bonus = 0.0
    for idx, gap in enumerate(enemy_min_gap):
        if idx in hit_enemy_ids:
            continue
        proximity_bonus += max(0.0, 3.0 - max(0.0, gap)) * 10.0

    score = (
        hits * 100_000.0
        + proximity_bonus
        - ally_hits * ally_penalty
        - coeff_norm * 0.1
    )

    return {
        "valid": True,
        "score": score,
        "points": points,
        "hit_enemy_ids": sorted(hit_enemy_ids),
        "out_of_bounds": out_of_bounds and not victory_reached,
        "hit_danger": hit_danger and not victory_reached,
        "ally_hits": ally_hits,
        "victory_reached": victory_reached,
        "proximity_bonus": proximity_bonus,
    }


def _basis_matrix(x_values, x0, coeff_count):
    x_norm = np.array([x_unit(x, x0) for x in x_values], dtype=np.float64)
    x0_unit = -1.0
    degrees = np.arange(1, coeff_count + 1, dtype=np.float64)
    return np.power(x_norm[:, None], degrees[None, :]) - np.power(x0_unit, degrees[None, :])


def _fit_polynomial_closed_form(
    x0,
    y0,
    train_points,
    rng,
    coeff_count=COEFS_COUNT,
    reg_l2=1e-6,
    init_coeffs=None,
):
    if not train_points:
        return (0.0,) * coeff_count, 0.0

    xs = np.array([pt[0] for pt in train_points], dtype=np.float64)
    ys = np.array([pt[1] for pt in train_points], dtype=np.float64)
    ws = np.array([pt[2] for pt in train_points], dtype=np.float64)
    ws = np.maximum(ws, 1e-6)
    ws /= np.mean(ws)

    basis = _basis_matrix(xs, x0, coeff_count)
    target = ys - float(y0)

    w_sqrt = np.sqrt(ws)
    a_mat = basis * w_sqrt[:, None]
    b_vec = target * w_sqrt

    lhs = a_mat.T @ a_mat + reg_l2 * np.eye(coeff_count)
    rhs = a_mat.T @ b_vec
    try:
        coeffs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    coeffs = np.array(_clip_coeffs(coeffs, count=coeff_count), dtype=np.float64)
    pred = basis @ coeffs
    err = pred - target
    loss = float(np.mean(ws * err * err) + reg_l2 * np.mean(coeffs * coeffs))
    return tuple(float(v) for v in coeffs), loss


def _build_train_points(rng, x0, y0, enemies, dangers, support_indices):
    points = [(x0, y0, 4.0)]
    for idx in support_indices:
        ex, ey, _ = enemies[idx]
        points.append((float(ex), float(ey), 10.0))
    return points


def polynomial_to_graphwar_formula(x0, y0, coeffs):
    x0 = fmt_game(x0)
    y0 = fmt_game(y0)
    rounded = [fmt_game(c) for c in coeffs]
    scale = fmt_game(x_scale_for_shot(x0))
    u_term = f"(2*(x + {abs(x0)})/{scale} - 1)" if x0 < 0 else f"(2*(x - {x0})/{scale} - 1)"
    u0 = -1

    parts = [f"{y0}"]
    for power, coeff in enumerate(rounded, start=1):
        parts.append(f"({coeff})*(({u_term})^{power} - ({u0})^{power})")
    return " + ".join(parts).replace("+ (-", "- (")


def search_best_polynomial(
    x0,
    y0,
    enemies,
    dangers,
    allies,
    prev_best=None,
    rng=None,
    step=0.05,
    budget_ms=110,
    max_evals=280,
):
    rng = rng or random.Random()
    started = time.perf_counter()

    best_coeffs = _normalize_coeffs(prev_best, count=COEFS_COUNT)
    best_eval = None
    if best_coeffs is not None:
        best_eval = evaluate_polynomial(best_coeffs, x0, y0, enemies, dangers, allies, step=step)

    current_eval = None
    evals = 0
    rejects_danger = 0
    rejects_bounds = 0
    ally_penalty_hits = 0
    cap_count = 9999

    if (time.perf_counter() - started) * 1000.0 < budget_ms:
        support_indices = tuple(range(len(enemies)))
        train_points = _build_train_points(rng, x0, y0, enemies, dangers, support_indices)
        coeffs, _ = _fit_polynomial_closed_form(
            x0=x0,
            y0=y0,
            train_points=train_points,
            rng=rng,
            coeff_count=COEFS_COUNT,
            reg_l2=1e-6,
        )

        result = evaluate_polynomial(coeffs, x0, y0, enemies, dangers, allies, step=step)
        current_eval = result
        evals = 1

        if result["hit_danger"] and rejects_danger < cap_count:
            rejects_danger += 1
        if result["out_of_bounds"] and rejects_bounds < cap_count:
            rejects_bounds += 1
        if result["ally_hits"] > 0:
            ally_penalty_hits += 1

        if best_eval is None or result["score"] > best_eval["score"]:
            best_coeffs = coeffs
            best_eval = result

    # Refinement on all enemies intentionally disabled for now:
    # it tends to pull curve into compromise trajectories.

    if best_eval is None:
        best_coeffs = best_coeffs or (0.0,) * COEFS_COUNT
        best_eval = evaluate_polynomial(best_coeffs, x0, y0, enemies, dangers, allies, step=step)

    best_formula = polynomial_to_graphwar_formula(x0, y0, best_coeffs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "best_coeffs": best_coeffs,
        "best_score": best_eval["score"],
        "best_points": best_eval["points"],
        "hit_enemy_ids": best_eval["hit_enemy_ids"],
        "formula": best_formula,
        "current_points": [] if current_eval is None else current_eval["points"],
        "stats": {
            "evals": evals,
            "time_ms": int(elapsed_ms),
            "rejects_danger": rejects_danger,
            "rejects_bounds": rejects_bounds,
            "ally_penalty_hits": ally_penalty_hits,
            "ally_hits_best": best_eval["ally_hits"],
            "proximity_bonus_best": int(best_eval.get("proximity_bonus", 0.0)),
        },
    }


def game_to_field_px(gx, gy, field_width):
    fx = int((gx + 25) * field_width / 50)
    fy = int((15 - gy) * field_width / 50)
    return fx, fy


def draw_polynomial_curve_on_field(
    bgr,
    points,
    field_width,
    color=(0, 255, 0),
    thickness=2,
):
    if len(points) < 2:
        return bgr

    out = bgr.copy()
    poly = []
    for gx, gy in points:
        fx, fy = game_to_field_px(gx, gy, field_width)
        poly.append((fx, fy))

    for idx in range(len(poly) - 1):
        cv2.line(out, poly[idx], poly[idx + 1], color, thickness)
    return out


def draw_polynomial_stats_overlay(bgr, stats_lines):
    out = bgr.copy()
    y = 22
    for line in stats_lines:
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return out
