"""
Symbolic GA planner for Graphwar automatic mode.

This planner evolves symbolic expressions y=f(x), then scores the translated
trajectory y_shot(x)=f(x)+c where c anchors the curve to active player.
"""

import math
import random
import time

import cv2

from .avoidance import fmt_game, segment_intersects_circle

X_MIN, X_MAX = -25.0, 25.0
Y_MIN, Y_MAX = -15.0, 15.0

REJECT_SCORE = -1_000_000_000.0

DEFAULT_POPULATION = 50
DEFAULT_ELITES = 5
DEFAULT_MUTATED = 25
DEFAULT_DEPTH = 4
MAX_TREE_NODES = 55

BINARY_OPS = ("+", "-", "*", "/", "^")
UNARY_OPS = ("sqrt", "log", "ln", "abs", "sin", "cos", "tan", "exp")


def _const_node(value):
    return ("const", float(value))


def _var_node():
    return ("var",)


def _unary_node(op, child):
    return ("unary", op, child)


def _binary_node(op, left, right):
    return ("binary", op, left, right)


def _is_finite(value):
    return not (math.isnan(value) or math.isinf(value))


def _safe_eval_unary(op, value):
    if not _is_finite(value):
        return float("nan")
    try:
        if op == "sqrt":
            if value < 0:
                return float("nan")
            return math.sqrt(value)
        if op == "log":
            if value <= 0:
                return float("nan")
            return math.log10(value)
        if op == "ln":
            if value <= 0:
                return float("nan")
            return math.log(value)
        if op == "abs":
            return abs(value)
        if op == "sin":
            return math.sin(value)
        if op == "cos":
            return math.cos(value)
        if op == "tan":
            return math.tan(value)
        if op == "exp":
            out = math.exp(value)
            if not _is_finite(out):
                return float("nan")
            return out
    except (OverflowError, ValueError):
        return float("nan")
    return float("nan")


def _safe_eval_binary(op, left, right):
    if not _is_finite(left) or not _is_finite(right):
        return float("nan")
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if abs(right) < 1e-9:
                return float("nan")
            return left / right
        if op == "^":
            # Prevent undefined fractional power on negative bases.
            if left < 0 and abs(right - round(right)) > 1e-8:
                return float("nan")
            out = math.pow(left, right)
            if not _is_finite(out):
                return float("nan")
            return out
    except (OverflowError, ValueError):
        return float("nan")
    return float("nan")


def evaluate_expression(node, x_value):
    node_type = node[0]
    if node_type == "var":
        return float(x_value)
    if node_type == "const":
        return float(node[1])
    if node_type == "unary":
        op = node[1]
        child = node[2]
        return _safe_eval_unary(op, evaluate_expression(child, x_value))
    if node_type == "binary":
        op = node[1]
        left = node[2]
        right = node[3]
        return _safe_eval_binary(op, evaluate_expression(left, x_value), evaluate_expression(right, x_value))
    return float("nan")


def _tree_size(node):
    node_type = node[0]
    if node_type in ("var", "const"):
        return 1
    if node_type == "unary":
        return 1 + _tree_size(node[2])
    if node_type == "binary":
        return 1 + _tree_size(node[2]) + _tree_size(node[3])
    return 1


def _random_const(rng):
    return _const_node(rng.gauss(0.0, 8.0))


def _random_leaf(rng):
    if rng.random() < 0.55:
        return _var_node()
    return _random_const(rng)


def random_expression(rng, depth):
    if depth <= 0 or rng.random() < 0.25:
        return _random_leaf(rng)

    if rng.random() < 0.4:
        return _unary_node(rng.choice(UNARY_OPS), random_expression(rng, depth - 1))

    return _binary_node(
        rng.choice(BINARY_OPS),
        random_expression(rng, depth - 1),
        random_expression(rng, depth - 1),
    )


def _all_paths(node, base=()):
    out = [base]
    node_type = node[0]
    if node_type == "unary":
        out.extend(_all_paths(node[2], base + (0,)))
    elif node_type == "binary":
        out.extend(_all_paths(node[2], base + (0,)))
        out.extend(_all_paths(node[3], base + (1,)))
    return out


def _get_subtree(node, path):
    if not path:
        return node
    head, *tail = path
    node_type = node[0]
    if node_type == "unary" and head == 0:
        return _get_subtree(node[2], tuple(tail))
    if node_type == "binary":
        if head == 0:
            return _get_subtree(node[2], tuple(tail))
        if head == 1:
            return _get_subtree(node[3], tuple(tail))
    return node


def _replace_subtree(node, path, new_subtree):
    if not path:
        return new_subtree
    head, *tail = path
    node_type = node[0]
    if node_type == "unary" and head == 0:
        return _unary_node(node[1], _replace_subtree(node[2], tuple(tail), new_subtree))
    if node_type == "binary":
        if head == 0:
            return _binary_node(node[1], _replace_subtree(node[2], tuple(tail), new_subtree), node[3])
        if head == 1:
            return _binary_node(node[1], node[2], _replace_subtree(node[3], tuple(tail), new_subtree))
    return node


def _shrink_tree(node, rng, max_nodes=MAX_TREE_NODES):
    out = node
    guard = 0
    while _tree_size(out) > max_nodes and guard < 128:
        paths = _all_paths(out)
        if len(paths) <= 1:
            break
        victim = rng.choice(paths[1:])
        out = _replace_subtree(out, victim, _random_leaf(rng))
        guard += 1
    return out


def mutate_fine(node, rng):
    paths = _all_paths(node)
    const_paths = [path for path in paths if _get_subtree(node, path)[0] == "const"]
    if not const_paths:
        return mutate_region(node, rng)

    target = rng.choice(const_paths)
    old = _get_subtree(node, target)
    old_value = old[1]

    if rng.random() < 0.5:
        new_value = old_value * (1.0 + rng.gauss(0.0, 0.35))
    else:
        new_value = rng.gauss(0.0, 8.0)
    new_value = max(-50.0, min(50.0, new_value))

    return _replace_subtree(node, target, _const_node(new_value))


def mutate_region(node, rng):
    paths = _all_paths(node)
    target = rng.choice(paths)
    new_subtree = random_expression(rng, rng.randint(1, DEFAULT_DEPTH))
    out = _replace_subtree(node, target, new_subtree)
    return _shrink_tree(out, rng)


def crossover(parent_a, parent_b, rng):
    paths_a = _all_paths(parent_a)
    paths_b = _all_paths(parent_b)
    path_a = rng.choice(paths_a)
    path_b = rng.choice(paths_b)
    subtree_b = _get_subtree(parent_b, path_b)
    child = _replace_subtree(parent_a, path_a, subtree_b)
    return _shrink_tree(child, rng)


def _format_number(value):
    rounded = fmt_game(value)
    if abs(rounded) < 1e-9:
        rounded = 0.0
    return str(rounded)


def expression_to_formula(node):
    node_type = node[0]
    if node_type == "var":
        return "x"
    if node_type == "const":
        return _format_number(node[1])
    if node_type == "unary":
        op = node[1]
        child = expression_to_formula(node[2])
        return f"{op}({child})"
    if node_type == "binary":
        op = node[1]
        left = expression_to_formula(node[2])
        right = expression_to_formula(node[3])
        return f"({left}{op}{right})"
    return "0"


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
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def sample_translated_points(genome, x0, y0, step=0.05, x_max=X_MAX):
    if step <= 0:
        step = 0.05
    x_start = max(X_MIN, min(X_MAX, float(x0)))
    x_end = max(x_start, min(X_MAX, float(x_max)))

    base_at_x0 = evaluate_expression(genome, x_start)
    if not _is_finite(base_at_x0):
        return None, True

    offset = float(y0) - base_at_x0

    points = []
    x = x_start
    invalid_math = False
    while x <= x_end + 1e-12:
        y_expr = evaluate_expression(genome, x)
        if not _is_finite(y_expr):
            invalid_math = True
            break
        y = y_expr + offset
        if not _is_finite(y):
            invalid_math = True
            break
        points.append((fmt_game(x), fmt_game(y)))
        x += step
    return points, invalid_math


def evaluate_genome(
    genome,
    x0,
    y0,
    enemies,
    dangers,
    allies,
    bounds=(X_MIN, X_MAX, Y_MIN, Y_MAX),
    step=0.05,
):
    points, invalid_math = sample_translated_points(genome, x0, y0, step=step, x_max=bounds[1])
    if points is None or len(points) < 2:
        return {
            "valid": False,
            "score": REJECT_SCORE,
            "points": [] if points is None else points,
            "hit_enemy_ids": [],
            "ally_hits": 0,
            "hit_danger": False,
            "out_of_bounds": False,
            "invalid_math": True,
        }

    hit_enemy_ids = set()
    hit_ally_ids = set()
    enemy_min_dist_sq = float("inf")
    out_of_bounds = False
    hit_danger = False
    victory_reached = False

    if enemies:
        max_enemy_x = max(enemy[0] for enemy in enemies)
        rightmost_ids = {idx for idx, enemy in enumerate(enemies) if abs(enemy[0] - max_enemy_x) <= 1e-6}
        finish_x = max(enemy[0] + enemy[2] for enemy in enemies)
    else:
        rightmost_ids = set()
        finish_x = bounds[1]

    for idx in range(len(points) - 1):
        p1 = points[idx]
        p2 = points[idx + 1]

        if _segment_out_of_bounds(p1, p2, bounds):
            out_of_bounds = True
            break

        for danger in dangers:
            if segment_intersects_circle(p1, p2, danger, margin=0):
                hit_danger = True
                break
        if hit_danger:
            break

        for enemy_idx, enemy in enumerate(enemies):
            dist = _point_segment_distance(enemy[0], enemy[1], p1[0], p1[1], p2[0], p2[1])
            enemy_min_dist_sq = min(enemy_min_dist_sq, dist * dist)
            if enemy_idx in hit_enemy_ids:
                continue
            if segment_intersects_circle(p1, p2, enemy, margin=0):
                hit_enemy_ids.add(enemy_idx)

        all_enemies_killed = len(hit_enemy_ids) == len(enemies) and len(enemies) > 0
        rightmost_killed = bool(rightmost_ids.intersection(hit_enemy_ids))
        passed_finish_line = p2[0] >= finish_x - 1e-9
        if all_enemies_killed or (rightmost_killed and passed_finish_line):
            victory_reached = True
            break

        for ally_idx, ally in enumerate(allies):
            if ally_idx in hit_ally_ids:
                continue
            if segment_intersects_circle(p1, p2, ally, margin=0):
                hit_ally_ids.add(ally_idx)

    if enemy_min_dist_sq == float("inf"):
        enemy_min_dist_sq = 1_000_000.0

    hits = len(hit_enemy_ids)
    ally_hits = len(hit_ally_ids)

    score = 0.0
    score += hits * 2_000_000.0
    score -= ally_hits * 2_000_000.0
    score += 1_000_000.0 - min(enemy_min_dist_sq, 1_000_000.0)
    score -= _tree_size(genome) * 140.0

    if invalid_math and not victory_reached:
        score = min(score - 4_000_000.0, REJECT_SCORE + 1.0)
    if (out_of_bounds or hit_danger) and not victory_reached:
        score = min(score - 4_000_000.0, REJECT_SCORE + 1.0)

    valid = not (invalid_math or out_of_bounds or hit_danger) or victory_reached
    return {
        "valid": valid,
        "score": score,
        "points": points,
        "hit_enemy_ids": sorted(hit_enemy_ids),
        "ally_hits": ally_hits,
        "hit_danger": hit_danger and not victory_reached,
        "out_of_bounds": out_of_bounds and not victory_reached,
        "invalid_math": invalid_math and not victory_reached,
        "victory_reached": victory_reached,
    }


def _select_parent(scored, rng):
    if not scored:
        return None
    min_score = min(item[0] for item in scored)
    weights = [max(1.0, item[0] - min_score + 1.0) for item in scored]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for weight, item in zip(weights, scored):
        running += weight
        if running >= pick:
            return item[1]
    return scored[0][1]


def _initial_population(prev_best, population_size, rng):
    population = []
    if prev_best is not None:
        population.append(prev_best)
    while len(population) < population_size:
        population.append(random_expression(rng, DEFAULT_DEPTH))
    return population


def search_best_symbolic_ga(
    x0,
    y0,
    enemies,
    dangers,
    allies,
    prev_best=None,
    rng=None,
    step=0.05,
    budget_ms=320,
    max_evals=900,
    population_size=DEFAULT_POPULATION,
):
    rng = rng or random.Random()
    started = time.perf_counter()

    population_size = max(12, int(population_size))
    elite_count = min(DEFAULT_ELITES, max(1, population_size // 5))
    mutated_count = min(DEFAULT_MUTATED, max(1, population_size // 2))

    population = _initial_population(prev_best, population_size, rng)
    best_genome = None
    best_eval = None

    evals = 0
    generation = 0
    rejects_danger = 0
    rejects_bounds = 0
    rejects_invalid = 0
    ally_penalty_hits = 0

    while evals < max_evals and (time.perf_counter() - started) * 1000.0 < budget_ms:
        scored = []
        for genome in population:
            if evals >= max_evals:
                break
            result = evaluate_genome(genome, x0, y0, enemies, dangers, allies, step=step)
            evals += 1

            if result["hit_danger"]:
                rejects_danger += 1
            if result["out_of_bounds"]:
                rejects_bounds += 1
            if result["invalid_math"]:
                rejects_invalid += 1
            if result["ally_hits"] > 0:
                ally_penalty_hits += 1

            score = result["score"]
            scored.append((score, genome, result))
            if best_eval is None or score > best_eval["score"]:
                best_genome = genome
                best_eval = result

            if (time.perf_counter() - started) * 1000.0 >= budget_ms:
                break

        if not scored:
            break

        scored.sort(key=lambda item: item[0], reverse=True)
        generation += 1

        if evals >= max_evals or (time.perf_counter() - started) * 1000.0 >= budget_ms:
            break

        next_population = [item[1] for item in scored[:elite_count]]

        while len(next_population) < elite_count + mutated_count and len(next_population) < population_size:
            parent = _select_parent(scored, rng)
            if parent is None:
                break
            if rng.random() < 0.5:
                child = mutate_fine(parent, rng)
            else:
                child = mutate_region(parent, rng)
            next_population.append(_shrink_tree(child, rng))

        while len(next_population) < population_size:
            parent_a = _select_parent(scored, rng)
            parent_b = _select_parent(scored, rng)
            if parent_a is None or parent_b is None:
                next_population.append(random_expression(rng, DEFAULT_DEPTH))
                continue
            child = crossover(parent_a, parent_b, rng)
            if rng.random() < 0.3:
                child = mutate_fine(child, rng)
            next_population.append(_shrink_tree(child, rng))

        population = next_population

    if best_eval is None:
        best_genome = random_expression(rng, DEFAULT_DEPTH)
        best_eval = evaluate_genome(best_genome, x0, y0, enemies, dangers, allies, step=step)
        evals += 1

    best_formula = expression_to_formula(best_genome)
    elapsed_ms = int((time.perf_counter() - started) * 1000.0)

    return {
        "best_genome": best_genome,
        "best_score": best_eval["score"],
        "best_points": best_eval["points"],
        "hit_enemy_ids": best_eval["hit_enemy_ids"],
        "formula": best_formula,
        "stats": {
            "evals": evals,
            "generations": generation,
            "time_ms": elapsed_ms,
            "rejects_danger": rejects_danger,
            "rejects_bounds": rejects_bounds,
            "rejects_invalid": rejects_invalid,
            "ally_penalty_hits": ally_penalty_hits,
            "ally_hits_best": best_eval["ally_hits"],
        },
    }


def game_to_field_px(gx, gy, field_width):
    fx = int((gx + 25) * field_width / 50)
    fy = int((15 - gy) * field_width / 50)
    return fx, fy


def draw_symbolic_curve_on_field(
    bgr,
    points,
    field_width,
    color=(0, 255, 0),
    thickness=2,
):
    if len(points) < 2:
        return bgr

    out = bgr.copy()
    poly = [game_to_field_px(gx, gy, field_width) for gx, gy in points]
    for idx in range(len(poly) - 1):
        cv2.line(out, poly[idx], poly[idx + 1], color, thickness)
    return out


def draw_symbolic_stats_overlay(bgr, stats_lines):
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
