/*
 * Dot mode genetic search.
 *
 * This file deliberately has no p5.js or DOM dependencies. The UI owns clicks,
 * timing and rendering; DotEvolution owns genomes, fitness and reproduction.
 * Future constraints (for example lethal circles) can be added in
 * evaluateAgent() without changing the canvas lifecycle.
 */

class DotEvolution {
  constructor(start, targets, config = {}) {
    this.start = { x: Number(start.x), y: Number(start.y) };
    this.targets = targets.map((target) => ({
      x: Number(target.x),
      y: Number(target.y),
    }));
    this.config = {
      populationSize: Math.max(8, Math.round(config.populationSize ?? 48)),
      controlPoints: Math.max(4, Math.round(config.controlPoints ?? 12)),
      targetRadius: Math.max(0.05, Number(config.targetRadius ?? 0.65)),
      mutationScale: Math.max(0.02, Number(config.mutationScale ?? 0.9)),
      edgeOffset: Math.max(0, Number(config.edgeOffset ?? 1.0)),
      eliteFraction: Math.min(0.35, Math.max(0.04, Number(config.eliteFraction ?? 0.12))),
      yMin: Number(config.yMin ?? -15),
      yMax: Number(config.yMax ?? 15),
      xMax: Number(config.xMax ?? 25),
      constraintEvaluators: Array.isArray(config.constraintEvaluators)
        ? config.constraintEvaluators
        : [],
    };

    this.generation = 0;
    this.stagnantGenerations = 0;
    this.best = null;
    this.bestEver = null;
    this.knotXs = this.buildKnotXs();
    this.population = this.createInitialPopulation();
    this.evaluatePopulation();
  }

  buildKnotXs() {
    const farthestTargetX = this.targets.reduce(
      (value, target) => Math.max(value, target.x),
      this.start.x
    );
    const endX = Math.min(
      this.config.xMax,
      Math.max(this.start.x + 0.5, farthestTargetX)
    );
    const count = this.config.controlPoints;
    const xs = [];
    for (let i = 0; i < count; i++) {
      xs.push(this.start.x + ((endX - this.start.x) * i) / (count - 1));
    }
    return xs;
  }

  createInitialPopulation() {
    const population = [];
    const heuristic = this.buildHeuristicGenes();

    // A noisy informed minority gives the search a direction without handing it
    // a perfect answer; the rest of the generation keeps useful diversity.
    const guidedCount = Math.max(2, Math.floor(this.config.populationSize * 0.18));
    for (let i = 0; i < guidedCount; i++) {
      const genes = heuristic.slice();
      for (let gene = 1; gene < genes.length; gene++) {
        genes[gene] = this.clampY(
          genes[gene] + this.gaussian() * this.config.mutationScale * 2.2
        );
      }
      population.push(this.makeAgent(genes));
    }

    while (population.length < this.config.populationSize) {
      population.push(this.makeAgent(this.randomWalkGenes()));
    }
    return population;
  }

  buildHeuristicGenes() {
    const reachable = this.targets
      .filter((target) => target.x >= this.start.x)
      .slice()
      .sort((a, b) => a.x - b.x);
    if (reachable.length === 0) {
      return this.knotXs.map((_, index) => (index === 0 ? this.start.y : this.start.y));
    }

    const guides = [{ ...this.start }];
    for (const target of reachable) {
      const previous = guides[guides.length - 1];
      if (Math.abs(target.x - previous.x) < 1e-6) {
        // A mathematical function cannot visit two different y values at one x.
        // Never move the locked start; duplicate target guides can be averaged.
        if (guides.length > 1) previous.y = (previous.y + target.y) / 2;
      } else {
        guides.push({ x: target.x, y: target.y });
      }
    }

    return this.knotXs.map((x, index) => {
      if (index === 0) return this.start.y;
      return this.clampY(DotEvolution.interpolateGuides(guides, x));
    });
  }

  static interpolateGuides(guides, x) {
    if (x <= guides[0].x) return guides[0].y;
    if (x >= guides[guides.length - 1].x) return guides[guides.length - 1].y;
    for (let i = 0; i < guides.length - 1; i++) {
      const left = guides[i];
      const right = guides[i + 1];
      if (x >= left.x && x <= right.x) {
        const span = Math.max(1e-9, right.x - left.x);
        const t = (x - left.x) / span;
        return left.y + (right.y - left.y) * t;
      }
    }
    return guides[guides.length - 1].y;
  }

  randomWalkGenes() {
    const genes = [this.start.y];
    const stepScale = Math.max(1.4, (this.config.yMax - this.config.yMin) / 7);
    for (let i = 1; i < this.knotXs.length; i++) {
      const randomRestart = Math.random() < 0.14;
      const value = randomRestart
        ? this.randomBetween(this.config.yMin, this.config.yMax)
        : genes[i - 1] + this.gaussian() * stepScale;
      genes.push(this.clampY(value));
    }
    return genes;
  }

  makeAgent(genes) {
    const locked = genes.slice(0, this.knotXs.length);
    locked[0] = this.start.y;
    for (let i = 1; i < locked.length; i++) locked[i] = this.clampY(locked[i]);
    const path = this.knotXs.map((x, index) => ({ x, y: locked[index] }));
    return { genes: locked, path, fitness: null };
  }

  evaluatePopulation() {
    for (const agent of this.population) {
      agent.fitness = this.evaluateAgent(agent);
    }
    this.population.sort((a, b) => this.compareAgents(a, b));
    this.best = this.cloneAgent(this.population[0]);

    if (!this.bestEver || this.compareAgents(this.best, this.bestEver) < 0) {
      this.bestEver = this.cloneAgent(this.best);
      this.stagnantGenerations = 0;
    } else {
      this.stagnantGenerations += 1;
    }
  }

  evaluateAgent(agent) {
    // Empty in v1. A future lethal-zone evaluator can return a positive
    // violation count/penalty here and automatically outrank unsafe paths.
    const constraintPenalty = this.config.constraintEvaluators.reduce(
      (sum, evaluator) => sum + Math.max(0, Number(evaluator(agent.path)) || 0),
      0
    );
    const targetDistances = this.targets.map((target) =>
      DotEvolution.pointPolylineDistance(target, agent.path)
    );
    let hits = 0;
    let missDistance = 0;
    for (const distance of targetDistances) {
      if (distance <= this.config.targetRadius) {
        hits += 1;
      } else {
        missDistance += distance - this.config.targetRadius;
      }
    }

    let roughness = 0;
    let pathLength = 0;
    let previousSlope = null;
    for (let i = 0; i < agent.path.length - 1; i++) {
      const left = agent.path[i];
      const right = agent.path[i + 1];
      const dx = Math.max(1e-9, right.x - left.x);
      const dy = right.y - left.y;
      const slope = dy / dx;
      pathLength += Math.hypot(dx, dy);
      if (previousSlope !== null) {
        const bend = slope - previousSlope;
        roughness += bend * bend;
      }
      previousSlope = slope;
    }
    roughness /= Math.max(1, agent.path.length - 2);

    const horizontalSpan = Math.max(
      1e-9,
      agent.path[agent.path.length - 1].x - agent.path[0].x
    );
    const excessLength = Math.max(0, pathLength / horizontalSpan - 1);
    const edgePenalty = this.evaluateEdgePenalty(agent.path);

    return {
      constraintPenalty,
      hits,
      missDistance,
      edgePenalty,
      roughness,
      excessLength,
      targetDistances,
      // Display/debug value only. Selection uses compareAgents() below.
      score:
        hits * 1_000_000 -
        constraintPenalty * 10_000_000 -
        missDistance * 1_000 -
        edgePenalty * 25 -
        roughness * 2 -
        excessLength,
    };
  }

  compareAgents(left, right) {
    const a = left.fitness;
    const b = right.fitness;
    if (Math.abs(a.constraintPenalty - b.constraintPenalty) > 1e-9) {
      return a.constraintPenalty - b.constraintPenalty;
    }
    if (a.hits !== b.hits) return b.hits - a.hits;
    if (Math.abs(a.missDistance - b.missDistance) > 1e-9) {
      return a.missDistance - b.missDistance;
    }
    if (Math.abs(a.edgePenalty - b.edgePenalty) > 1e-9) {
      return a.edgePenalty - b.edgePenalty;
    }
    const aShape = a.roughness + a.excessLength * 0.08;
    const bShape = b.roughness + b.excessLength * 0.08;
    return aShape - bShape;
  }

  evaluateEdgePenalty(path) {
    const offset = this.config.edgeOffset;
    if (offset <= 0 || path.length < 2) return 0;

    let outsideSamples = 0;
    let samples = 0;
    const samplesPerSegment = 4;
    const lowerLine = this.config.yMin + offset;
    const upperLine = this.config.yMax - offset;
    for (let index = 0; index < path.length - 1; index++) {
      const left = path[index];
      const right = path[index + 1];
      for (let sample = 0; sample < samplesPerSegment; sample++) {
        const t = sample / samplesPerSegment;
        const y = left.y + (right.y - left.y) * t;
        if (y < lowerLine || y > upperLine) outsideSamples += 1;
        samples += 1;
      }
    }

    // Include the final point, which is not sampled by the half-open loops.
    const finalY = path[path.length - 1].y;
    if (finalY < lowerLine || finalY > upperLine) outsideSamples += 1;
    samples += 1;
    return outsideSamples / samples;
  }

  evolve() {
    const next = [];
    const eliteCount = Math.max(
      1,
      Math.floor(this.config.populationSize * this.config.eliteFraction)
    );
    for (let i = 0; i < eliteCount; i++) {
      next.push(this.makeAgent(this.population[i].genes));
    }

    while (next.length < this.config.populationSize) {
      if (this.stagnantGenerations >= 18 && Math.random() < 0.28) {
        next.push(this.makeAgent(this.randomWalkGenes()));
        continue;
      }
      const parentA = this.selectParent();
      const parentB = this.selectParent();
      const childGenes = this.crossover(parentA.genes, parentB.genes);
      this.mutate(childGenes);
      next.push(this.makeAgent(childGenes));
    }

    this.population = next;
    this.generation += 1;
    this.evaluatePopulation();
    return this.best;
  }

  selectParent() {
    let winner = null;
    const tournamentSize = 4;
    for (let i = 0; i < tournamentSize; i++) {
      const candidate =
        this.population[Math.floor(Math.random() * this.population.length)];
      if (!winner || this.compareAgents(candidate, winner) < 0) winner = candidate;
    }
    return winner;
  }

  crossover(a, b) {
    const child = [this.start.y];
    for (let i = 1; i < a.length; i++) {
      const blend = this.randomBetween(-0.12, 1.12);
      child.push(this.clampY(a[i] * blend + b[i] * (1 - blend)));
    }
    return child;
  }

  mutate(genes) {
    const cooling = Math.max(0.2, Math.exp(-this.generation / 110));
    const sigma = this.config.mutationScale * cooling;
    for (let i = 1; i < genes.length; i++) {
      if (Math.random() < 0.34) {
        const burst = Math.random() < 0.045 ? 3.2 : 1;
        genes[i] = this.clampY(genes[i] + this.gaussian() * sigma * burst);
      }
    }
    genes[0] = this.start.y;
  }

  cloneAgent(agent) {
    return {
      genes: agent.genes.slice(),
      path: agent.path.map((point) => ({ ...point })),
      fitness: {
        ...agent.fitness,
        targetDistances: agent.fitness.targetDistances.slice(),
      },
    };
  }

  clampY(value) {
    return Math.min(this.config.yMax, Math.max(this.config.yMin, value));
  }

  randomBetween(min, max) {
    return min + Math.random() * (max - min);
  }

  gaussian() {
    const u = Math.max(Number.EPSILON, Math.random());
    const v = Math.max(Number.EPSILON, Math.random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  static pointPolylineDistance(point, path) {
    let best = Infinity;
    for (let i = 0; i < path.length - 1; i++) {
      best = Math.min(best, DotEvolution.pointSegmentDistance(point, path[i], path[i + 1]));
    }
    return best;
  }

  static pointSegmentDistance(point, start, end) {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared < 1e-12) return Math.hypot(point.x - start.x, point.y - start.y);
    const projection = Math.min(
      1,
      Math.max(
        0,
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared
      )
    );
    const closestX = start.x + projection * dx;
    const closestY = start.y + projection * dy;
    return Math.hypot(point.x - closestX, point.y - closestY);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { DotEvolution };
}
