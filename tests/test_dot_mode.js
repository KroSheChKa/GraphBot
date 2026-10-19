const assert = require("assert");
const { DotEvolution } = require("../Visuals in p5.js/universal-approximator/dot-mode.js");

function testBasic() {
  const evo = new DotEvolution({ x: 0, y: 0 }, [{ x: 5, y: 2 }, { x: 10, y: -1 }]);
  assert.ok(evo.best, "best exists");
  assert.ok(evo.best.trajectory.length > 0);
  assert.ok(Number.isFinite(evo.best.fitness.hits));
  // evolve a few generations
  for (let i = 0; i < 5; i++) evo.evolve();
  assert.ok(evo.generation === 5);
  assert.ok(evo.bestEver, "bestEver exists");
  // fitness improves or at least non-null
  console.log("gen5 best hits:", evo.best.fitness.hits, "miss:", evo.best.fitness.missDistance);
}

function testUnreachableTargetsLeft() {
  const evo = new DotEvolution({ x: 5, y: 0 }, [{ x: 0, y: 2 }]);
  // targets left of start are filtered in heuristic; should still run
  assert.ok(evo.population.length > 0);
  for (let i = 0; i < 3; i++) evo.evolve();
  console.log("left-target best hits:", evo.best.fitness.hits);
}

function testBounds() {
  const evo = new DotEvolution({ x: -24, y: 0 }, [{ x: 24, y: 0 }], { yMin: -5, yMax: 5, xMax: 25 });
  for (const agent of evo.population) {
    for (const p of agent.trajectory) {
      assert.ok(p.y >= -5 - 1e-9 && p.y <= 5 + 1e-9, "y within bounds");
    }
  }
}

function testConstructorValidation() {
  // populationSize clamped
  const evo = new DotEvolution({ x: 0, y: 0 }, [{ x: 1, y: 1 }], { populationSize: 2, controlPoints: 2 });
  assert.ok(evo.config.populationSize >= 8);
  assert.ok(evo.config.controlPoints >= 4);
}

testBasic();
testUnreachableTargetsLeft();
testBounds();
testConstructorValidation();
console.log("All dot-mode tests passed");
