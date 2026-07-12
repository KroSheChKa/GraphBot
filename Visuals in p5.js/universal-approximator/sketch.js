// Universal Approximator — uniform x0 + step heights

const X_MIN = -25;
const X_MAX = 25;
const Y_MIN = -15;
const Y_MAX = 15;
const ASPECT = 5 / 3;
const MERGE_X_EPS = 0.05;
const CURVE_STEP = 0.15;

const CANVAS_WIDTH = 1000;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const DEFAULT_PARAMS = {
  sampleStep: 0.5,
  sigmoidK: 100,
  numNeurons: 50,
  trainEpochs: 500,
  trainLr: 0.02,
  showNeurons: true,
  stepHeights: true,
  freezeX0: true,
};

const COLORS = {
  background: [28, 28, 32],
  gridMinor: [55, 55, 62, 80],
  gridMajor: [75, 75, 85, 120],
  axis: [140, 140, 150],
  axisLabel: [160, 160, 170],
  curve: [255, 90, 90],
  curveGlow: [255, 90, 90, 40],
  sample: [100, 180, 255],
  approx: [80, 255, 140],
  approxGlow: [80, 255, 140, 35],
  panelText: [210, 210, 220],
  neuronLine: [255, 200, 80, 90],
};

let params = { ...DEFAULT_PARAMS };
let drawnPoints = [];
let trainingData = [];
let network = null;
let approxXMin = null;
let approxXMax = null;
let isDrawing = false;
let controlsEl;
let statusMseEl;

function setup() {
  const cnv = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
  document.getElementById("canvas-container").appendChild(cnv.elt);
  controlsEl = document.getElementById("controls-panel");
  buildControlsPanel();
}

function draw() {
  background(...COLORS.background);
  drawGrid();
  drawAxes();
  drawCurve();
  drawSamplePoints();
  drawApproximation();
  drawNetworkOverlay();
}

function buildControlsPanel() {
  controlsEl.innerHTML = `
    <h2>Параметры</h2>
    ${controlSlider("sampleStep", "Шаг датасета", 0.1, 2, 0.05, params.sampleStep, 2)}
    ${controlSlider("sigmoidK", "k (крутизна σ)", 1, 300, 1, params.sigmoidK, 0)}
    ${controlSlider("numNeurons", "Нейронов (= ступенек)", 5, 150, 1, params.numNeurons, 0)}
    ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
    ${controlSlider("trainLr", "Learning rate", 0.01, 0.2, 0.01, params.trainLr, 2)}
    ${controlCheckbox("stepHeights", "Высота w из линии", params.stepHeights)}
    ${controlCheckbox("freezeX0", "Фиксировать x₀ (равном. шаг)", params.freezeX0)}
    ${controlCheckbox("showNeurons", "Показать линии x₀", params.showNeurons)}
    <button id="btn-retrain" type="button">Переобучить</button>
    <button id="btn-reset" type="button" class="secondary">Сброс</button>
    <p id="status-mse" class="status">MSE: —</p>
    <p class="note">x₀ равномерно по x. w[i] = скачок высоты линии на x₀[i]. k — резкость скачка.</p>
    <p class="legend">
      <span class="swatch target"></span> цель
      <span class="swatch approx"></span> сеть
      <span class="swatch sample"></span> обучение
    </p>
  `;

  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    input.addEventListener("input", onParamInput);
    input.addEventListener("change", onParamChange);
  });

  document.getElementById("btn-retrain").addEventListener("click", rerunPipeline);
  document.getElementById("btn-reset").addEventListener("click", resetParams);
  statusMseEl = document.getElementById("status-mse");
  syncSliderLabels();
}

function controlSlider(key, label, min, max, step, value, decimals) {
  return `
    <label class="control">
      <div class="control-head">
        <span>${label}</span>
        <span class="control-value" data-value-for="${key}">${formatParam(key, value, decimals)}</span>
      </div>
      <input
        type="range"
        data-param="${key}"
        data-decimals="${decimals}"
        min="${min}"
        max="${max}"
        step="${step}"
        value="${value}"
      />
    </label>
  `;
}

function controlCheckbox(key, label, checked) {
  return `
    <label class="control checkbox">
      <input type="checkbox" data-param="${key}" ${checked ? "checked" : ""} />
      <span>${label}</span>
    </label>
  `;
}

function formatParam(key, value, decimals) {
  if (key === "numNeurons" || key === "trainEpochs" || key === "sigmoidK") {
    return String(Math.round(value));
  }
  return Number(value).toFixed(decimals ?? 2);
}

function syncSliderLabels() {
  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    if (input.type === "checkbox") return;
    const key = input.dataset.param;
    const decimals = parseInt(input.dataset.decimals || "2", 10);
    const label = controlsEl.querySelector(`[data-value-for="${key}"]`);
    if (label) label.textContent = formatParam(key, parseFloat(input.value), decimals);
  });
}

function readParamsFromUI() {
  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    if (input.type === "checkbox") {
      params[key] = input.checked;
    } else {
      params[key] = parseFloat(input.value);
    }
  });
}

function onParamInput(event) {
  const input = event.target;
  const key = input.dataset.param;
  const decimals = parseInt(input.dataset.decimals || "2", 10);
  const label = controlsEl.querySelector(`[data-value-for="${key}"]`);
  if (label) label.textContent = formatParam(key, parseFloat(input.value), decimals);
}

function onParamChange() {
  readParamsFromUI();
  if (drawnPoints.length > 0) rerunPipeline();
}

function resetParams() {
  params = { ...DEFAULT_PARAMS };
  buildControlsPanel();
  if (drawnPoints.length > 0) rerunPipeline();
}

function drawGrid() {
  strokeWeight(1);
  for (let x = X_MIN; x <= X_MAX; x++) {
    stroke(...(x % 5 === 0 ? COLORS.gridMajor : COLORS.gridMinor));
    lineWorld(x, Y_MIN, x, Y_MAX);
  }
  for (let y = Y_MIN; y <= Y_MAX; y++) {
    stroke(...(y % 5 === 0 ? COLORS.gridMajor : COLORS.gridMinor));
    lineWorld(X_MIN, y, X_MAX, y);
  }
}

function drawAxes() {
  stroke(...COLORS.axis);
  strokeWeight(1.5);
  lineWorld(X_MIN, 0, X_MAX, 0);
  lineWorld(0, Y_MIN, 0, Y_MAX);

  noStroke();
  fill(...COLORS.axisLabel);
  textSize(11);
  textAlign(CENTER, TOP);
  text("x", worldToScreen(X_MAX, 0).x + 14, worldToScreen(X_MAX, 0).y + 4);
  textAlign(RIGHT, CENTER);
  text("y", worldToScreen(0, Y_MAX).x - 6, worldToScreen(0, Y_MAX).y - 10);

  textSize(10);
  textAlign(CENTER, TOP);
  for (let x = -25; x <= 25; x += 5) {
    if (x === 0) continue;
    const p = worldToScreen(x, 0);
    text(String(x), p.x, p.y + 4);
  }
  textAlign(RIGHT, CENTER);
  for (let y = -15; y <= 15; y += 5) {
    if (y === 0) continue;
    const p = worldToScreen(0, y);
    text(String(y), p.x - 6, p.y);
  }
}

function drawCurve() {
  if (drawnPoints.length < 2) return;
  noFill();
  stroke(...COLORS.curveGlow);
  strokeWeight(6);
  beginShape();
  for (const pt of drawnPoints) {
    const s = worldToScreen(pt.x, pt.y);
    vertex(s.x, s.y);
  }
  endShape();
  stroke(...COLORS.curve);
  strokeWeight(2.5);
  beginShape();
  for (const pt of drawnPoints) {
    const s = worldToScreen(pt.x, pt.y);
    vertex(s.x, s.y);
  }
  endShape();
}

function drawSamplePoints() {
  if (trainingData.length === 0) return;
  noStroke();
  fill(...COLORS.sample);
  for (const pt of trainingData) {
    const s = worldToScreen(pt.x, pt.y);
    circle(s.x, s.y, 5);
  }
}

function drawApproximation() {
  if (!network || approxXMin === null || approxXMax === null) return;
  noFill();
  stroke(...COLORS.approxGlow);
  strokeWeight(5);
  beginShape();
  for (let x = approxXMin; x <= approxXMax + 1e-9; x += CURVE_STEP) {
    const y = constrain(network.predict(x), Y_MIN, Y_MAX);
    const s = worldToScreen(x, y);
    vertex(s.x, s.y);
  }
  endShape();
  stroke(...COLORS.approx);
  strokeWeight(2.5);
  beginShape();
  for (let x = approxXMin; x <= approxXMax + 1e-9; x += CURVE_STEP) {
    const y = constrain(network.predict(x), Y_MIN, Y_MAX);
    const s = worldToScreen(x, y);
    vertex(s.x, s.y);
  }
  endShape();
}

function drawNetworkOverlay() {
  if (!network) return;
  noStroke();
  fill(...COLORS.panelText);
  textSize(12);
  textAlign(LEFT, TOP);
  const stepText =
    network.x0Step !== null ? `  Δx₀=${roundCoord(network.x0Step)}` : "";
  text(`k = ${params.sigmoidK}${stepText}`, 12, 12);
  fill(...COLORS.approx);
  text(`MSE = ${formatMse(network.mse)}`, 12, 28);
  if (statusMseEl) {
    statusMseEl.textContent = `MSE: ${formatMse(network.mse)}  |  точек: ${trainingData.length}  |  Δx₀: ${roundCoord(network.x0Step)}`;
  }
  if (!params.showNeurons) return;
  stroke(...COLORS.neuronLine);
  strokeWeight(1);
  drawingContext.setLineDash([4, 6]);
  for (const x0 of network.x0) {
    lineWorld(x0, Y_MIN, x0, Y_MAX);
  }
  drawingContext.setLineDash([]);
}

function mergeByX(points) {
  if (points.length === 0) return [];
  const sorted = points.map((p) => ({ x: p.x, y: p.y })).sort((a, b) => a.x - b.x);
  const merged = [{ x: sorted[0].x, y: sorted[0].y, n: 1 }];

  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1];
    if (Math.abs(sorted[i].x - last.x) < MERGE_X_EPS) {
      last.y = (last.y * last.n + sorted[i].y) / (last.n + 1);
      last.n += 1;
    } else {
      merged.push({ x: sorted[i].x, y: sorted[i].y, n: 1 });
    }
  }

  return merged.map(({ x, y }) => ({ x, y }));
}

function interpolateY(merged, x) {
  if (merged.length === 0) return 0;
  if (x <= merged[0].x) return merged[0].y;
  if (x >= merged[merged.length - 1].x) return merged[merged.length - 1].y;
  for (let i = 0; i < merged.length - 1; i++) {
    const a = merged[i];
    const b = merged[i + 1];
    if (x >= a.x && x <= b.x) {
      if (Math.abs(b.x - a.x) < 1e-9) return (a.y + b.y) / 2;
      const t = (x - a.x) / (b.x - a.x);
      return a.y + t * (b.y - a.y);
    }
  }
  return merged[merged.length - 1].y;
}

function resampleUniform(merged, step) {
  if (merged.length === 0) return [];
  if (merged.length === 1) {
    return [{ x: merged[0].x, y: merged[0].y }];
  }

  const xStart = merged[0].x;
  const xEnd = merged[merged.length - 1].x;
  const samples = [];

  for (let x = xStart; x <= xEnd + 1e-9; x += step) {
    samples.push({ x, y: interpolateY(merged, x) });
  }

  if (samples[samples.length - 1].x !== xEnd) {
    samples.push({ x: xEnd, y: interpolateY(merged, xEnd) });
  }

  return samples;
}

function buildTrainingData(points) {
  return resampleUniform(mergeByX(points), params.sampleStep);
}

function trainNetwork() {
  if (trainingData.length < 2) {
    network = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
    return;
  }

  approxXMin = trainingData[0].x;
  approxXMax = trainingData[trainingData.length - 1].x;

  network = new SigmoidNetwork(params.numNeurons, params.sigmoidK);
  network.initFromData(trainingData, approxXMin, approxXMax, {
    stepHeights: params.stepHeights,
  });
  network.train(
    trainingData,
    approxXMin,
    approxXMax,
    params.trainEpochs,
    params.trainLr,
    params.freezeX0
  );

  console.log(
    `MSE = ${formatMse(network.mse)}  |  Δx₀ = ${roundCoord(network.x0Step)}  |  нейронов: ${params.numNeurons}`
  );
  console.log("Итоговая функция:");
  console.log(network.toFormulaText());
  console.log("σ(z) = 1 / (1 + exp(-z))");
}

function rerunPipeline() {
  readParamsFromUI();
  if (drawnPoints.length === 0) return;
  trainingData = buildTrainingData(drawnPoints);
  trainNetwork();
}

function finishDrawing() {
  if (drawnPoints.length === 0) return;
  readParamsFromUI();
  trainingData = buildTrainingData(drawnPoints);
  trainNetwork();
}

function formatMse(value) {
  return value !== null && Number.isFinite(value) ? roundCoord(value) : "—";
}

function roundCoord(value) {
  return Math.round(value * 1000) / 1000;
}

function worldToScreen(wx, wy) {
  return createVector(
    map(wx, X_MIN, X_MAX, 0, width),
    map(wy, Y_MAX, Y_MIN, 0, height)
  );
}

function screenToWorld(sx, sy) {
  return createVector(
    map(sx, 0, width, X_MIN, X_MAX),
    map(sy, 0, height, Y_MAX, Y_MIN)
  );
}

function lineWorld(x1, y1, x2, y2) {
  const a = worldToScreen(x1, y1);
  const b = worldToScreen(x2, y2);
  line(a.x, a.y, b.x, b.y);
}

function mousePressed() {
  if (!mouseInsideCanvas()) return;
  isDrawing = true;
  drawnPoints = [];
  trainingData = [];
  network = null;
  approxXMin = null;
  approxXMax = null;
  if (statusMseEl) statusMseEl.textContent = "MSE: —";
  addPointAtMouse();
}

function mouseDragged() {
  if (!isDrawing || !mouseInsideCanvas()) return;
  addPointAtMouse();
}

function mouseReleased() {
  isDrawing = false;
  if (drawnPoints.length > 0) finishDrawing();
}

function addPointAtMouse() {
  const w = screenToWorld(mouseX, mouseY);
  w.x = constrain(w.x, X_MIN, X_MAX);
  w.y = constrain(w.y, Y_MIN, Y_MAX);
  if (drawnPoints.length > 0) {
    const prev = drawnPoints[drawnPoints.length - 1];
    const dx = w.x - prev.x;
    const dy = w.y - prev.y;
    if (dx * dx + dy * dy < 0.02) return;
  }
  drawnPoints.push(w.copy());
}

function keyPressed() {
  if (key === "c" || key === "C" || key === "с" || key === "С") {
    drawnPoints = [];
    trainingData = [];
    network = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
  }
}

function mouseInsideCanvas() {
  return mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height;
}
