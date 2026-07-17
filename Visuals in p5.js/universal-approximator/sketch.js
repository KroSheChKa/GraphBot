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
  approxMethod: "sigmoid",
  sampleStep: 0.5,
  sigmoidK: 100,
  numNeurons: 50,
  taylorOrder: 8,
  taylorHiddenLayers: 2,
  taylorHiddenSize: 16,
  fourierHarmonics: 8,
  fourierHiddenLayers: 2,
  fourierHiddenSize: 16,
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
let mergedWaypoints = [];
let linearWaypoints = [];
let trainingData = [];
let network = null;
let linearMse = null;
let sigmoidMse = null;
let taylorMse = null;
let fourierMse = null;
let approxXMin = null;
let approxXMax = null;
let isDrawing = false;
let controlsEl;
let statusMseEl;
let copyBtnEl;
let captureBtnEl;
let bgImage = null;
let isCapturing = false;

function setup() {
  const cnv = createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
  document.getElementById("canvas-container").appendChild(cnv.elt);
  controlsEl = document.getElementById("controls-panel");
  buildControlsPanel();
}

function draw() {
  if (bgImage) {
    image(bgImage, 0, 0, width, height);
  } else {
    background(...COLORS.background);
  }
  drawGrid();
  drawAxes();
  drawCurve();
  drawSamplePoints();
  drawApproximation();
  drawNetworkOverlay();
}

const FEATURE_MLP_ACTIVATION = "tanh";

function buildControlsPanel() {
  let methodControls = `
    <p class="note">Прямые между синими точками датасета. Число сегментов задаётся шагом датасета.</p>
  `;

  if (params.approxMethod === "sigmoid") {
    methodControls = `
    ${controlSlider("sigmoidK", "k (крутизна σ)", 1, 300, 1, params.sigmoidK, 0)}
    ${controlSlider("numNeurons", "Нейронов (= ступенек)", 5, 150, 1, params.numNeurons, 0)}
    ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
    ${controlSlider("trainLr", "Learning rate", 0.01, 0.2, 0.01, params.trainLr, 2)}
    ${controlCheckbox("stepHeights", "Высота w из линии", params.stepHeights)}
    ${controlCheckbox("freezeX0", "Фиксировать x₀ (равном. шаг)", params.freezeX0)}
    ${controlCheckbox("showNeurons", "Показать линии x₀", params.showNeurons)}
    <button id="btn-retrain" type="button">Переобучить</button>
    <p class="note">x₀ равномерно по x. w[i] = скачок высоты линии на x₀[i]. k — резкость скачка.</p>
  `;
  } else if (params.approxMethod === "taylor") {
    methodControls = `
    ${controlSlider("taylorOrder", "Порядок n (вход φ)", 1, 20, 1, params.taylorOrder, 0)}
    ${controlSlider("taylorHiddenLayers", "Скрытых слоёв", 0, 4, 1, params.taylorHiddenLayers, 0)}
    ${controlSlider("taylorHiddenSize", "Нейронов в слое", 2, 64, 1, params.taylorHiddenSize, 0)}
    ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
    ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
    <button id="btn-retrain" type="button">Переобучить</button>
    <p class="note">φ(t)=[1,t,…,tⁿ], t=(x−c)/s → MLP → y. 0 скрытых слоёв = чистый полином; &gt;0 = между φ и y стоит tanh.</p>
  `;
  } else if (params.approxMethod === "fourier") {
    methodControls = `
    ${controlSlider("fourierHarmonics", "Гармоник K", 1, 40, 1, params.fourierHarmonics, 0)}
    ${controlSlider("fourierHiddenLayers", "Скрытых слоёв", 0, 4, 1, params.fourierHiddenLayers, 0)}
    ${controlSlider("fourierHiddenSize", "Нейронов в слое", 2, 64, 1, params.fourierHiddenSize, 0)}
    ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
    ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
    <button id="btn-retrain" type="button">Переобучить</button>
    <p class="note">φ(t)=[1,cos(kπt),sin(kπt),…], π=3.1416, t=(x−c)/s → MLP → y. 0 слоёв = ряд Фурье.</p>
  `;
  }

  controlsEl.innerHTML = `
    <h2>Параметры</h2>
    <button id="btn-capture-field" type="button">Захват поля</button>
    ${controlMethodPicker()}
    ${controlSlider("sampleStep", "Шаг датасета", 0.1, 2, 0.05, params.sampleStep, 2)}
    ${methodControls}
    <button id="btn-copy-formula" type="button" disabled>Копировать y</button>
    <button id="btn-reset" type="button" class="secondary">Сброс</button>
    <p id="status-mse" class="status">MSE: —</p>
    <p class="legend">
      <span class="swatch target"></span> цель
      <span class="swatch approx"></span> аппроксимация
      <span class="swatch sample"></span> обучение
    </p>
  `;

  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    input.addEventListener("input", onParamInput);
    input.addEventListener("change", onParamChange);
  });

  const retrainBtn = document.getElementById("btn-retrain");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", () => {
      rerunPipeline();
      logActiveFormula();
    });
  }
  document.getElementById("btn-reset").addEventListener("click", resetParams);
  copyBtnEl = document.getElementById("btn-copy-formula");
  copyBtnEl.addEventListener("click", (event) => {
    event.preventDefault();
    copyActiveFormula();
  });
  captureBtnEl = document.getElementById("btn-capture-field");
  captureBtnEl.addEventListener("click", (event) => {
    event.preventDefault();
    captureGameField();
  });
  statusMseEl = document.getElementById("status-mse");
  syncSliderLabels();
  updateCopyButton();
  updateCaptureButton();
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

function controlSelect(key, label, options, value) {
  const opts = options
    .map(
      (opt) =>
        `<option value="${opt.value}" ${opt.value === value ? "selected" : ""}>${opt.label}</option>`
    )
    .join("");
  return `
    <label class="control">
      <div class="control-head"><span>${label}</span></div>
      <select data-param="${key}">${opts}</select>
    </label>
  `;
}

function controlMethodPicker() {
  return `
    <div class="control method-picker">
      <div class="control-head"><span>Метод аппроксимации</span></div>
      <div class="method-options">
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="linear" ${params.approxMethod === "linear" ? "checked" : ""} />
          <span>1. Линейный (сегменты)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="sigmoid" ${params.approxMethod === "sigmoid" ? "checked" : ""} />
          <span>2. Сигмоиды (сеть)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="taylor" ${params.approxMethod === "taylor" ? "checked" : ""} />
          <span>3. Тейлор (полином) (beta)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="fourier" ${params.approxMethod === "fourier" ? "checked" : ""} />
          <span>4. Фурье (гармоники)</span>
        </label>
      </div>
    </div>
  `;
}

function formatParam(key, value, decimals) {
  if (
    key === "numNeurons" ||
    key === "trainEpochs" ||
    key === "sigmoidK" ||
    key === "taylorOrder" ||
    key === "taylorHiddenLayers" ||
    key === "taylorHiddenSize" ||
    key === "fourierHarmonics" ||
    key === "fourierHiddenLayers" ||
    key === "fourierHiddenSize"
  ) {
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
    } else if (input.type === "radio") {
      if (input.checked) params[key] = input.value;
    } else if (input.tagName === "SELECT") {
      params[key] = input.value;
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

function onParamChange(event) {
  const prevMethod = params.approxMethod;
  readParamsFromUI();
  if (event?.target?.dataset?.param === "approxMethod" && params.approxMethod !== prevMethod) {
    buildControlsPanel();
    if (params.approxMethod === "sigmoid" && trainingData.length >= 2) {
      trainSigmoidNetwork();
    } else if (params.approxMethod === "taylor" && trainingData.length >= 2) {
      trainTaylorNetwork();
    } else if (params.approxMethod === "fourier" && trainingData.length >= 2) {
      trainFourierNetwork();
    } else if (params.approxMethod === "linear") {
      network = null;
    }
    updateStatusMse();
    updateCopyButton();
    logActiveFormula();
    return;
  }
  if (drawnPoints.length > 0) rerunPipeline();
}

function resetParams() {
  params = { ...DEFAULT_PARAMS };
  buildControlsPanel();
  if (drawnPoints.length > 0) rerunPipeline();
}

function drawGrid() {
  const onField = bgImage !== null;
  strokeWeight(1);
  for (let x = X_MIN; x <= X_MAX; x++) {
    const col = x % 5 === 0 ? COLORS.gridMajor : COLORS.gridMinor;
    stroke(col[0], col[1], col[2], onField ? col[3] * 0.45 : col[3]);
    lineWorld(x, Y_MIN, x, Y_MAX);
  }
  for (let y = Y_MIN; y <= Y_MAX; y++) {
    const col = y % 5 === 0 ? COLORS.gridMajor : COLORS.gridMinor;
    stroke(col[0], col[1], col[2], onField ? col[3] * 0.45 : col[3]);
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

function drawLinearSegments(weight, color) {
  stroke(...color);
  strokeWeight(weight);
  for (let i = 0; i < linearWaypoints.length - 1; i++) {
    const a = linearWaypoints[i];
    const b = linearWaypoints[i + 1];
    lineWorld(a.x, a.y, b.x, b.y);
  }
}

function usesFeatureNetwork() {
  return (
    params.approxMethod === "sigmoid" ||
    params.approxMethod === "taylor" ||
    params.approxMethod === "fourier"
  );
}

function drawApproximation() {
  if (approxXMin === null || approxXMax === null) return;
  if (usesFeatureNetwork() && !network) return;
  if (params.approxMethod === "linear" && linearWaypoints.length < 2) return;

  if (params.approxMethod === "linear") {
    drawLinearSegments(5, COLORS.approxGlow);
    drawLinearSegments(2.5, COLORS.approx);
    return;
  }

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
  if (trainingData.length === 0) return;

  noStroke();
  fill(...COLORS.panelText);
  textSize(12);
  textAlign(LEFT, TOP);

  const activeMse =
    params.approxMethod === "linear" ? linearMse : network ? network.mse : null;

  if (params.approxMethod === "sigmoid" && network) {
    const stepText =
      network.x0Step !== null ? `  Δx₀=${roundCoord(network.x0Step)}` : "";
    text(`k = ${params.sigmoidK}${stepText}`, 12, 12);
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 28);
  } else if (params.approxMethod === "taylor" && network) {
    const center = roundCoord(network.center);
    const scale = roundCoord(network.scale);
    text(`n=${params.taylorOrder}  L=${params.taylorHiddenLayers}×${params.taylorHiddenSize}`, 12, 12);
    text(`c=${center}  s=${scale}`, 12, 28);
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 44);
  } else if (params.approxMethod === "fourier" && network) {
    const center = roundCoord(network.center);
    const scale = roundCoord(network.scale);
    text(`K=${params.fourierHarmonics}  L=${params.fourierHiddenLayers}×${params.fourierHiddenSize}`, 12, 12);
    text(`c=${center}  s=${scale}`, 12, 28);
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 44);
  } else {
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 12);
    text(`сегментов: ${linearWaypoints.length - 1}`, 12, 28);
  }

  updateStatusMse();

  if (params.approxMethod !== "sigmoid" || !network || !params.showNeurons) return;
  stroke(...COLORS.neuronLine);
  strokeWeight(1);
  drawingContext.setLineDash([4, 6]);
  for (const x0 of network.x0) {
    lineWorld(x0, Y_MIN, x0, Y_MAX);
  }
  drawingContext.setLineDash([]);
}

function updateStatusMse() {
  if (!statusMseEl) return;
  if (trainingData.length === 0) {
    statusMseEl.textContent = "MSE: —";
    return;
  }

  const activeMethodLabel = methodLabel();
  const active =
    params.approxMethod === "linear"
      ? `активный: ${activeMethodLabel} ${formatMse(linearMse)}`
      : `активный: ${activeMethodLabel} ${formatMse(network?.mse)}`;
  const compare = `сравнение — линейный: ${formatMse(linearMse)}  |  сигмоиды: ${formatMse(sigmoidMse)}  |  Тейлор: ${formatMse(taylorMse)}  |  Фурье: ${formatMse(fourierMse)}`;
  const meta = `точек обучения: ${trainingData.length}  |  сегментов: ${Math.max(0, linearWaypoints.length - 1)}`;
  statusMseEl.textContent = `${active}  ||  ${compare}  ||  ${meta}`;
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

function evalDirectLineSegment(p1, p2, x) {
  const x1 = p1.x;
  const y1 = p1.y;
  const x2 = p2.x;
  const y2 = p2.y;
  let dx = x2 - x1;
  if (Math.abs(dx) < 1e-12) {
    dx = y1 !== y2 ? 1e-6 : 1e-6;
  }
  const dist = -((y1 - y2) / 2) / dx;
  return dist * (Math.abs(x - x1) - Math.abs(x - x2));
}

function evalLinearWaypoints(waypoints, x) {
  let y = 0;
  for (let i = 0; i < waypoints.length - 1; i++) {
    y += evalDirectLineSegment(waypoints[i], waypoints[i + 1], x);
  }
  return y;
}

function buildTrainingData(points) {
  mergedWaypoints = mergeByX(points);
  return resampleUniform(mergedWaypoints, params.sampleStep);
}

function syncLinearWaypointsFromDataset() {
  linearWaypoints = trainingData.map((pt) => ({ x: pt.x, y: pt.y }));
}

function computeMseOnData(predictFn, data) {
  if (data.length === 0) return null;
  let sum = 0;
  for (const pt of data) {
    const pred = predictFn(pt.x);
    if (!Number.isFinite(pred)) return null;
    const err = pred - pt.y;
    sum += err * err;
  }
  return sum / data.length;
}

function directLineFormula(p1, p2) {
  const x1 = roundCoord(p1.x);
  const y1 = roundCoord(p1.y);
  const x2 = roundCoord(p2.x);
  const y2 = roundCoord(p2.y);
  let dx = x2 - x1;
  if (Math.abs(dx) < 1e-12) {
    dx = y1 !== y2 ? 1e-6 : 1e-6;
  }
  const dist = roundCoord(-((y1 - y2) / 2) / dx);
  return `${dist}*(abs(x - ${x1}) - abs(x - ${x2}))`;
}

function linearFormulaText(waypoints) {
  if (waypoints.length < 2) return null;
  const parts = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    parts.push(directLineFormula(waypoints[i], waypoints[i + 1]));
  }
  return `y=${parts.join("+")}`;
}

function getActiveFormulaText() {
  if (trainingData.length < 2) return null;
  let formula = null;
  if (params.approxMethod === "linear") {
    formula = linearFormulaText(linearWaypoints);
  } else if (network) {
    formula = network.toDesmosText();
  }
  return formula ? normalizeFormula(formula) : null;
}

function methodLabel(method = params.approxMethod) {
  if (method === "linear") return "линейный";
  if (method === "taylor") return "Тейлор";
  if (method === "fourier") return "Фурье";
  return "сигмоиды";
}

function logActiveFormula() {
  const formula = getActiveFormulaText();
  if (!formula) {
    if ((params.approxMethod === "taylor" || params.approxMethod === "fourier") && network) {
      console.log(`[${methodLabel()}] MSE = ${formatMse(network.mse)}`);
      console.log(network.architectureText());
    }
    return;
  }
  const label = methodLabel();
  const mse =
    params.approxMethod === "linear" ? formatMse(linearMse) : formatMse(network?.mse);
  console.log(`[${label}] MSE = ${mse}`);
  console.log(formula);
}

async function copyActiveFormula() {
  const formula = getActiveFormulaText();
  if (!formula || !copyBtnEl) return;
  const prev = copyBtnEl.textContent;
  try {
    await navigator.clipboard.writeText(formula);
    copyBtnEl.textContent = "Скопировано";
  } catch {
    copyBtnEl.textContent = "Ошибка";
  }
  setTimeout(() => {
    if (copyBtnEl) copyBtnEl.textContent = prev;
  }, 1200);
}

function updateCopyButton() {
  if (!copyBtnEl) return;
  copyBtnEl.disabled = getActiveFormulaText() === null;
}

function updateCaptureButton() {
  if (!captureBtnEl) return;
  captureBtnEl.disabled = isCapturing;
  captureBtnEl.textContent = isCapturing ? "Захват..." : "Захват поля";
}

async function captureGameField() {
  if (isCapturing) return;
  isCapturing = true;
  updateCaptureButton();

  try {
    const response = await fetch("/api/capture", { method: "POST" });
    const data = await response.json();
    if (!data.ok) {
      alert(data.error || "Не удалось захватить поле");
      return;
    }

    await new Promise((resolve, reject) => {
      loadImage(
        data.image,
        (img) => {
          bgImage = img;
          resolve();
        },
        (err) => reject(err)
      );
    });
  } catch {
    alert(
      "Сервер захвата недоступен.\nЗапусти: python tools/approximator_server.py\nи открой http://127.0.0.1:8765/"
    );
  } finally {
    isCapturing = false;
    updateCaptureButton();
  }
}

function trainSigmoidNetwork() {
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
  sigmoidMse = network.mse;
}

function trainTaylorNetwork() {
  network = new TaylorNetwork(
    params.taylorOrder,
    params.taylorHiddenLayers,
    params.taylorHiddenSize,
    FEATURE_MLP_ACTIVATION
  );
  network.initFromData(trainingData, approxXMin, approxXMax);
  network.train(trainingData, params.trainEpochs, params.trainLr);
  taylorMse = network.mse;
}

function trainFourierNetwork() {
  network = new FourierNetwork(
    params.fourierHarmonics,
    params.fourierHiddenLayers,
    params.fourierHiddenSize,
    FEATURE_MLP_ACTIVATION
  );
  network.initFromData(trainingData, approxXMin, approxXMax);
  network.train(trainingData, params.trainEpochs, params.trainLr);
  fourierMse = network.mse;
}

function processPipeline({ logFormula = false } = {}) {
  if (drawnPoints.length === 0) return;

  trainingData = buildTrainingData(drawnPoints);
  if (trainingData.length < 2) {
    network = null;
    linearWaypoints = [];
    linearMse = null;
    sigmoidMse = null;
    taylorMse = null;
    fourierMse = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
    updateCopyButton();
    return;
  }

  syncLinearWaypointsFromDataset();
  approxXMin = trainingData[0].x;
  approxXMax = trainingData[trainingData.length - 1].x;
  linearMse = computeMseOnData((x) => evalLinearWaypoints(linearWaypoints, x), trainingData);

  if (params.approxMethod === "sigmoid") {
    trainSigmoidNetwork();
  } else if (params.approxMethod === "taylor") {
    trainTaylorNetwork();
  } else if (params.approxMethod === "fourier") {
    trainFourierNetwork();
  } else {
    network = null;
  }

  updateStatusMse();
  updateCopyButton();
  if (logFormula) logActiveFormula();
}

function rerunPipeline() {
  readParamsFromUI();
  processPipeline();
}

function finishDrawing() {
  if (drawnPoints.length === 0) return;
  readParamsFromUI();
  processPipeline({ logFormula: true });
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
  mergedWaypoints = [];
  linearWaypoints = [];
  trainingData = [];
  network = null;
  linearMse = null;
  sigmoidMse = null;
  taylorMse = null;
  fourierMse = null;
  approxXMin = null;
  approxXMax = null;
  if (statusMseEl) statusMseEl.textContent = "MSE: —";
  updateCopyButton();
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
    mergedWaypoints = [];
    linearWaypoints = [];
    trainingData = [];
    network = null;
    linearMse = null;
    sigmoidMse = null;
    taylorMse = null;
    fourierMse = null;
    approxXMin = null;
    approxXMax = null;
    if (statusMseEl) statusMseEl.textContent = "MSE: —";
    updateCopyButton();
  }
}

function mouseInsideCanvas() {
  return mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height;
}
