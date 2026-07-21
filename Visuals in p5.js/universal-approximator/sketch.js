// Universal Approximator — uniform x0 + step heights

const X_MIN = -25;
const X_MAX = 25;
const Y_MIN = -15;
const Y_MAX = 15;
const ASPECT = 5 / 3;
const MERGE_X_EPS = 0.05;
const CURVE_STEP = 0.15;
const GAME_PRECISION = 5;
const CLICK_LEFT_TOLERANCE = 0.08;
const VERTICAL_MAX_COEFF = 999;
const VERTICAL_MIN_EPS = 0.001;

const CANVAS_WIDTH = 1000;
const CANVAS_HEIGHT = CANVAS_WIDTH / ASPECT;

const DEFAULT_PARAMS = {
  inputMode: "click",
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
  click: [255, 176, 80],
  clickGlow: [255, 176, 80, 50],
  anchor: [180, 140, 255],
  anchorGlow: [180, 140, 255, 55],
  approx: [80, 255, 140],
  approxGlow: [80, 255, 140, 35],
  panelText: [210, 210, 220],
  neuronLine: [255, 200, 80, 90],
};

let params = { ...DEFAULT_PARAMS };
let clickPoints = [];
let clickWaypoints = [];
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
  cnv.elt.oncontextmenu = () => false;
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
  if (params.inputMode === "click") {
    drawClickMode();
  } else {
    drawCurve();
    drawSamplePoints();
    drawApproximation();
    drawNetworkOverlay();
  }
}

const FEATURE_MLP_ACTIVATION = "tanh";

function buildControlsPanel() {
  let drawMethodControls = "";

  if (params.inputMode === "draw") {
    if (params.approxMethod === "linear") {
      drawMethodControls = `
        <p class="note">Прямые между синими точками датасета. Число сегментов задаётся шагом датасета.</p>
      `;
    } else if (params.approxMethod === "sigmoid") {
      drawMethodControls = `
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
      drawMethodControls = `
      ${controlSlider("taylorOrder", "Порядок n (вход φ)", 1, 20, 1, params.taylorOrder, 0)}
      ${controlSlider("taylorHiddenLayers", "Скрытых слоёв", 0, 4, 1, params.taylorHiddenLayers, 0)}
      ${controlSlider("taylorHiddenSize", "Нейронов в слое", 2, 64, 1, params.taylorHiddenSize, 0)}
      ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
      ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
      <button id="btn-retrain" type="button">Переобучить</button>
      <p class="note">φ(t)=[1,t,…,tⁿ], t=(x−c)/s → MLP → y. 0 скрытых слоёв = чистый полином; &gt;0 = между φ и y стоит tanh.</p>
    `;
    } else if (params.approxMethod === "fourier") {
      drawMethodControls = `
      ${controlSlider("fourierHarmonics", "Гармоник K", 1, 40, 1, params.fourierHarmonics, 0)}
      ${controlSlider("fourierHiddenLayers", "Скрытых слоёв", 0, 4, 1, params.fourierHiddenLayers, 0)}
      ${controlSlider("fourierHiddenSize", "Нейронов в слое", 2, 64, 1, params.fourierHiddenSize, 0)}
      ${controlSlider("trainEpochs", "Эпох", 500, 10000, 100, params.trainEpochs, 0)}
      ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
      <button id="btn-retrain" type="button">Переобучить</button>
      <p class="note">φ(t)=[1,cos(kπt),sin(kπt),…], π=3.1416, t=(x−c)/s → MLP → y. 0 слоёв = ряд Фурье.</p>
    `;
    }
  }

  const modeSpecificControls =
    params.inputMode === "click"
      ? `
    <p class="note">1-й клик — <strong>активный солдат</strong> (фиолетовая <strong>A</strong>). Дальше — цели в порядке кликов. Клик <strong>левее</strong> предыдущей точки → вертикальный сегмент. Формула без <code>y=</code>. <strong>Ctrl+Z</strong> / ПКМ / Backspace — отменить точку.</p>
    <button id="btn-undo-click" type="button" class="secondary">Отменить последний клик</button>
  `
      : `
    <div class="draw-mode-section">
      ${controlDrawMethodPicker()}
      ${controlSlider("sampleStep", "Шаг датасета", 0.1, 2, 0.05, params.sampleStep, 2)}
      ${drawMethodControls}
    </div>
  `;

  const legendHtml =
    params.inputMode === "click"
      ? `
    <p class="legend">
      <span class="swatch anchor"></span> активный (A)
      <span class="swatch click"></span> цели
      <span class="swatch approx"></span> сегменты
    </p>
  `
      : `
    <p class="legend">
      <span class="swatch target"></span> цель
      <span class="swatch approx"></span> аппроксимация
      <span class="swatch sample"></span> обучение
    </p>
  `;

  controlsEl.innerHTML = `
    <h2>Параметры</h2>
    <button id="btn-capture-field" type="button">Захват поля</button>
    ${controlInputModePicker()}
    ${modeSpecificControls}
    <button id="btn-copy-formula" type="button" disabled>Копировать y</button>
    <button id="btn-reset" type="button" class="secondary">Сброс</button>
    <p id="status-mse" class="status">—</p>
    ${legendHtml}
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
  const undoClickBtn = document.getElementById("btn-undo-click");
  if (undoClickBtn) {
    undoClickBtn.addEventListener("click", () => undoLastClick());
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
  updateStatusPanel();
}

function controlInputModePicker() {
  return `
    <div class="control mode-picker">
      <div class="control-head"><span>Режим</span></div>
      <div class="mode-options">
        <label class="mode-option">
          <input type="radio" name="inputMode" data-param="inputMode" value="click" ${params.inputMode === "click" ? "checked" : ""} />
          <span>1. Click mode</span>
        </label>
        <label class="mode-option">
          <input type="radio" name="inputMode" data-param="inputMode" value="draw" ${params.inputMode === "draw" ? "checked" : ""} />
          <span>2. Draw mode</span>
        </label>
      </div>
    </div>
  `;
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

function controlDrawMethodPicker() {
  return `
    <div class="control method-picker nested">
      <div class="control-head"><span>Метод аппроксимации</span></div>
      <div class="method-options">
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="linear" ${params.approxMethod === "linear" ? "checked" : ""} />
          <span>2.1 Линейный (сегменты)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="sigmoid" ${params.approxMethod === "sigmoid" ? "checked" : ""} />
          <span>2.2 Сигмоиды (сеть)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="taylor" ${params.approxMethod === "taylor" ? "checked" : ""} />
          <span>2.3 Тейлор (полином) (beta)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="fourier" ${params.approxMethod === "fourier" ? "checked" : ""} />
          <span>2.4 Фурье (гармоники)</span>
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
  const prevInputMode = params.inputMode;
  const prevMethod = params.approxMethod;
  readParamsFromUI();

  if (event?.target?.dataset?.param === "inputMode" && params.inputMode !== prevInputMode) {
    clearWorkspaceState();
    buildControlsPanel();
    return;
  }

  if (params.inputMode !== "draw") return;

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
    updateStatusPanel();
    updateCopyButton();
    logActiveFormula();
    return;
  }
  if (drawnPoints.length > 0) rerunPipeline();
}

function resetParams() {
  params = { ...DEFAULT_PARAMS };
  clearWorkspaceState();
  buildControlsPanel();
  if (params.inputMode === "draw" && drawnPoints.length > 0) rerunPipeline();
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

function drawClickMode() {
  if (clickWaypoints.length >= 2) {
    const prevLinear = linearWaypoints;
    linearWaypoints = clickWaypoints;
    drawLinearSegments(5, COLORS.approxGlow);
    drawLinearSegments(2.5, COLORS.approx);
    linearWaypoints = prevLinear;
  }

  if (clickPoints.length === 0) {
    drawClickModeHint();
    return;
  }

  noStroke();
  textSize(11);
  textAlign(CENTER, CENTER);
  for (let i = 0; i < clickPoints.length; i++) {
    const pt = clickPoints[i];
    const s = worldToScreen(pt.x, pt.y);
    const isAnchor = i === 0;
    fill(...(isAnchor ? COLORS.anchorGlow : COLORS.clickGlow));
    circle(s.x, s.y, isAnchor ? 20 : 18);
    fill(...(isAnchor ? COLORS.anchor : COLORS.click));
    circle(s.x, s.y, isAnchor ? 11 : 10);
    fill(20, 20, 24);
    text(isAnchor ? "A" : String(i), s.x, s.y);
  }
}

function drawClickModeHint() {
  noStroke();
  fill(...COLORS.panelText);
  textSize(13);
  textAlign(CENTER, TOP);
  const cx = width / 2;
  text("1-й клик — активный солдат (A)", cx, 14);
  textSize(11);
  fill(...COLORS.axisLabel);
  text("Дальше кликай цели; клик левее → вертикальный сегмент", cx, 34);
}

function drawNetworkOverlay() {
  if (params.inputMode !== "draw" || trainingData.length === 0) return;

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

  updateStatusPanel();

  if (params.approxMethod !== "sigmoid" || !network || !params.showNeurons) return;
  stroke(...COLORS.neuronLine);
  strokeWeight(1);
  drawingContext.setLineDash([4, 6]);
  for (const x0 of network.x0) {
    lineWorld(x0, Y_MIN, x0, Y_MAX);
  }
  drawingContext.setLineDash([]);
}

function updateStatusPanel() {
  if (!statusMseEl) return;

  if (params.inputMode === "click") {
    const segments = Math.max(0, clickWaypoints.length - 1);
    const targets = Math.max(0, clickPoints.length - 1);
    statusMseEl.textContent = `активный (A): ${clickPoints.length > 0 ? "да" : "нет"}  |  целей: ${targets}  |  сегментов: ${segments}`;
    return;
  }

  updateStatusMse();
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

function fmtGame(value) {
  const factor = 10 ** GAME_PRECISION;
  return Math.round(Number(value) * factor) / factor;
}

function verticalEps(yFrom, yTo, maxCoeff = VERTICAL_MAX_COEFF) {
  const dy = Math.abs(yTo - yFrom);
  if (dy < 1e-9) return VERTICAL_MIN_EPS;
  return Math.max(VERTICAL_MIN_EPS, dy / (2 * maxCoeff));
}

function buildClickFormulaWaypoints() {
  if (clickPoints.length < 2) return [];

  const waypoints = [];
  for (const pt of clickPoints) {
    const gameX = fmtGame(pt.x);
    const gameY = fmtGame(pt.y);

    if (waypoints.length === 0) {
      waypoints.push({ x: gameX, y: gameY });
      continue;
    }

    const prev = waypoints[waypoints.length - 1];
    if (gameX < prev.x - CLICK_LEFT_TOLERANCE) {
      if (Math.abs(gameY - prev.y) < 1e-6) continue;
      const eps = verticalEps(prev.y, gameY);
      waypoints.push({ x: fmtGame(prev.x + eps), y: gameY });
    } else {
      waypoints.push({ x: gameX, y: gameY });
    }
  }

  return waypoints.length >= 2 ? waypoints : [];
}

function syncClickWaypoints() {
  clickWaypoints = buildClickFormulaWaypoints();
}

function addClickPointAtMouse() {
  const w = screenToWorld(mouseX, mouseY);
  w.x = constrain(w.x, X_MIN, X_MAX);
  w.y = constrain(w.y, Y_MIN, Y_MAX);
  clickPoints.push({ x: w.x, y: w.y });
  syncClickWaypoints();
  updateStatusPanel();
  updateCopyButton();
  logActiveFormula();
}

function undoLastClick() {
  if (clickPoints.length === 0) return;
  clickPoints.pop();
  syncClickWaypoints();
  updateStatusPanel();
  updateCopyButton();
  logActiveFormula();
}

function clearWorkspaceState() {
  clickPoints = [];
  clickWaypoints = [];
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
  isDrawing = false;
  updateStatusPanel();
  updateCopyButton();
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

function directLineFormula(p1, p2, useGamePrecision = false) {
  const round = useGamePrecision ? fmtGame : roundCoord;
  const x1 = round(p1.x);
  const y1 = round(p1.y);
  const x2 = round(p2.x);
  const y2 = round(p2.y);
  let dx = x2 - x1;
  if (Math.abs(dx) < 1e-12) {
    dx = y1 !== y2 ? verticalEps(y1, y2) : VERTICAL_MIN_EPS;
  }
  const dist = round(-((y1 - y2) / 2) / dx);
  return `${dist}*(abs(x - ${x1}) - abs(x - ${x2}))`;
}

function waypointsFormulaText(waypoints, useGamePrecision = false, includeYPrefix = true) {
  if (waypoints.length < 2) return null;
  const parts = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    parts.push(directLineFormula(waypoints[i], waypoints[i + 1], useGamePrecision));
  }
  const body = normalizeFormula(parts.join(" + "));
  return includeYPrefix ? `y=${body.replace(/^y=/, "")}` : body;
}

function linearFormulaText(waypoints, useGamePrecision = false) {
  return waypointsFormulaText(waypoints, useGamePrecision, true);
}

function clickFormulaText(waypoints) {
  return waypointsFormulaText(waypoints, true, false);
}

function getActiveFormulaText() {
  if (params.inputMode === "click") {
    if (clickWaypoints.length < 2) return null;
    return clickFormulaText(clickWaypoints);
  }

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
    if (
      params.inputMode === "draw" &&
      (params.approxMethod === "taylor" || params.approxMethod === "fourier") &&
      network
    ) {
      console.log(`[${methodLabel()}] MSE = ${formatMse(network.mse)}`);
      console.log(network.architectureText());
    }
    return;
  }
  const label = params.inputMode === "click" ? "click mode" : methodLabel();
  const mse =
    params.inputMode === "click" || params.approxMethod === "linear"
      ? formatMse(linearMse)
      : formatMse(network?.mse);
  if (params.inputMode === "click") {
    console.log(`[${label}] сегментов: ${Math.max(0, clickWaypoints.length - 1)}`);
  } else {
    console.log(`[${label}] MSE = ${mse}`);
  }
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

    clearWorkspaceState();
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
  if (params.inputMode !== "draw" || drawnPoints.length === 0) return;

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

  if (params.inputMode === "click") {
    if (mouseButton === RIGHT) {
      undoLastClick();
      return false;
    }
    if (mouseButton === LEFT) {
      addClickPointAtMouse();
    }
    return;
  }

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
  if (params.inputMode !== "draw" || !isDrawing || !mouseInsideCanvas()) return;
  addPointAtMouse();
}

function mouseReleased() {
  if (params.inputMode !== "draw") return;
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

function isUndoShortcut() {
  if (!keyIsDown(CONTROL) && !keyIsDown(91)) return false;
  // EN: Ctrl+Z · RU: Ctrl+Я (same physical key, keyCode 90)
  if (keyCode === 90) return true;
  return key === "z" || key === "Z" || key === "я" || key === "Я";
}

function keyPressed() {
  if (params.inputMode === "click" && (keyCode === BACKSPACE || keyCode === DELETE || isUndoShortcut())) {
    undoLastClick();
    return false;
  }

  if (key === "c" || key === "C" || key === "с" || key === "С") {
    if (params.inputMode === "click") {
      clearWorkspaceState();
      updateStatusPanel();
      return false;
    }
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
