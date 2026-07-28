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
  mlpActivation: "tanh",
  trainEpochs: 500,
  trainLr: 0.02,
  showNeurons: true,
  stepHeights: true,
  freezeX0: true,
  dotPopulation: 48,
  dotControlPoints: 12,
  dotTargetRadius: 0.65,
  dotMutationScale: 0.9,
  dotEdgeOffset: 1.0,
  dotGenerationMs: 850,
  dotAvoidForbidden: true,
  dotShowForbidden: true,
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
  dotAgent: [88, 185, 255],
  dotChampion: [80, 255, 140],
  dotEnemy: [255, 176, 80],
  dotUnreachable: [255, 90, 90],
  dotForbidden: [255, 55, 75],
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
let dotPoints = [];
let dotEvolution = null;
let dotRunning = false;
let dotGenerationStartedAt = 0;
let forbiddenGrid = null;
let forbiddenStats = null;
let forbiddenError = null;

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
  } else if (params.inputMode === "dot") {
    updateDotEvolution();
    drawDotMode();
  } else {
    drawCurve();
    drawSamplePoints();
    drawApproximation();
    drawNetworkOverlay();
  }
}

const MLP_ACTIVATION_OPTIONS = [
  { value: "tanh", label: "tanh" },
  { value: "sigmoid", label: "sigmoid (σ)" },
  { value: "relu", label: "ReLU" },
  { value: "leaky_relu", label: "Leaky ReLU" },
  { value: "softplus", label: "Softplus" },
  { value: "swish", label: "Swish / SiLU" },
  { value: "gelu", label: "GELU (approx)" },
  { value: "mish", label: "Mish" },
];

function controlActivationSelect() {
  const disabled = mlpHiddenLayerCount() <= 0;
  return controlSelect(
    "mlpActivation",
    "Hidden-layer activation",
    MLP_ACTIVATION_OPTIONS,
    params.mlpActivation ?? "tanh",
    { disabled, title: disabled ? "Requires at least one hidden layer" : "" }
  );
}

function mlpHiddenLayerCount() {
  if (params.approxMethod === "taylor") return params.taylorHiddenLayers;
  if (params.approxMethod === "fourier") return params.fourierHiddenLayers;
  return 0;
}

function buildControlsPanel() {
  let drawMethodControls = "";

  if (params.inputMode === "draw") {
    if (params.approxMethod === "linear") {
      drawMethodControls = `
        <p class="note">Straight lines between blue dataset points. Segment count follows the dataset step.</p>
      `;
    } else if (params.approxMethod === "sigmoid") {
      drawMethodControls = `
      ${controlSlider("sigmoidK", "k (sigmoid steepness σ)", 1, 300, 1, params.sigmoidK, 0)}
      ${controlSlider("numNeurons", "Neurons (= steps)", 5, 150, 1, params.numNeurons, 0)}
      ${controlSlider("trainEpochs", "Epochs", 500, 10000, 100, params.trainEpochs, 0)}
      ${controlSlider("trainLr", "Learning rate", 0.01, 0.2, 0.01, params.trainLr, 2)}
      ${controlCheckbox("stepHeights", "Initialize w from line", params.stepHeights)}
      ${controlCheckbox("freezeX0", "Freeze x₀ (uniform spacing)", params.freezeX0)}
      ${controlCheckbox("showNeurons", "Show x₀ lines", params.showNeurons)}
      <button id="btn-retrain" type="button">Retrain</button>
      <p class="note">Uniform x₀ along x. w[i] = height jump of the line at x₀[i]. k controls step sharpness.</p>
    `;
    } else if (params.approxMethod === "taylor") {
      drawMethodControls = `
      ${controlSlider("taylorOrder", "Order n (φ input)", 1, 20, 1, params.taylorOrder, 0)}
      ${controlSlider("taylorHiddenLayers", "Hidden layers", 0, 4, 1, params.taylorHiddenLayers, 0)}
      ${controlSlider("taylorHiddenSize", "Neurons per layer", 2, 64, 1, params.taylorHiddenSize, 0)}
      ${controlActivationSelect()}
      ${controlSlider("trainEpochs", "Epochs", 500, 10000, 100, params.trainEpochs, 0)}
      ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
      <button id="btn-retrain" type="button">Retrain</button>
      <p class="note">φ(t)=[1,t,…,tⁿ], t=(x−c)/s → MLP → y. 0 hidden layers = pure polynomial; &gt;0 = nonlinear activation between φ and y.</p>
    `;
    } else if (params.approxMethod === "fourier") {
      drawMethodControls = `
      ${controlSlider("fourierHarmonics", "Harmonics K", 1, 40, 1, params.fourierHarmonics, 0)}
      ${controlSlider("fourierHiddenLayers", "Hidden layers", 0, 4, 1, params.fourierHiddenLayers, 0)}
      ${controlSlider("fourierHiddenSize", "Neurons per layer", 2, 64, 1, params.fourierHiddenSize, 0)}
      ${controlActivationSelect()}
      ${controlSlider("trainEpochs", "Epochs", 500, 10000, 100, params.trainEpochs, 0)}
      ${controlSlider("trainLr", "Learning rate", 0.001, 0.2, 0.001, params.trainLr, 3)}
      <button id="btn-retrain" type="button">Retrain</button>
      <p class="note">φ(t)=[1,cos(kπt),sin(kπt),…], π=3.1416, t=(x−c)/s → MLP → y. 0 layers = Fourier series.</p>
    `;
    }
  }

  let modeSpecificControls = "";
  if (params.inputMode === "click") {
    modeSpecificControls = `
    <p class="note">1st click — <strong>active soldier</strong> (purple <strong>A</strong>). Then place targets in click order. Click <strong>left</strong> of the previous point → vertical segment. Formula has no <code>y=</code> prefix. <strong>Ctrl+Z</strong> / right-click / Backspace — undo last point.</p>
    <button id="btn-undo-click" type="button" class="secondary">Undo last click</button>
  `;
  } else if (params.inputMode === "dot") {
    modeSpecificControls = `
    <div class="dot-mode-section">
      <p class="note">1st click — active soldier. Other clicks are unordered enemy targets. Evolution always moves right; targets left of A are marked unreachable.</p>
      ${controlSlider("dotPopulation", "Population", 12, 120, 1, params.dotPopulation, 0)}
      ${controlSlider("dotControlPoints", "Control points", 4, 24, 1, params.dotControlPoints, 0)}
      ${controlSlider("dotTargetRadius", "Hit radius", 0.2, 1.5, 0.05, params.dotTargetRadius, 2)}
      ${controlSlider("dotMutationScale", "Mutation scale", 0.1, 2.5, 0.05, params.dotMutationScale, 2)}
      ${controlSlider("dotEdgeOffset", "Edge penalty offset", 0, 4, 0.1, params.dotEdgeOffset, 1)}
      ${controlSlider("dotGenerationMs", "Generation time (ms)", 250, 2000, 50, params.dotGenerationMs, 0)}
      ${controlCheckbox("dotAvoidForbidden", "Avoid detected black zones", params.dotAvoidForbidden)}
      ${controlCheckbox("dotShowForbidden", "Show forbidden-mask overlay", params.dotShowForbidden)}
      <div class="dot-actions">
        <button id="btn-dot-start" type="button">Start evolution</button>
        <button id="btn-dot-stop" type="button" class="secondary">Stop</button>
        <button id="btn-dot-undo" type="button" class="secondary">Undo point</button>
        <button id="btn-dot-clear" type="button" class="secondary">Clear points</button>
      </div>
    </div>
  `;
  } else {
    modeSpecificControls = `
    <div class="draw-mode-section">
      ${controlDrawMethodPicker()}
      ${controlSlider("sampleStep", "Dataset step", 0.1, 2, 0.05, params.sampleStep, 2)}
      ${drawMethodControls}
    </div>
  `;
  }

  let legendHtml = "";
  if (params.inputMode === "click") {
    legendHtml = `
    <p class="legend">
      <span class="swatch anchor"></span> active (A)
      <span class="swatch click"></span> targets
      <span class="swatch approx"></span> segments
    </p>
  `;
  } else if (params.inputMode === "dot") {
    legendHtml = `
    <p class="legend">
      <span class="swatch anchor"></span> active
      <span class="swatch enemy"></span> enemies
      <span class="swatch agent"></span> population
      <span class="swatch champion"></span> champion
      <span class="swatch forbidden"></span> forbidden
    </p>
  `;
  } else {
    legendHtml = `
    <p class="legend">
      <span class="swatch target"></span> target curve
      <span class="swatch approx"></span> approximation
      <span class="swatch sample"></span> training
    </p>
  `;
  }

  controlsEl.innerHTML = `
    <h2>Parameters</h2>
    <button id="btn-capture-field" type="button">Capture field</button>
    ${controlInputModePicker()}
    ${modeSpecificControls}
    <button id="btn-copy-formula" type="button" disabled>Copy y</button>
    <button id="btn-reset" type="button" class="secondary">Reset</button>
    <p id="status-mse" class="status">—</p>
    ${legendHtml}
  `;

  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    if (input.dataset.param === "mlpActivation") {
      input.dataset.lastValue = input.value;
      input.addEventListener("change", onMlpActivationChange);
      return;
    }
    if (input.tagName === "SELECT") {
      input.addEventListener("change", onParamChange);
      return;
    }
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
  const dotStartBtn = document.getElementById("btn-dot-start");
  if (dotStartBtn) dotStartBtn.addEventListener("click", startDotEvolution);
  const dotStopBtn = document.getElementById("btn-dot-stop");
  if (dotStopBtn) dotStopBtn.addEventListener("click", stopDotEvolution);
  const dotUndoBtn = document.getElementById("btn-dot-undo");
  if (dotUndoBtn) dotUndoBtn.addEventListener("click", undoLastDotPoint);
  const dotClearBtn = document.getElementById("btn-dot-clear");
  if (dotClearBtn) dotClearBtn.addEventListener("click", clearDotPoints);
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
  syncActivationSelectState();
  updateCopyButton();
  updateCaptureButton();
  updateStatusPanel();
  updateDotButtons();
}

function controlInputModePicker() {
  return `
    <div class="control mode-picker">
      <div class="control-head"><span>Mode</span></div>
      <div class="mode-options">
        <label class="mode-option">
          <input type="radio" name="inputMode" data-param="inputMode" value="click" ${params.inputMode === "click" ? "checked" : ""} />
          <span>1. Click mode</span>
        </label>
        <label class="mode-option">
          <input type="radio" name="inputMode" data-param="inputMode" value="draw" ${params.inputMode === "draw" ? "checked" : ""} />
          <span>2. Draw mode</span>
        </label>
        <label class="mode-option">
          <input type="radio" name="inputMode" data-param="inputMode" value="dot" ${params.inputMode === "dot" ? "checked" : ""} />
          <span>3. Dot mode</span>
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

function controlSelect(key, label, options, value, { disabled = false, title = "" } = {}) {
  const opts = options
    .map(
      (opt) =>
        `<option value="${opt.value}" ${opt.value === value ? "selected" : ""}>${opt.label}</option>`
    )
    .join("");
  const disabledAttr = disabled ? " disabled" : "";
  const titleAttr = title ? ` title="${title}"` : "";
  return `
    <label class="control">
      <div class="control-head"><span>${label}</span></div>
      <select data-param="${key}"${disabledAttr}${titleAttr}>${opts}</select>
    </label>
  `;
}

function controlDrawMethodPicker() {
  return `
    <div class="control method-picker nested">
      <div class="control-head"><span>Approximation method</span></div>
      <div class="method-options">
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="linear" ${params.approxMethod === "linear" ? "checked" : ""} />
          <span>2.1 Linear (segments)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="sigmoid" ${params.approxMethod === "sigmoid" ? "checked" : ""} />
          <span>2.2 Sigmoid network</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="taylor" ${params.approxMethod === "taylor" ? "checked" : ""} />
          <span>2.3 Taylor (polynomial) (beta)</span>
        </label>
        <label class="method-option">
          <input type="radio" name="approxMethod" data-param="approxMethod" value="fourier" ${params.approxMethod === "fourier" ? "checked" : ""} />
          <span>2.4 Fourier (harmonics)</span>
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
    key === "fourierHiddenSize" ||
    key === "dotPopulation" ||
    key === "dotControlPoints" ||
    key === "dotGenerationMs"
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

function readParamFromInput(input) {
  if (input.type === "checkbox") return input.checked;
  if (input.type === "radio") return input.value;
  if (input.tagName === "SELECT") return input.value;
  return parseFloat(input.value);
}

function readParamsFromUI() {
  controlsEl.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    if (input.type === "radio" && !input.checked) return;
    params[key] = readParamFromInput(input);
  });
}

function onParamInput(event) {
  const input = event.target;
  const key = input.dataset.param;
  const decimals = parseInt(input.dataset.decimals || "2", 10);
  const label = controlsEl.querySelector(`[data-value-for="${key}"]`);
  if (label) label.textContent = formatParam(key, parseFloat(input.value), decimals);
}

function onMlpActivationChange(event) {
  const select = event.target;
  const newValue = select.value;
  if (newValue === select.dataset.lastValue) return;
  select.dataset.lastValue = newValue;
  params.mlpActivation = newValue;
  if (params.inputMode !== "draw" || drawnPoints.length === 0) return;
  rerunPipeline();
}

function onParamChange(event) {
  const input = event.target;
  const changedKey = input?.dataset?.param;
  if (!changedKey) return;
  if (input.type === "radio" && !input.checked) return;

  const prevInputMode = params.inputMode;
  const prevMethod = params.approxMethod;
  const prevValue = params[changedKey];
  const newValue = readParamFromInput(input);

  readParamsFromUI();
  syncActivationSelectState();

  if (changedKey === "inputMode" && params.inputMode !== prevInputMode) {
    clearWorkspaceState();
    buildControlsPanel();
    return;
  }

  if (params.inputMode === "dot") {
    if (newValue !== prevValue && changedKey !== "dotShowForbidden") {
      resetDotEvolutionEngine();
    }
    updateStatusPanel();
    updateCopyButton();
    updateDotButtons();
    return;
  }

  if (params.inputMode !== "draw") return;

  if (changedKey === "approxMethod" && params.approxMethod !== prevMethod) {
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
  if (drawnPoints.length > 0 && newValue !== prevValue) {
    rerunPipeline();
  }
}

function syncActivationSelectState() {
  const select = controlsEl?.querySelector('[data-param="mlpActivation"]');
  if (!select) return;
  const enabled = mlpHiddenLayerCount() > 0;
  select.disabled = !enabled;
  select.title = enabled ? "" : "Requires at least one hidden layer";
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
  text("1st click — active soldier (A)", cx, 14);
  textSize(11);
  fill(...COLORS.axisLabel);
  text("Then click targets; click left of previous → vertical segment", cx, 34);
}

function dotConfigFromParams() {
  const ignoredPoints = dotPoints.map((point) => ({ ...point }));
  const constraintEvaluators =
    params.dotAvoidForbidden && forbiddenGrid
      ? [(path) => forbiddenGrid.pathPenalty(path, ignoredPoints, 0.55)]
      : [];
  return {
    populationSize: params.dotPopulation,
    controlPoints: params.dotControlPoints,
    targetRadius: params.dotTargetRadius,
    mutationScale: params.dotMutationScale,
    edgeOffset: params.dotEdgeOffset,
    yMin: Y_MIN,
    yMax: Y_MAX,
    xMax: X_MAX,
    constraintEvaluators,
  };
}

function startDotEvolution() {
  readParamsFromUI();
  if (dotPoints.length < 2) return;
  if (params.dotAvoidForbidden && !forbiddenGrid) {
    alert(
      forbiddenError ||
        "Forbidden mask is not loaded.\nRestart the Python server and press Capture field again, or disable “Avoid detected black zones”."
    );
    return;
  }
  dotEvolution = new DotEvolution(dotPoints[0], dotPoints.slice(1), dotConfigFromParams());
  dotRunning = true;
  dotGenerationStartedAt = millis();
  updateStatusPanel();
  updateCopyButton();
  updateDotButtons();
}

function stopDotEvolution() {
  dotRunning = false;
  updateStatusPanel();
  updateDotButtons();
}

function resetDotEvolutionEngine() {
  dotRunning = false;
  dotEvolution = null;
  dotGenerationStartedAt = 0;
  updateStatusPanel();
  updateCopyButton();
  updateDotButtons();
}

function updateDotEvolution() {
  if (!dotRunning || !dotEvolution) return;
  if (millis() - dotGenerationStartedAt < params.dotGenerationMs) return;
  dotEvolution.evolve();
  dotGenerationStartedAt = millis();
  updateStatusPanel();
  updateCopyButton();
}

function updateDotButtons() {
  const startBtn = document.getElementById("btn-dot-start");
  const stopBtn = document.getElementById("btn-dot-stop");
  const undoBtn = document.getElementById("btn-dot-undo");
  const clearBtn = document.getElementById("btn-dot-clear");
  if (startBtn) {
    startBtn.disabled = dotPoints.length < 2;
    startBtn.textContent = dotEvolution ? "Restart evolution" : "Start evolution";
  }
  if (stopBtn) stopBtn.disabled = !dotRunning;
  if (undoBtn) undoBtn.disabled = dotPoints.length === 0;
  if (clearBtn) clearBtn.disabled = dotPoints.length === 0;
}

function addDotPointAtMouse() {
  const world = screenToWorld(mouseX, mouseY);
  dotPoints.push({
    x: constrain(world.x, X_MIN, X_MAX),
    y: constrain(world.y, Y_MIN, Y_MAX),
  });
  resetDotEvolutionEngine();
  updateStatusPanel();
  updateDotButtons();
}

function undoLastDotPoint() {
  if (dotPoints.length === 0) return;
  dotPoints.pop();
  resetDotEvolutionEngine();
}

function clearDotPoints() {
  dotPoints = [];
  resetDotEvolutionEngine();
}

function drawDotMode() {
  const duration = Math.max(1, params.dotGenerationMs);
  const rawProgress = dotRunning
    ? constrain((millis() - dotGenerationStartedAt) / (duration * 0.78), 0, 1)
    : 1;
  const progress = 1 - Math.pow(1 - rawProgress, 3);

  drawDotEdgeOffsetLines();
  if (params.dotShowForbidden && forbiddenGrid) drawForbiddenGridOverlay();
  if (dotEvolution) drawDotPopulation(progress);
  drawDotMarkers();
  drawDotOverlay();

  if (dotPoints.length === 0) drawDotModeHint();
}

function drawDotEdgeOffsetLines() {
  const offset = params.dotEdgeOffset;
  if (offset <= 0) return;
  const lowerLine = Y_MIN + offset;
  const upperLine = Y_MAX - offset;
  stroke(COLORS.dotUnreachable[0], COLORS.dotUnreachable[1], COLORS.dotUnreachable[2], 105);
  strokeWeight(1);
  drawingContext.setLineDash([7, 7]);
  lineWorld(X_MIN, lowerLine, X_MAX, lowerLine);
  lineWorld(X_MIN, upperLine, X_MAX, upperLine);
  drawingContext.setLineDash([]);
}

function drawForbiddenGridOverlay() {
  const payload = forbiddenGrid.payload;
  noStroke();
  fill(COLORS.dotForbidden[0], COLORS.dotForbidden[1], COLORS.dotForbidden[2], 38);
  for (let row = 0; row < payload.rows_rle.length; row++) {
    const y = (row * payload.cell_px * height) / payload.image_height;
    const cellHeight = (payload.cell_px * height) / payload.image_height + 0.35;
    for (const [start, length] of payload.rows_rle[row]) {
      const x = (start * payload.cell_px * width) / payload.image_width;
      const runWidth = (length * payload.cell_px * width) / payload.image_width + 0.35;
      rect(x, y, runWidth, cellHeight);
    }
  }
}

function drawDotPopulation(progress) {
  const population = dotEvolution.population;
  for (let index = population.length - 1; index >= 1; index--) {
    const quality = 1 - index / Math.max(1, population.length - 1);
    const alpha = 11 + quality * 34;
    stroke(COLORS.dotAgent[0], COLORS.dotAgent[1], COLORS.dotAgent[2], alpha);
    strokeWeight(0.7 + quality * 0.5);
    noFill();
    drawPartialDotPath(population[index].path, progress);

    if (index < 11 && progress > 0.015) {
      const head = pointOnDotPath(population[index].path, progress);
      const screen = worldToScreen(head.x, head.y);
      noStroke();
      fill(COLORS.dotAgent[0], COLORS.dotAgent[1], COLORS.dotAgent[2], 55 + quality * 80);
      circle(screen.x, screen.y, 2.5 + quality * 1.5);
    }
  }

  const champion = population[0];
  if (!champion) return;
  noFill();
  stroke(COLORS.dotChampion[0], COLORS.dotChampion[1], COLORS.dotChampion[2], 45);
  strokeWeight(7);
  drawPartialDotPath(champion.path, progress);
  stroke(COLORS.dotChampion[0], COLORS.dotChampion[1], COLORS.dotChampion[2], 235);
  strokeWeight(2.3);
  drawPartialDotPath(champion.path, progress);

  if (progress > 0.015) {
    const head = pointOnDotPath(champion.path, progress);
    const screen = worldToScreen(head.x, head.y);
    noStroke();
    fill(COLORS.dotChampion[0], COLORS.dotChampion[1], COLORS.dotChampion[2], 55);
    circle(screen.x, screen.y, 15);
    fill(...COLORS.dotChampion);
    circle(screen.x, screen.y, 6);
  }
}

function drawPartialDotPath(path, progress) {
  if (!path || path.length < 2 || progress <= 0) return;
  const position = constrain(progress, 0, 1) * (path.length - 1);
  const fullSegments = Math.floor(position);
  const remainder = position - fullSegments;
  beginShape();
  for (let i = 0; i <= fullSegments && i < path.length; i++) {
    const screen = worldToScreen(path[i].x, path[i].y);
    vertex(screen.x, screen.y);
  }
  if (fullSegments < path.length - 1 && remainder > 1e-6) {
    const left = path[fullSegments];
    const right = path[fullSegments + 1];
    const x = left.x + (right.x - left.x) * remainder;
    const y = left.y + (right.y - left.y) * remainder;
    const screen = worldToScreen(x, y);
    vertex(screen.x, screen.y);
  }
  endShape();
}

function pointOnDotPath(path, progress) {
  const position = constrain(progress, 0, 1) * (path.length - 1);
  const index = Math.min(path.length - 2, Math.floor(position));
  const t = Math.min(1, position - index);
  return {
    x: path[index].x + (path[index + 1].x - path[index].x) * t,
    y: path[index].y + (path[index + 1].y - path[index].y) * t,
  };
}

function drawDotMarkers() {
  if (dotPoints.length === 0) return;
  const start = dotPoints[0];
  const radiusPx = (params.dotTargetRadius * width) / (X_MAX - X_MIN);
  const bestDistances = dotEvolution?.bestEver?.fitness?.targetDistances ?? [];

  noStroke();
  textSize(11);
  textAlign(CENTER, CENTER);
  for (let index = 1; index < dotPoints.length; index++) {
    const target = dotPoints[index];
    const screen = worldToScreen(target.x, target.y);
    const unreachable = target.x < start.x - params.dotTargetRadius;
    const hit = bestDistances[index - 1] <= params.dotTargetRadius;
    const color = unreachable
      ? COLORS.dotUnreachable
      : hit
        ? COLORS.dotChampion
        : COLORS.dotEnemy;
    const pulse = 1 + Math.sin(frameCount * 0.08 + index) * 0.08;

    fill(color[0], color[1], color[2], 22);
    circle(screen.x, screen.y, radiusPx * 2 * pulse);
    fill(color[0], color[1], color[2], 54);
    circle(screen.x, screen.y, 20);
    fill(...color);
    circle(screen.x, screen.y, 10);
    fill(20, 20, 24);
    text(unreachable ? "!" : String(index), screen.x, screen.y);
  }

  const anchor = worldToScreen(start.x, start.y);
  fill(...COLORS.anchorGlow);
  circle(anchor.x, anchor.y, 24);
  fill(...COLORS.anchor);
  circle(anchor.x, anchor.y, 12);
  fill(20, 20, 24);
  text("A", anchor.x, anchor.y);
}

function drawDotOverlay() {
  if (!dotEvolution) return;
  const best = dotEvolution.bestEver.fitness;
  noStroke();
  fill(...COLORS.panelText);
  textAlign(LEFT, TOP);
  textSize(12);
  const phase = dotRunning ? "evolving" : "paused";
  text(`generation ${dotEvolution.generation}  •  ${phase}`, 12, 12);
  fill(...COLORS.dotChampion);
  text(
    `champion: ${best.hits}/${dotEvolution.targets.length} hits  •  miss ${roundCoord(best.missDistance)}`,
    12,
    29
  );
  fill(
    ...(best.constraintPenalty > 0 ? COLORS.dotUnreachable : COLORS.dotChampion)
  );
  text(
    best.constraintPenalty > 0
      ? `forbidden collision penalty ${roundCoord(best.constraintPenalty)}`
      : "trajectory is clear of the detected mask",
    12,
    46
  );
}

function drawDotModeHint() {
  noStroke();
  fill(...COLORS.panelText);
  textSize(13);
  textAlign(CENTER, TOP);
  text("1st click — active soldier (A)", width / 2, 14);
  textSize(11);
  fill(...COLORS.axisLabel);
  text("Then place enemy targets and start evolution", width / 2, 34);
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
    const act = params.taylorHiddenLayers > 0 ? `  ${network.activation}` : "";
    text(`n=${params.taylorOrder}  L=${params.taylorHiddenLayers}×${params.taylorHiddenSize}${act}`, 12, 12);
    text(`c=${center}  s=${scale}`, 12, 28);
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 44);
  } else if (params.approxMethod === "fourier" && network) {
    const center = roundCoord(network.center);
    const scale = roundCoord(network.scale);
    const act = params.fourierHiddenLayers > 0 ? `  ${network.activation}` : "";
    text(`K=${params.fourierHarmonics}  L=${params.fourierHiddenLayers}×${params.fourierHiddenSize}${act}`, 12, 12);
    text(`c=${center}  s=${scale}`, 12, 28);
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 44);
  } else {
    fill(...COLORS.approx);
    text(`MSE (${methodLabel()}) = ${formatMse(activeMse)}`, 12, 12);
    text(`segments: ${linearWaypoints.length - 1}`, 12, 28);
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
    statusMseEl.textContent = `active (A): ${clickPoints.length > 0 ? "yes" : "no"}  |  targets: ${targets}  |  segments: ${segments}`;
    return;
  }

  if (params.inputMode === "dot") {
    const targets = Math.max(0, dotPoints.length - 1);
    const start = dotPoints[0];
    const unreachable = start
      ? dotPoints.slice(1).filter((target) => target.x < start.x - params.dotTargetRadius).length
      : 0;
    if (!dotEvolution) {
      const maskState = forbiddenError
        ? `mask error: ${forbiddenError}`
        : forbiddenStats
          ? `mask cells: ${forbiddenStats.grid_forbidden_cells}`
          : "mask: capture field first";
      statusMseEl.textContent =
        `active (A): ${start ? "yes" : "no"}  |  targets: ${targets}` +
        `  |  ${maskState}` +
        (unreachable > 0 ? `  |  unreachable: ${unreachable}` : "");
      return;
    }
    const best = dotEvolution.bestEver.fitness;
    const state = dotRunning ? "running" : "paused";
    const collisionState =
      best.constraintPenalty > 0 ? `blocked ${roundCoord(best.constraintPenalty)}` : "safe";
    statusMseEl.textContent =
      `${state}  |  generation: ${dotEvolution.generation}` +
      `  |  hits: ${best.hits}/${targets}` +
      `  |  obstacles: ${collisionState}` +
      `  |  miss: ${roundCoord(best.missDistance)}` +
      `  |  edge: ${roundCoord(best.edgePenalty)}` +
      `  |  smoothness: ${roundCoord(best.roughness)}` +
      (unreachable > 0 ? `  |  unreachable: ${unreachable}` : "");
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
      ? `active: ${activeMethodLabel} ${formatMse(linearMse)}`
      : `active: ${activeMethodLabel} ${formatMse(network?.mse)}`;
  const compare = `compare — linear: ${formatMse(linearMse)}  |  sigmoid: ${formatMse(sigmoidMse)}  |  Taylor: ${formatMse(taylorMse)}  |  Fourier: ${formatMse(fourierMse)}`;
  const meta = `training points: ${trainingData.length}  |  segments: ${Math.max(0, linearWaypoints.length - 1)}`;
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
  dotPoints = [];
  dotRunning = false;
  dotEvolution = null;
  dotGenerationStartedAt = 0;
  updateStatusPanel();
  updateCopyButton();
  updateDotButtons();
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

  if (params.inputMode === "dot") {
    const path = dotEvolution?.bestEver?.path;
    if (!path || path.length < 2) return null;
    return clickFormulaText(path);
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
  if (method === "linear") return "linear";
  if (method === "taylor") return "Taylor";
  if (method === "fourier") return "Fourier";
  return "sigmoid";
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
  const label =
    params.inputMode === "click"
      ? "click mode"
      : params.inputMode === "dot"
        ? "dot mode"
        : methodLabel();
  const mse =
    params.inputMode === "click" ||
    params.inputMode === "dot" ||
    params.approxMethod === "linear"
      ? formatMse(linearMse)
      : formatMse(network?.mse);
  if (params.inputMode === "click") {
    console.log(`[${label}] segments: ${Math.max(0, clickWaypoints.length - 1)}`);
  } else if (params.inputMode === "dot") {
    const best = dotEvolution?.bestEver?.fitness;
    console.log(
      `[${label}] generation: ${dotEvolution?.generation ?? 0}, hits: ${best?.hits ?? 0}/${Math.max(0, dotPoints.length - 1)}`
    );
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
    copyBtnEl.textContent = "Copied";
  } catch {
    copyBtnEl.textContent = "Error";
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
  captureBtnEl.textContent = isCapturing ? "Capturing..." : "Capture field";
}

async function captureGameField() {
  if (isCapturing) return;
  isCapturing = true;
  updateCaptureButton();

  try {
    const response = await fetch("/api/capture", { method: "POST" });
    const data = await response.json();
    if (!data.ok) {
      alert(data.error || "Failed to capture field");
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
    forbiddenGrid = data.forbidden_grid ? new ForbiddenGrid(data.forbidden_grid) : null;
    forbiddenStats = data.forbidden_stats ?? null;
    forbiddenError =
      data.forbidden_error ||
      (data.forbidden_grid
        ? null
        : "Capture response has no forbidden grid. Restart the Python server.");
    if (forbiddenStats) {
      console.info(
        `[forbidden mask] components=${forbiddenStats.components}, cells=${forbiddenStats.grid_forbidden_cells}/${forbiddenStats.grid_total_cells}`
      );
    }
    if (forbiddenError) {
      console.warn(`[forbidden mask] ${forbiddenError}`);
      alert(
        `${forbiddenError}\n\nStop the current server, run python tools/approximator_server.py again, then press Capture field.`
      );
    }
    updateStatusPanel();
  } catch {
    alert(
      "Capture server unavailable.\nRun: python tools/approximator_server.py\nThen open http://127.0.0.1:8765/"
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
  const activation = params.mlpActivation ?? "tanh";
  network = new TaylorNetwork(
    params.taylorOrder,
    params.taylorHiddenLayers,
    params.taylorHiddenSize,
    activation
  );
  network.initFromData(trainingData, approxXMin, approxXMax);
  network.train(trainingData, params.trainEpochs, params.trainLr);
  taylorMse = network.mse;
}

function trainFourierNetwork() {
  const activation = params.mlpActivation ?? "tanh";
  network = new FourierNetwork(
    params.fourierHarmonics,
    params.fourierHiddenLayers,
    params.fourierHiddenSize,
    activation
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

  if (params.inputMode === "dot") {
    if (mouseButton === RIGHT) {
      undoLastDotPoint();
      return false;
    }
    if (mouseButton === LEFT) addDotPointAtMouse();
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
  if (params.inputMode !== "draw" || !isDrawing) return;
  isDrawing = false;
  finishDrawing();
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
  // EN: Ctrl+Z · RU keyboard: Ctrl+Я (same physical key, keyCode 90)
  if (keyCode === 90) return true;
  return key === "z" || key === "Z" || key === "я" || key === "Я";
}

function keyPressed() {
  if (params.inputMode === "click" && (keyCode === BACKSPACE || keyCode === DELETE || isUndoShortcut())) {
    undoLastClick();
    return false;
  }

  if (
    params.inputMode === "dot" &&
    (keyCode === BACKSPACE || keyCode === DELETE || isUndoShortcut())
  ) {
    undoLastDotPoint();
    return false;
  }

  if (params.inputMode === "dot" && key === " ") {
    if (dotRunning) {
      stopDotEvolution();
    } else {
      startDotEvolution();
    }
    return false;
  }

  if (key === "c" || key === "C" || key === "с" || key === "С") {
    if (params.inputMode === "click") {
      clearWorkspaceState();
      updateStatusPanel();
      return false;
    }
    if (params.inputMode === "dot") {
      clearDotPoints();
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
