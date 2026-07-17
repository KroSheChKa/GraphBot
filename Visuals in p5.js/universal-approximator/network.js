// y = bias + Σ w_i · σ(k · (x - x0_i))

const MAX_WEIGHT = 30;
const MAX_BIAS = 20;

class SigmoidNetwork {
  constructor(numNeurons, k = 100) {
    this.k = k;
    this.numNeurons = numNeurons;
    this.x0 = new Array(numNeurons).fill(0);
    this.w = new Array(numNeurons).fill(0);
    this.bias = 0;
    this.mse = null;
    this.x0Mode = "uniform";
    this.x0Step = null;
  }

  sigmoid(z) {
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    }
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  forward(x) {
    let y = this.bias;
    const hidden = [];

    for (let i = 0; i < this.numNeurons; i++) {
      const s = this.sigmoid(this.k * (x - this.x0[i]));
      hidden.push(s);
      y += this.w[i] * s;
    }

    return { y, hidden };
  }

  predict(x) {
    return this.forward(x).y;
  }

  initFromData(data, xMin, xMax, options = {}) {
    this.x0 = uniformX0(this.numNeurons, xMin, xMax);
    this.x0Mode = "uniform";
    this.x0Step =
      this.numNeurons > 1 ? (xMax - xMin) / (this.numNeurons - 1) : 0;

    if (options.stepHeights !== false) {
      this.initStepHeights(data, xMin);
    } else {
      const meanY = data.reduce((sum, pt) => sum + pt.y, 0) / data.length;
      this.bias = meanY;
      this.w = new Array(this.numNeurons).fill(0);
      this.sanitizeParams();
    }
  }

  sanitizeParams() {
    if (!Number.isFinite(this.bias)) this.bias = 0;
    this.bias = clamp(this.bias, -MAX_BIAS, MAX_BIAS);

    for (let i = 0; i < this.numNeurons; i++) {
      if (!Number.isFinite(this.w[i])) this.w[i] = 0;
      this.w[i] = clamp(this.w[i], -MAX_WEIGHT, MAX_WEIGHT);
      if (!Number.isFinite(this.x0[i])) this.x0[i] = 0;
    }
  }

  initStepHeights(data, xMin) {
    const yStart = sampleYFromData(data, xMin);
    this.bias = Number.isFinite(yStart) ? yStart : 0;

    let prevY = this.bias;
    for (let i = 0; i < this.numNeurons; i++) {
      const yi = sampleYFromData(data, this.x0[i]);
      const safeY = Number.isFinite(yi) ? yi : prevY;
      this.w[i] = safeY - prevY;
      prevY = safeY;
    }

    this.sanitizeParams();
  }

  train(data, xMin, xMax, epochs = 4000, lr = 0.08, freezeX0 = true) {
    if (epochs <= 0 || data.length === 0) {
      this.mse = this.computeMse(data);
      return this.mse;
    }

    const safeLr = Math.min(lr, 0.05);

    for (let epoch = 0; epoch < epochs; epoch++) {
      const n = data.length;
      let gradBias = 0;
      const gradW = new Array(this.numNeurons).fill(0);
      const gradX0 = new Array(this.numNeurons).fill(0);

      for (const pt of data) {
        const { y: pred, hidden } = this.forward(pt.x);
        if (!Number.isFinite(pred)) {
          this.sanitizeParams();
          break;
        }

        const err = pred - pt.y;
        const grad = (2 / n) * err;

        if (!Number.isFinite(grad)) continue;

        gradBias += grad;

        for (let i = 0; i < this.numNeurons; i++) {
          const s = hidden[i];
          gradW[i] += grad * s;
          if (!freezeX0) {
            const ds = s * (1 - s);
            gradX0[i] += grad * this.w[i] * ds * (-this.k);
          }
        }
      }

      this.bias -= safeLr * gradBias;
      for (let i = 0; i < this.numNeurons; i++) {
        this.w[i] -= safeLr * gradW[i];
        if (!freezeX0) {
          this.x0[i] -= safeLr * gradX0[i];
          this.x0[i] = clamp(this.x0[i], xMin - 5, xMax + 5);
        }
      }

      this.sanitizeParams();
      if (!Number.isFinite(this.bias) || this.w.some((w) => !Number.isFinite(w))) {
        break;
      }
    }

    this.mse = this.computeMse(data);
    return this.mse;
  }

  computeMse(data) {
    if (data.length === 0) return null;
    let sum = 0;
    let count = 0;

    for (const pt of data) {
      const pred = this.predict(pt.x);
      if (!Number.isFinite(pred)) return null;
      const err = pred - pt.y;
      sum += err * err;
      count += 1;
    }

    return count > 0 ? sum / count : null;
  }

  toFormulaText(minWeight = 0.005) {
    const bias = round3(this.bias);
    const parts = [bias !== null ? `${bias}` : "0"];

    for (let i = 0; i < this.numNeurons; i++) {
      if (!Number.isFinite(this.w[i]) || Math.abs(this.w[i]) < minWeight) continue;
      const w = round3(this.w[i]);
      const x0 = round3(this.x0[i]);
      if (w === null || x0 === null) continue;
      parts.push(`${w}·σ(${this.k}·(x − ${x0}))`);
    }

    return `y = ${parts.join(" + ")}`;
  }

  toDesmosText(minWeight = 0.005) {
    const bias = round3(this.bias);
    const parts = [bias !== null ? `${bias}` : "0"];

    for (let i = 0; i < this.numNeurons; i++) {
      if (!Number.isFinite(this.w[i]) || Math.abs(this.w[i]) < minWeight) continue;
      const w = round3(this.w[i]);
      const x0 = round3(this.x0[i]);
      if (w === null || x0 === null) continue;
      parts.push(`${w}/(1+exp(-${this.k}*(x-${x0})))`);
    }

    return `y=${parts.join("+")}`;
  }
}

// Feature MLP: φ(t) → hidden layers → y  (Taylor / Fourier features)

const MAX_FEATURE_MLP_WEIGHT = 30;
const MAX_FEATURE_MLP_BIAS = 20;
const FEATURE_MLP_ACTIVATIONS = ["tanh", "relu", "sigmoid"];
const FOURIER_PI = Math.round(Math.PI * 10000) / 10000;

class FeatureMlpNetwork {
  constructor(numHiddenLayers, hiddenSize, activation = "tanh") {
    this.numHiddenLayers = numHiddenLayers;
    this.hiddenSize = hiddenSize;
    this.activation = FEATURE_MLP_ACTIVATIONS.includes(activation) ? activation : "tanh";
    this.center = 0;
    this.scale = 1;
    this.layers = [];
    this.mse = null;
  }

  networkName() {
    return "Feature-MLP";
  }

  inputSize() {
    throw new Error("inputSize() must be implemented by subclass");
  }

  features(x) {
    throw new Error("features(x) must be implemented by subclass");
  }

  featureExprsForDesmos() {
    throw new Error("featureExprsForDesmos() must be implemented by subclass");
  }

  scaledX(x) {
    return (x - this.center) / this.scale;
  }

  activate(z) {
    if (this.activation === "relu") return Math.max(0, z);
    if (this.activation === "sigmoid") return featureMlpSigmoid(z);
    return Math.tanh(z);
  }

  activateDerivFromPre(z, activated) {
    if (this.activation === "relu") return z > 0 ? 1 : 0;
    if (this.activation === "sigmoid") return activated * (1 - activated);
    return 1 - activated * activated;
  }

  buildLayers(outputBias = 0) {
    const sizes = [this.inputSize()];
    for (let i = 0; i < this.numHiddenLayers; i++) sizes.push(this.hiddenSize);
    sizes.push(1);

    this.layers = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const inD = sizes[l];
      const outD = sizes[l + 1];
      const isHidden = l < sizes.length - 2;
      const scale = isHidden && this.activation === "relu"
        ? Math.sqrt(2 / inD)
        : Math.sqrt(6 / (inD + outD));

      const W = [];
      for (let o = 0; o < outD; o++) {
        W[o] = [];
        for (let i = 0; i < inD; i++) {
          W[o][i] = (Math.random() * 2 - 1) * scale;
        }
      }

      const b = new Array(outD).fill(0);
      if (!isHidden) b[0] = outputBias;
      this.layers.push({ W, b });
    }
  }

  forward(x) {
    const activations = [this.features(x)];
    const preActivations = [];

    for (let l = 0; l < this.layers.length - 1; l++) {
      const { W, b } = this.layers[l];
      const z = linearLayer(W, b, activations[l]);
      preActivations.push(z);
      activations.push(z.map((v) => this.activate(v)));
    }

    const last = this.layers[this.layers.length - 1];
    const y = dot(last.W[0], activations[activations.length - 1]) + last.b[0];
    return { y, activations, preActivations };
  }

  predict(x) {
    return this.forward(x).y;
  }

  initFromData(data, xMin, xMax) {
    this.center = (xMin + xMax) / 2;
    this.scale = Math.max((xMax - xMin) / 2, 0.5);
    const meanY = data.reduce((sum, pt) => sum + pt.y, 0) / data.length;
    this.buildLayers(Number.isFinite(meanY) ? meanY : 0);
    this.sanitizeParams();
  }

  sanitizeParams() {
    for (const layer of this.layers) {
      for (let o = 0; o < layer.b.length; o++) {
        if (!Number.isFinite(layer.b[o])) layer.b[o] = 0;
        layer.b[o] = clamp(layer.b[o], -MAX_FEATURE_MLP_BIAS, MAX_FEATURE_MLP_BIAS);
      }
      for (let o = 0; o < layer.W.length; o++) {
        for (let i = 0; i < layer.W[o].length; i++) {
          if (!Number.isFinite(layer.W[o][i])) layer.W[o][i] = 0;
          layer.W[o][i] = clamp(layer.W[o][i], -MAX_FEATURE_MLP_WEIGHT, MAX_FEATURE_MLP_WEIGHT);
        }
      }
    }
  }

  accumulateGradients(gradOut, activations, preActivations, gradW, gradB) {
    let delta = gradOut;
    const lastIdx = this.layers.length - 1;

    for (let j = 0; j < this.layers[lastIdx].W[0].length; j++) {
      gradW[lastIdx][0][j] += delta * activations[lastIdx][j];
    }
    gradB[lastIdx][0] += delta;

    delta = this.layers[lastIdx].W[0].map((w) => w * delta);

    for (let l = lastIdx - 1; l >= 0; l--) {
      const z = preActivations[l];
      const a = activations[l + 1];
      const nextDelta = new Array(this.layers[l].W[0].length).fill(0);

      for (let i = 0; i < delta.length; i++) {
        const localGrad = delta[i] * this.activateDerivFromPre(z[i], a[i]);
        for (let j = 0; j < this.layers[l].W[i].length; j++) {
          gradW[l][i][j] += localGrad * activations[l][j];
          nextDelta[j] += localGrad * this.layers[l].W[i][j];
        }
        gradB[l][i] += localGrad;
      }

      delta = nextDelta;
    }
  }

  train(data, epochs = 4000, lr = 0.02) {
    if (epochs <= 0 || data.length === 0 || this.layers.length === 0) {
      this.mse = this.computeMse(data);
      return this.mse;
    }

    const safeLr = Math.min(lr, 0.2);
    const gradW = this.layers.map((layer) => layer.W.map((row) => row.map(() => 0)));
    const gradB = this.layers.map((layer) => layer.b.map(() => 0));

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let l = 0; l < this.layers.length; l++) {
        for (let o = 0; o < gradW[l].length; o++) {
          gradW[l][o].fill(0);
        }
        gradB[l].fill(0);
      }

      let badStep = false;
      const n = data.length;

      for (const pt of data) {
        const { y, activations, preActivations } = this.forward(pt.x);
        if (!Number.isFinite(y)) {
          badStep = true;
          break;
        }

        const err = y - pt.y;
        const gradOut = (2 / n) * err;
        if (!Number.isFinite(gradOut)) continue;

        this.accumulateGradients(gradOut, activations, preActivations, gradW, gradB);
      }

      if (badStep) {
        this.sanitizeParams();
        break;
      }

      for (let l = 0; l < this.layers.length; l++) {
        for (let o = 0; o < this.layers[l].W.length; o++) {
          for (let i = 0; i < this.layers[l].W[o].length; i++) {
            this.layers[l].W[o][i] -= safeLr * gradW[l][o][i];
          }
          this.layers[l].b[o] -= safeLr * gradB[l][o];
        }
      }

      this.sanitizeParams();
    }

    this.mse = this.computeMse(data);
    return this.mse;
  }

  computeMse(data) {
    if (data.length === 0) return null;
    let sum = 0;
    for (const pt of data) {
      const pred = this.predict(pt.x);
      if (!Number.isFinite(pred)) return null;
      const err = pred - pt.y;
      sum += err * err;
    }
    return sum / data.length;
  }

  architectureText() {
    const hidden = this.numHiddenLayers > 0
      ? `${this.numHiddenLayers}×${this.hiddenSize} (${this.activation})`
      : "линейный на φ";
    return `${this.networkName()}: φ∈R^${this.inputSize()} → ${hidden} → 1`;
  }

  desmosTVar() {
    const c = round3(this.center);
    const s = round3(this.scale);
    if (c === null || s === null) return "x";
    if (Math.abs(c) < 1e-9) return s === 1 ? "x" : `(x/${s})`;
    return `((x-${c})/${s})`;
  }

  linearComboDesmos(W, b, inputs, minWeight = 0.005) {
    const exprs = [];
    for (let o = 0; o < W.length; o++) {
      const terms = [];
      for (let i = 0; i < W[o].length; i++) {
        if (Math.abs(W[o][i]) < minWeight) continue;
        const w = round3(W[o][i]);
        if (w === null) continue;
        terms.push(inputs[i] === "1" ? `${w}` : `${w}*${inputs[i]}`);
      }
      const bias = round3(b[o]);
      if (bias !== null && Math.abs(bias) >= minWeight) terms.push(`${bias}`);
      exprs.push(terms.length > 0 ? `(${terms.join("+")})` : "0");
    }
    return exprs;
  }

  activationDesmos(zExpr) {
    if (this.activation === "relu") return `max(0,${zExpr})`;
    if (this.activation === "sigmoid") return `(1/(1+exp(-${zExpr})))`;
    return `tanh(${zExpr})`;
  }

  linearDesmosFromOutputLayer(minWeight = 0.005) {
    if (this.layers.length === 0) return null;
    const inputs = this.featureExprsForDesmos();
    const last = this.layers[this.layers.length - 1];
    const yTerms = [];
    for (let j = 0; j < last.W[0].length; j++) {
      if (Math.abs(last.W[0][j]) < minWeight) continue;
      const w = round3(last.W[0][j]);
      if (w === null) continue;
      yTerms.push(inputs[j] === "1" ? `${w}` : `${w}*${inputs[j]}`);
    }
    const bias = round3(last.b[0]);
    if (bias !== null && Math.abs(bias) >= minWeight) yTerms.push(`${bias}`);
    return yTerms.length > 0 ? `y=${yTerms.join("+")}` : "y=0";
  }

  mlpDesmosText(minWeight = 0.005) {
    if (this.layers.length === 0) return null;

    let inputs = this.featureExprsForDesmos();
    for (let l = 0; l < this.layers.length - 1; l++) {
      const zExprs = this.linearComboDesmos(this.layers[l].W, this.layers[l].b, inputs, minWeight);
      inputs = zExprs.map((z) => this.activationDesmos(z));
    }

    const last = this.layers[this.layers.length - 1];
    const yTerms = [];
    for (let j = 0; j < last.W[0].length; j++) {
      if (Math.abs(last.W[0][j]) < minWeight) continue;
      const w = round3(last.W[0][j]);
      if (w === null) continue;
      yTerms.push(`${w}*${inputs[j]}`);
    }
    const bias = round3(last.b[0]);
    if (bias !== null && Math.abs(bias) >= minWeight) yTerms.push(`${bias}`);

    return yTerms.length > 0 ? `y=${yTerms.join("+")}` : "y=0";
  }

  toFormulaText() {
    return this.architectureText();
  }

  toDesmosText(minWeight = 0.005) {
    if (this.numHiddenLayers === 0) return this.linearDesmosFromOutputLayer(minWeight);
    return this.mlpDesmosText(minWeight);
  }
}

class TaylorNetwork extends FeatureMlpNetwork {
  constructor(order, numHiddenLayers, hiddenSize, activation = "tanh") {
    super(numHiddenLayers, hiddenSize, activation);
    this.order = order;
  }

  networkName() {
    return "Taylor-MLP";
  }

  inputSize() {
    return this.order + 1;
  }

  features(x) {
    const t = this.scaledX(x);
    const phi = [1];
    let power = t;
    for (let i = 1; i <= this.order; i++) {
      phi.push(power);
      power *= t;
    }
    return phi;
  }

  featureExprsForDesmos() {
    const t = this.desmosTVar();
    const exprs = ["1", t];
    for (let i = 2; i <= this.order; i++) {
      exprs.push(`(${t})^${i}`);
    }
    return exprs;
  }

  expandedXPowers(minWeight = 0.005) {
    if (this.numHiddenLayers !== 0 || this.layers.length === 0) return null;

    const coeffs = new Array(this.order + 1).fill(0);
    const { W, b } = this.layers[0];
    coeffs[0] += b[0];

    for (let i = 0; i <= this.order; i++) {
      const wi = W[0][i];
      if (!Number.isFinite(wi) || Math.abs(wi) < minWeight) continue;
      const invS = 1 / this.scale;
      for (let j = 0; j <= i; j++) {
        coeffs[j] += wi * Math.pow(invS, i) * binomial(i, j) * Math.pow(-this.center, i - j);
      }
    }

    return coeffs;
  }

  xPolynomialDesmosText(coeffs, minWeight = 0.005) {
    const parts = [];
    for (let i = 0; i < coeffs.length; i++) {
      if (!Number.isFinite(coeffs[i]) || Math.abs(coeffs[i]) < minWeight) continue;
      const c = round3(coeffs[i]);
      if (c === null) continue;
      if (i === 0) parts.push(`${c}`);
      else if (i === 1) parts.push(`${c}*x`);
      else parts.push(`${c}*x^${i}`);
    }
    return parts.length > 0 ? `y=${parts.join("+")}` : "y=0";
  }

  toFormulaText(minWeight = 0.005) {
    if (this.numHiddenLayers === 0) {
      const coeffs = this.expandedXPowers(minWeight);
      if (coeffs) {
        const parts = [];
        for (let i = 0; i < coeffs.length; i++) {
          if (Math.abs(coeffs[i]) < minWeight) continue;
          const c = round3(coeffs[i]);
          if (c === null) continue;
          if (i === 0) parts.push(`${c}`);
          else if (i === 1) parts.push(`${c}·x`);
          else parts.push(`${c}·x^${i}`);
        }
        return parts.length > 0 ? `y = ${parts.join(" + ")}` : "y = 0";
      }
    }
    return this.architectureText();
  }

  toDesmosText(minWeight = 0.005) {
    if (this.layers.length === 0) return null;

    if (this.numHiddenLayers === 0) {
      const coeffs = this.expandedXPowers(minWeight);
      if (coeffs) return this.xPolynomialDesmosText(coeffs, minWeight);
    }

    return this.mlpDesmosText(minWeight);
  }
}

class FourierNetwork extends FeatureMlpNetwork {
  constructor(harmonics, numHiddenLayers, hiddenSize, activation = "tanh") {
    super(numHiddenLayers, hiddenSize, activation);
    this.harmonics = harmonics;
  }

  networkName() {
    return "Fourier-MLP";
  }

  inputSize() {
    return 1 + 2 * this.harmonics;
  }

  harmonicAngle(x, k) {
    return k * FOURIER_PI * this.scaledX(x);
  }

  features(x) {
    const phi = [1];
    for (let k = 1; k <= this.harmonics; k++) {
      const angle = this.harmonicAngle(x, k);
      phi.push(Math.cos(angle), Math.sin(angle));
    }
    return phi;
  }

  fourierAngleDesmos(k) {
    const t = this.desmosTVar();
    const coeff = round4(k * FOURIER_PI);
    if (coeff === null || Math.abs(coeff) < 1e-9) return t;
    return `${coeff}*(${t})`;
  }

  featureExprsForDesmos() {
    const exprs = ["1"];
    for (let k = 1; k <= this.harmonics; k++) {
      const ang = this.fourierAngleDesmos(k);
      exprs.push(`cos(${ang})`, `sin(${ang})`);
    }
    return exprs;
  }

  toFormulaText(minWeight = 0.005) {
    if (this.numHiddenLayers === 0) {
      const text = this.linearDesmosFromOutputLayer(minWeight);
      if (text) return text.replace(/^y=/, "y = ").replace(/\*/g, "·");
    }
    return this.architectureText();
  }
}

function featureMlpSigmoid(z) {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function linearLayer(W, b, input) {
  const out = new Array(W.length);
  for (let o = 0; o < W.length; o++) {
    out[o] = dot(W[o], input) + b[o];
  }
  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  k = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = (result * (n - i)) / (i + 1);
  }
  return result;
}

function normalizeFormula(text) {
  let s = String(text);
  let prev;
  do {
    prev = s;
    s = s.replace(/-\s*-/g, "+");
    s = s.replace(/\+\s*\+/g, "+");
    s = s.replace(/\+\s*-/g, "-");
    s = s.replace(/-\s*\+/g, "-");
  } while (s !== prev);
  return s;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function round3(value) {
  if (!Number.isFinite(value)) return null;
  return Math.round(value * 1000) / 1000;
}

function round4(value) {
  if (!Number.isFinite(value)) return null;
  return Math.round(value * 10000) / 10000;
}

function uniformX0(numNeurons, xMin, xMax) {
  const x0 = [];
  for (let i = 0; i < numNeurons; i++) {
    const t = numNeurons === 1 ? 0.5 : i / (numNeurons - 1);
    x0.push(xMin + t * (xMax - xMin));
  }
  return x0;
}

function sampleYFromData(data, x) {
  if (data.length === 0) return 0;
  if (x <= data[0].x) return data[0].y;
  if (x >= data[data.length - 1].x) return data[data.length - 1].y;

  for (let i = 0; i < data.length - 1; i++) {
    const a = data[i];
    const b = data[i + 1];
    if (x >= a.x && x <= b.x) {
      if (Math.abs(b.x - a.x) < 1e-9) return (a.y + b.y) / 2;
      const t = (x - a.x) / (b.x - a.x);
      return a.y + t * (b.y - a.y);
    }
  }

  return data[data.length - 1].y;
}
