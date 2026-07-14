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
