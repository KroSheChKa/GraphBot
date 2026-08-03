// Cubic interpolation and cubic B-spline helpers for Draw mode.

class CubicSplineModel {
  constructor(data, boundary = "natural") {
    this.data = data.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
    this.boundary = boundary;
    this.intervals = [];
    this.build();
  }

  build() {
    const n = this.data.length;
    if (n < 2) return;

    const h = [];
    const slopes = [];
    for (let i = 0; i < n - 1; i++) {
      const dx = this.data[i + 1].x - this.data[i].x;
      h.push(Math.max(Math.abs(dx), 1e-9));
      slopes.push((this.data[i + 1].y - this.data[i].y) / h[i]);
    }

    const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
    const rhs = new Array(n).fill(0);
    if (this.boundary === "clamped") {
      matrix[0][0] = 2 * h[0];
      matrix[0][1] = h[0];
      rhs[0] = 6 * slopes[0];
      matrix[n - 1][n - 2] = h[n - 2];
      matrix[n - 1][n - 1] = 2 * h[n - 2];
      rhs[n - 1] = -6 * slopes[slopes.length - 1];
    } else {
      matrix[0][0] = 1;
      matrix[n - 1][n - 1] = 1;
    }

    for (let i = 1; i < n - 1; i++) {
      matrix[i][i - 1] = h[i - 1];
      matrix[i][i] = 2 * (h[i - 1] + h[i]);
      matrix[i][i + 1] = h[i];
      rhs[i] = 6 * (slopes[i] - slopes[i - 1]);
    }

    const second = solveLinearSystem(matrix, rhs);
    for (let i = 0; i < n - 1; i++) {
      const span = h[i];
      const a = this.data[i].y;
      const b = slopes[i] - (span * (2 * second[i] + second[i + 1])) / 6;
      const c = second[i] / 2;
      const d = (second[i + 1] - second[i]) / (6 * span);
      this.intervals.push({ x: this.data[i].x, a, b, c, d, h: span });
    }
  }

  predict(x) {
    if (this.intervals.length === 0) return 0;
    const index = intervalIndex(this.intervals, x);
    const part = x - this.intervals[index].x;
    const item = this.intervals[index];
    return item.a + part * (item.b + part * (item.c + part * item.d));
  }

  toFormulaText(decimals = 14) {
    return piecewiseCubicFormulaText(this.intervals, decimals);
  }
}

class BSplineModel {
  constructor(data, controlPointCount = 12, smoothing = 0) {
    this.data = data.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
    this.controlPointCount = Math.max(4, Math.round(controlPointCount));
    this.smoothing = Math.max(0, Number(smoothing) || 0);
    this.degree = 3;
    this.knots = [];
    this.coefficients = [];
    this.intervals = [];
    this.build();
  }

  build() {
    if (this.data.length < 2) return;
    const xMin = this.data[0].x;
    const xMax = this.data[this.data.length - 1].x;
    const count = Math.min(this.controlPointCount, Math.max(4, this.data.length));
    this.controlPointCount = count;

    const internalCount = count - this.degree - 1;
    const spacing = (xMax - xMin) / Math.max(1, internalCount + 1);
    this.knots = [];
    for (let i = 0; i < this.degree + 1; i++) this.knots.push(xMin);
    for (let i = 1; i <= internalCount; i++) this.knots.push(xMin + i * spacing);
    for (let i = 0; i < this.degree + 1; i++) this.knots.push(xMax);

    const rows = this.data.map((point) =>
      Array.from({ length: count }, (_, index) =>
        bsplineBasis(index, this.degree, point.x, this.knots)
      )
    );
    const normal = Array.from({ length: count }, () => new Array(count).fill(0));
    const rhs = new Array(count).fill(0);
    for (let row = 0; row < rows.length; row++) {
      for (let i = 0; i < count; i++) {
        rhs[i] += rows[row][i] * this.data[row].y;
        for (let j = 0; j < count; j++) normal[i][j] += rows[row][i] * rows[row][j];
      }
    }

    // λ=0 is the interpolation/least-squares limit. A tiny diagonal term
    // keeps the solve stable for sparse or nearly duplicate input points.
    const lambda = this.smoothing * 0.1 + 1e-8;
    for (let i = 0; i < count; i++) normal[i][i] += lambda;
    this.coefficients = solveLinearSystem(normal, rhs);
    this.intervals = this.buildPiecewiseIntervals();
  }

  predict(x) {
    if (this.coefficients.length === 0) return 0;
    let value = 0;
    for (let i = 0; i < this.coefficients.length; i++) {
      value += this.coefficients[i] * bsplineBasis(i, this.degree, x, this.knots);
    }
    return value;
  }

  buildPiecewiseIntervals() {
    const uniqueKnots = [];
    for (const knot of this.knots) {
      if (uniqueKnots.length === 0 || Math.abs(knot - uniqueKnots[uniqueKnots.length - 1]) > 1e-9) {
        uniqueKnots.push(knot);
      }
    }
    const intervals = [];
    const tValues = [0, 1 / 3, 2 / 3, 1];
    for (let i = 0; i < uniqueKnots.length - 1; i++) {
      const x0 = uniqueKnots[i];
      const h = uniqueKnots[i + 1] - x0;
      if (h <= 1e-9) continue;
      const values = tValues.map((t) => this.predict(x0 + t * h));
      const normalized = solveLinearSystem(
        tValues.map((t) => [1, t, t * t, t * t * t]),
        values
      );
      intervals.push({
        x: x0,
        a: normalized[0],
        b: normalized[1] / h,
        c: normalized[2] / (h * h),
        d: normalized[3] / (h * h * h),
        h,
      });
    }
    return intervals;
  }

  toFormulaText(decimals = 14) {
    return piecewiseCubicFormulaText(this.intervals, decimals);
  }
}

function intervalIndex(intervals, x) {
  if (x <= intervals[0].x) return 0;
  for (let i = intervals.length - 1; i >= 0; i--) {
    if (x >= intervals[i].x) return i;
  }
  return 0;
}

function solveLinearSystem(inputMatrix, inputRhs) {
  const matrix = inputMatrix.map((row, rowIndex) => [
    ...row.map((value) => (Number.isFinite(value) ? value : 0)),
    Number.isFinite(inputRhs[rowIndex]) ? inputRhs[rowIndex] : 0,
  ]);
  const n = matrix.length;
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    if (Math.abs(matrix[pivot][column]) < 1e-12) {
      matrix[pivot][column] = 1e-12;
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const divisor = matrix[column][column];
    for (let j = column; j <= n; j++) matrix[column][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === column) continue;
      const factor = matrix[row][column];
      if (Math.abs(factor) < 1e-15) continue;
      for (let j = column; j <= n; j++) matrix[row][j] -= factor * matrix[column][j];
    }
  }
  return matrix.map((row) => (Number.isFinite(row[n]) ? row[n] : 0));
}

function bsplineBasis(index, degree, x, knots) {
  const last = knots[knots.length - 1];
  if (x === last) {
    return degree > 0 && index === knots.length - degree - 2 ? 1 : 0;
  }
  if (degree === 0) {
    return x >= knots[index] && x < knots[index + 1] ? 1 : 0;
  }
  const leftDenominator = knots[index + degree] - knots[index];
  const rightDenominator = knots[index + degree + 1] - knots[index + 1];
  const left = leftDenominator > 0
    ? ((x - knots[index]) / leftDenominator) * bsplineBasis(index, degree - 1, x, knots)
    : 0;
  const right = rightDenominator > 0
    ? ((knots[index + degree + 1] - x) / rightDenominator) * bsplineBasis(index + 1, degree - 1, x, knots)
    : 0;
  return left + right;
}

function piecewiseCubicFormulaText(intervals, decimals = 14) {
  if (!intervals || intervals.length === 0) return null;
  const terms = [];
  const first = intervals[0];
  const baseX = splineNumber(first.x, decimals);
  terms.push(splinePolynomial(first.a, first.b, first.c, first.d, `x - ${baseX}`, decimals));

  for (let i = 1; i < intervals.length; i++) {
    const jump = intervals[i].d - intervals[i - 1].d;
    if (Math.abs(jump) < 10 ** (-(decimals + 1))) continue;
    const knot = splineNumber(intervals[i].x, decimals);
    const positivePart = `((x - ${knot}) + abs(x - ${knot}))/2`;
    terms.push(`${splineNumber(jump, decimals)}*(${positivePart})^3`);
  }
  return `y=${normalizeSplineFormula(terms.join(" + "))}`;
}

function splinePolynomial(a, b, c, d, variable, decimals) {
  const parts = [splineNumber(a, decimals)];
  if (Math.abs(b) >= 10 ** (-(decimals + 1))) parts.push(`${splineNumber(b, decimals)}*(${variable})`);
  if (Math.abs(c) >= 10 ** (-(decimals + 1))) parts.push(`${splineNumber(c, decimals)}*(${variable})^2`);
  if (Math.abs(d) >= 10 ** (-(decimals + 1))) parts.push(`${splineNumber(d, decimals)}*(${variable})^3`);
  return parts.join(" + ");
}

function splineNumber(value, decimals = 14) {
  if (!Number.isFinite(value)) return "0";
  const safe = Math.abs(value) < 0.5 * 10 ** -decimals ? 0 : value;
  return Number(safe.toFixed(decimals)).toString();
}

function normalizeSplineFormula(text) {
  return String(text)
    .replace(/\+\s*-/g, "-")
    .replace(/-\s*-/g, "+");
}
