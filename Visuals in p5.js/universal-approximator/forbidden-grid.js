/* Compact collision reader for the row-run grid returned by Python capture. */

class ForbiddenGrid {
  constructor(payload) {
    this.payload = payload;
    this.cols = Number(payload.cols);
    this.rows = Number(payload.rows);
    this.cellPx = Number(payload.cell_px);
    this.imageWidth = Number(payload.image_width);
    this.imageHeight = Number(payload.image_height);
    this.rowsRle = payload.rows_rle || [];
    this.cells = new Uint8Array(this.cols * this.rows);
    this.decodeRuns();
  }

  decodeRuns() {
    for (let row = 0; row < this.rowsRle.length && row < this.rows; row++) {
      for (const run of this.rowsRle[row]) {
        const start = Math.max(0, Number(run[0]));
        const end = Math.min(this.cols, start + Number(run[1]));
        for (let col = start; col < end; col++) {
          this.cells[row * this.cols + col] = 1;
        }
      }
    }
  }

  isOccupied(col, row) {
    if (col < 0 || row < 0 || col >= this.cols || row >= this.rows) return false;
    return this.cells[row * this.cols + col] === 1;
  }

  worldToGrid(x, y) {
    // Matches Python field_to_game(): both axes use field image width as scale.
    const pixelX = ((x + 25) * this.imageWidth) / 50;
    const pixelY = ((15 - y) * this.imageWidth) / 50;
    return {
      x: pixelX / this.cellPx,
      y: pixelY / this.cellPx,
    };
  }

  pathPenalty(path, ignoredPoints = [], ignoreRadius = 0.55) {
    if (!path || path.length < 2) return 0;
    const visited = new Set();
    let collisions = 0;

    for (let index = 0; index < path.length - 1; index++) {
      const left = path[index];
      const right = path[index + 1];
      const gridLeft = this.worldToGrid(left.x, left.y);
      const gridRight = this.worldToGrid(right.x, right.y);
      const span = Math.max(
        Math.abs(gridRight.x - gridLeft.x),
        Math.abs(gridRight.y - gridLeft.y)
      );
      // Half-cell sampling plus a safety-dilated source mask avoids gaps
      // without paying for per-pixel collision checks.
      const steps = Math.max(1, Math.ceil(span * 2));

      for (let step = 0; step <= steps; step++) {
        const t = step / steps;
        const x = left.x + (right.x - left.x) * t;
        const y = left.y + (right.y - left.y) * t;
        if (ForbiddenGrid.nearIgnoredPoint(x, y, ignoredPoints, ignoreRadius)) continue;

        const grid = this.worldToGrid(x, y);
        const col = Math.floor(grid.x);
        const row = Math.floor(grid.y);
        const cellId = row * this.cols + col;
        if (!visited.has(cellId) && this.isOccupied(col, row)) {
          visited.add(cellId);
          collisions += 1;
        }
      }
    }

    // Zero is safe. Any collision is lexicographically worse than every safe
    // agent; the fractional part still guides unsafe agents toward an exit.
    return collisions === 0 ? 0 : 1 + collisions / 1000;
  }

  static nearIgnoredPoint(x, y, points, radius) {
    const radiusSquared = radius * radius;
    return points.some((point) => {
      const dx = x - point.x;
      const dy = y - point.y;
      return dx * dx + dy * dy <= radiusSquared;
    });
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ForbiddenGrid };
}
