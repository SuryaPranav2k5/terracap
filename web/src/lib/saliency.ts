/**
 * Real visual saliency and model spatial attention renderer.
 */

export interface SaliencyMap {
  w: number;
  h: number;
  sharp: Float32Array; // normalized 0..1
  soft: Float32Array;
}

function boxBlur(
  src: Float32Array,
  w: number,
  h: number,
  radius: number
): Float32Array {
  if (radius < 1) return src.slice();
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const win = radius * 2 + 1;

  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = -radius; x <= radius; x++) {
      acc += src[row + Math.min(w - 1, Math.max(0, x))];
    }
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / win;
      const add = Math.min(w - 1, x + radius + 1);
      const sub = Math.max(0, x - radius);
      acc += src[row + add] - src[row + sub];
    }
  }

  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -radius; y <= radius; y++) {
      acc += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    }
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / win;
      const add = Math.min(h - 1, y + radius + 1);
      const sub = Math.max(0, y - radius);
      acc += tmp[add * w + x] - tmp[sub * w + x];
    }
  }
  return out;
}

function normalizePercentile(map: Float32Array): void {
  const sorted = Float32Array.from(map).sort();
  const n = sorted.length;
  const lo = sorted[Math.floor(n * 0.02)];
  const hi = sorted[Math.floor(n * 0.985)];
  const range = Math.max(1e-6, hi - lo);
  for (let i = 0; i < map.length; i++) {
    map[i] = Math.min(1, Math.max(0, (map[i] - lo) / range));
  }
}

export function computeSaliency(
  source: HTMLImageElement | HTMLCanvasElement,
  maxDim = 160
): SaliencyMap | null {
  const sw =
    source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const sh =
    source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  if (!sw || !sh) return null;

  const scale = Math.min(1, maxDim / Math.max(sw, sh));
  const w = Math.max(12, Math.round(sw * scale));
  const h = Math.max(12, Math.round(sh * scale));

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, w, h);

  let data: Uint8ClampedArray;
  try {
    data = ctx.getImageData(0, 0, w, h).data;
  } catch {
    return null;
  }

  const n = w * h;
  const I = new Float32Array(n);
  const RG = new Float32Array(n);
  const BY = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const r = data[o];
    const g = data[o + 1];
    const b = data[o + 2];
    I[i] = (r + g + b) / 765;
    RG[i] = Math.max(0, (r - g) / 255);
    BY[i] = Math.max(0, (b - (r + g) / 2) / 255);
  }

  const sharp = new Float32Array(n);
  const scales = [2, 5, 11];
  for (const r of scales) {
    const blurredI = boxBlur(I, w, h, r);
    const blurredRG = boxBlur(RG, w, h, r);
    const blurredBY = boxBlur(BY, w, h, r);
    for (let i = 0; i < n; i++) {
      const dI = Math.abs(I[i] - blurredI[i]);
      const dC =
        Math.abs(RG[i] - blurredRG[i]) + Math.abs(BY[i] - blurredBY[i]);
      sharp[i] += dI * 0.6 + dC * 0.4;
    }
  }

  normalizePercentile(sharp);
  const soft = boxBlur(sharp, w, h, Math.max(3, Math.round(w / 14)));
  normalizePercentile(soft);

  return { w, h, sharp, soft };
}

/** Classic jet colormap: v in [0,1] -> [r,g,b] */
export function jetColor(v: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, v));
  const r = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(4 * t - 3))));
  const g = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(4 * t - 2))));
  const b = Math.min(255, Math.max(0, 255 * (1.5 - Math.abs(4 * t - 1))));
  return [Math.round(r), Math.round(g), Math.round(b)];
}

export interface AttentionRenderOptions {
  step: number;
  totalSteps: number;
  maxAlpha?: number;
}

function computeFit(cw: number, ch: number, image: HTMLImageElement | HTMLCanvasElement) {
  const imgW = ("naturalWidth" in image && image.naturalWidth) ? image.naturalWidth : image.width || 1;
  const imgH = ("naturalHeight" in image && image.naturalHeight) ? image.naturalHeight : image.height || 1;
  const imgAspect = imgW / imgH;
  const canvasAspect = cw / ch;

  let drawW = cw;
  let drawH = ch;
  let offsetX = 0;
  let offsetY = 0;

  if (canvasAspect > imgAspect) {
    drawH = ch;
    drawW = Math.round(ch * imgAspect);
    offsetX = Math.round((cw - drawW) / 2);
  } else {
    drawW = cw;
    drawH = Math.round(cw / imgAspect);
    offsetY = Math.round((ch - drawH) / 2);
  }

  return { drawW, drawH, offsetX, offsetY };
}

/**
 * Draws the real 8x8 PyTorch spatial attention grid over the base image
 * using bilinear interpolation and a thermal jet overlay, preserving natural aspect ratio.
 */
export function renderRealAttentionGrid(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement | HTMLCanvasElement,
  grid: number[][],
  opts?: { maxAlpha?: number }
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || !grid || grid.length === 0) return;

  const cw = canvas.width;
  const ch = canvas.height;
  const { drawW, drawH, offsetX, offsetY } = computeFit(cw, ch, image);

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(image, offsetX, offsetY, drawW, drawH);

  const gh = grid.length;
  const gw = grid[0]?.length || 8;
  const maxAlpha = opts?.maxAlpha ?? 0.68;

  // Find min and max for robust normalization
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let r = 0; r < gh; r++) {
    for (let c = 0; c < gw; c++) {
      const v = grid[r][c];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const range = Math.max(1e-7, maxVal - minVal);

  const overlay = ctx.createImageData(drawW, drawH);
  const out = overlay.data;

  // Bilinear interpolation across (drawW, drawH)
  for (let y = 0; y < drawH; y++) {
    const gy = (y / drawH) * (gh - 1);
    const r0 = Math.floor(gy);
    const r1 = Math.min(gh - 1, r0 + 1);
    const fy = gy - r0;

    for (let x = 0; x < drawW; x++) {
      const gx = (x / drawW) * (gw - 1);
      const c0 = Math.floor(gx);
      const c1 = Math.min(gw - 1, c0 + 1);
      const fx = gx - c0;

      const v00 = grid[r0][c0];
      const v01 = grid[r0][c1];
      const v10 = grid[r1][c0];
      const v11 = grid[r1][c1];

      const vTop = v00 * (1 - fx) + v01 * fx;
      const vBot = v10 * (1 - fx) + v11 * fx;
      const rawVal = vTop * (1 - fy) + vBot * fy;

      const normalized = Math.min(1, Math.max(0, (rawVal - minVal) / range));
      const [r, g, b] = jetColor(normalized);

      const di = (y * drawW + x) * 4;
      out[di] = r;
      out[di + 1] = g;
      out[di + 2] = b;
      out[di + 3] = Math.round(255 * normalized * maxAlpha);
    }
  }

  const tmp = document.createElement("canvas");
  tmp.width = drawW;
  tmp.height = drawH;
  tmp.getContext("2d")?.putImageData(overlay, 0, 0);
  ctx.drawImage(tmp, offsetX, offsetY);
}

/** Fallback saliency renderer preserving aspect ratio */
export function renderAttention(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement | HTMLCanvasElement,
  saliency: SaliencyMap,
  opts: AttentionRenderOptions
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { step, totalSteps, maxAlpha = 0.7 } = opts;

  const cw = canvas.width;
  const ch = canvas.height;
  const { drawW, drawH, offsetX, offsetY } = computeFit(cw, ch, image);

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(image, offsetX, offsetY, drawW, drawH);

  const focus = totalSteps > 1 ? step / (totalSteps - 1) : 1;
  const { w, h, sharp, soft } = saliency;
  const overlay = ctx.createImageData(drawW, drawH);
  const out = overlay.data;

  for (let y = 0; y < drawH; y++) {
    const sy = Math.min(h - 1, Math.round((y / drawH) * h));
    for (let x = 0; x < drawW; x++) {
      const sx = Math.min(w - 1, Math.round((x / drawW) * w));
      const si = sy * w + sx;
      const blended = soft[si] * (1 - focus) + sharp[si] * focus;
      const [r, g, b] = jetColor(blended);
      const di = (y * drawW + x) * 4;
      out[di] = r;
      out[di + 1] = g;
      out[di + 2] = b;
      out[di + 3] = Math.round(255 * blended * maxAlpha);
    }
  }

  const tmp = document.createElement("canvas");
  tmp.width = drawW;
  tmp.height = drawH;
  tmp.getContext("2d")?.putImageData(overlay, 0, 0);
  ctx.drawImage(tmp, offsetX, offsetY);
}
