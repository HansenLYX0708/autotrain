/**
 * PaddleSeg pseudo-color map utilities.
 *
 * PaddleSeg writes pseudo-color masks as paletted PNGs whose palette is
 * produced by `get_color_map_list` (the PASCAL VOC color map with the leading
 * black entry dropped). The class index of each mask pixel therefore maps to a
 * fixed RGB color: class 0 -> [128,0,0], class 1 -> [0,128,0], class 2 ->
 * [128,128,0], class 3 -> [0,0,128], ...
 *
 * Reproducing the map here lets the API (pixel statistics) and the UI (legend
 * + overlay) agree on colors without decoding each PNG's palette.
 */

export type RGB = [number, number, number];

/** Returns the RGB color for class indices 0..numClasses-1. */
export function getSegColorMap(numClasses: number): RGB[] {
  const count = Math.max(1, numClasses);
  const n = count + 1;
  const cm = new Array(n * 3).fill(0);
  for (let i = 0; i < n; i++) {
    let lab = i;
    let j = 0;
    while (lab) {
      cm[i * 3] |= ((lab >> 0) & 1) << (7 - j);
      cm[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j);
      cm[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j);
      j += 1;
      lab >>= 3;
    }
  }
  // Drop the first entry ([0,0,0]) to match PaddleSeg's `color_map[3:]`.
  const shifted = cm.slice(3);
  const out: RGB[] = [];
  for (let k = 0; k < count; k++) {
    out.push([shifted[k * 3] || 0, shifted[k * 3 + 1] || 0, shifted[k * 3 + 2] || 0]);
  }
  return out;
}

/** Builds an "r,g,b" -> classIndex lookup for the given class count. */
export function buildColorIndex(numClasses: number): Map<string, number> {
  const map = new Map<string, number>();
  getSegColorMap(numClasses).forEach((c, i) => map.set(`${c[0]},${c[1]},${c[2]}`, i));
  return map;
}

/** Formats an RGB tuple as a CSS color string. */
export function rgbToCss(rgb: RGB | undefined): string {
  if (!rgb) return 'rgb(0, 0, 0)';
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
