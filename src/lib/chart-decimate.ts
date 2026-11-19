/**
 * Min/max bucket decimation for chart series.
 *
 * A long run produces tens of thousands of TrainingLog rows; rendering them
 * all makes the monitoring charts unusably slow and visually noisy. Naive
 * downsampling hides exactly what matters (loss spikes), so each bucket keeps
 * its minimum- and maximum-valued rows instead — peaks survive at the cost of
 * at most 2 points per bucket. Pure and shared so the check script can assert
 * the bucket behaviour directly.
 */
export function decimateMinMax<T>(
  rows: T[],
  value: (r: T) => number | null,
  maxPoints: number,
): T[] {
  if (rows.length <= maxPoints || maxPoints <= 0) return rows;

  const bucketCount = Math.max(1, Math.floor(maxPoints / 2));
  const bucketSize = rows.length / bucketCount;
  const keep = new Set<number>();

  for (let b = 0; b < bucketCount; b++) {
    const lo = Math.floor(b * bucketSize);
    const hi = Math.min(rows.length, Math.floor((b + 1) * bucketSize));
    let minIdx = -1;
    let maxIdx = -1;
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = lo; i < hi; i++) {
      const v = value(rows[i]);
      if (v === null || !Number.isFinite(v)) continue;
      if (v < minVal) { minVal = v; minIdx = i; }
      if (v > maxVal) { maxVal = v; maxIdx = i; }
    }
    // Buckets where every row is null-valued still contribute their first row
    // so gaps stay visible rather than collapsing to nothing.
    keep.add(minIdx === -1 ? lo : minIdx);
    if (maxIdx !== -1) keep.add(maxIdx);
  }

  // First and last rows are always kept so the curve's endpoints are exact.
  keep.add(0);
  keep.add(rows.length - 1);

  return [...keep].sort((a, b) => a - b).map((i) => rows[i]);
}

export type ChartXUnit = 'iter' | 'epoch' | 'point'

/**
 * Pick a monotonically increasing numeric x value for a log row so a chart's
 * XAxis can run type="number" and read start-to-end. Data-driven, not
 * framework-driven:
 *
 * - `iteration` already non-decreasing (segmentation/anomaly steps) → x =
 *   iteration, unit "iter".
 * - `iteration` restarts per epoch (epoch-counting trainers) → x = fractional
 *   epoch: `epoch - 1 + iteration / totalIter` for 1-based logs (so a
 *   200-epoch run ends at ~200), `epoch + iteration / totalIter` for 0-based.
 * - Any row missing `totalIter` → x = row index, unit "point".
 *
 * Compute x BEFORE decimateMinMax so the axis domain spans the full run.
 */
export function chartXAxis<T extends { epoch: number; iteration: number; totalIter: number | null }>(
  rows: T[],
): { unit: ChartXUnit; epochBase: number; x: (r: T, i: number) => number } {
  let monotonic = true
  let minEpoch = Infinity
  let hasTotalIter = true
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]
    if (i > 0 && r.iteration < rows[i - 1].iteration) monotonic = false
    if (r.epoch < minEpoch) minEpoch = r.epoch
    if (!r.totalIter) hasTotalIter = false
  }
  if (monotonic) return { unit: 'iter', epochBase: 0, x: (r) => r.iteration }
  if (!hasTotalIter) return { unit: 'point', epochBase: 0, x: (_r, i) => i }
  const oneBased = minEpoch >= 1
  return {
    unit: 'epoch',
    epochBase: oneBased ? 1 : 0,
    x: (r) => (oneBased ? r.epoch - 1 : r.epoch) + r.iteration / (r.totalIter as number),
  }
}
