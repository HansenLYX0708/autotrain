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
