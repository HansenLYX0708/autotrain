/**
 * PaddleSeg list-file (train.txt / val.txt) utilities.
 *
 * Each line pairs an image and its label mask, canonically written as
 *     JPEGImages/<name>.<ext> Annotations/<name>.png
 * separated by whitespace.  Because PaddleSeg (and older Python readers)
 * `str.split()` on any whitespace, filenames with embedded spaces would break
 * both training and preview.  The `labelme-to-paddleseg` converter sanitizes
 * filenames on write, but existing datasets converted before that fix, or
 * datasets curated by hand, may still contain spaces.  This helper anchors
 * on the mask suffix (`Annotations/*.png`) so those datasets still preview
 * correctly even if training would need a re-conversion.
 */

export interface ListEntry {
  imageRel: string;
  maskRel: string;
}

/**
 * Parse one line of a PaddleSeg list file. Returns null when the line does
 * not look like a valid "image mask" pair.
 */
export function parseListLine(line: string): ListEntry | null {
  const trimmed = line.trim();
  if (!trimmed) return null;

  // Preferred: anchor on the mask token so filenames with spaces still parse.
  // The mask always lives under `Annotations/` (our writer's convention) or
  // similar and ends in `.png`.
  const anchored = trimmed.match(/^(.+?)\s+((?:Annotations|annotations|masks|Masks)\/[^\s].*\.png)\s*$/);
  if (anchored) return { imageRel: anchored[1].trim(), maskRel: anchored[2].trim() };

  // Fallback: assume the LAST whitespace-delimited token is the mask, and
  // everything before it is the image (which may contain spaces).
  const idx = trimmed.lastIndexOf(' ');
  const tabIdx = trimmed.lastIndexOf('\t');
  const cut = Math.max(idx, tabIdx);
  if (cut > 0) {
    return {
      imageRel: trimmed.slice(0, cut).trim(),
      maskRel: trimmed.slice(cut + 1).trim(),
    };
  }
  return null;
}

/** Parse the full file contents into a list of `{imageRel, maskRel}` pairs. */
export function parseListFile(content: string): ListEntry[] {
  return content
    .split(/\r?\n/)
    .map(parseListLine)
    .filter((e): e is ListEntry => e !== null);
}
