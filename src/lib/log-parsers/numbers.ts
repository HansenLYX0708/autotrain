/**
 * Numeric-token helpers shared by every log parser.
 *
 * Training frameworks print non-finite values straight to stdout when a run
 * diverges or a metric has no support in the batch:
 *   loss: nan          loss_iou: -nan        mIoU: nan
 *   [0.9952 nan 0.6496]                     image_auroc: nan
 *
 * The parsers used to run `[\d.]+`-style captures over those lines, which did
 * one of three bad things: matched nothing (value silently dropped), matched
 * just the `-` of `-nan` (`parseFloat("-")` → NaN leaked into the record and
 * from there into the DB / charts), or — for numpy class arrays — rejected the
 * whole array, which closed the EVAL block early and lost the per-class
 * metrics *and* the trailing best-checkpoint line.
 *
 * Every numeric read goes through this module so a non-finite token becomes
 * `{ value: null, nonFinite: true }` instead of a silent drop or a NaN leak.
 *
 * This module is bundled for the browser too: no `fs`, and no regex
 * lookbehind (not supported in all target runtimes).
 */

/** Signed finite number: `12`, `-0.3`, `.5`, `1.5e-05`, `+2E3`. */
const FINITE = '[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?'

/**
 * Non-finite tokens as they appear in training logs. `infinity` is listed
 * before `inf` so the longer alternative wins; `nan` likewise precedes `na`
 * inside the alternation order below (they live in separate alternatives, and
 * the trailing `(?![\\w.])` guard keeps `na` from matching the head of `nan`
 * or words like `name` / `national`).
 */
const NONFINITE = '[-+]?(?:nan|infinity|inf)|n\\/a|na|none'

/**
 * A single printable metric value — finite number or a recognised non-finite
 * token. Use with `i` and, when matching inside a line, append `(?![\\w.])`
 * (done for you in {@link NUMERIC_TOKEN_BOUNDED}).
 */
export const NUMERIC_TOKEN = `(?:${FINITE}|${NONFINITE})`

/** NUMERIC_TOKEN with the trailing-garbage guard already applied. */
const NUMERIC_TOKEN_BOUNDED = `${NUMERIC_TOKEN}(?![\\w.])`

const FINITE_TOKEN_RE = new RegExp(`^${FINITE}$`, 'i')
const NUMERIC_TOKEN_RE = new RegExp(`^${NUMERIC_TOKEN}$`, 'i')

/** Token allowed inside a numpy-style `[a b c]` class array: finite, nan, inf. */
const ARRAY_TOKEN = `(?:${FINITE}|[-+]?(?:nan|infinity|inf))`
const ARRAY_TOKEN_RE = new RegExp(`^${ARRAY_TOKEN}$`, 'i')

export interface NumericResult {
  /** Finite value, or `null` when the token was non-finite. */
  value: number | null
  /** True when the printed token was NaN/Inf/NA/None rather than a number. */
  nonFinite: boolean
}

/**
 * Interpret one token already extracted from a line. Finite tokens yield
 * `{ value, false }`; any non-finite token — or a finite-looking token that
 * still parses to NaN/±Infinity — yields `{ value: null, nonFinite: true }`.
 */
export function parseNumericToken(token: string): NumericResult {
  const trimmed = token.trim()
  if (FINITE_TOKEN_RE.test(trimmed)) {
    const value = parseFloat(trimmed)
    if (Number.isFinite(value)) return { value, nonFinite: false }
  }
  return { value: null, nonFinite: true }
}

/**
 * Read `key: <value>` / `key=<value>` from a line.
 *
 * `key` is matched as a whole key: it must be preceded by start-of-line,
 * whitespace, `,`, `|`, `[` or `(` — so `lr` never fires inside
 * `learning_rate` — and followed by `\s*[:=]\s*`, so `loss` never fires on
 * `loss_cls:`. Returns `null` when the key is absent entirely (distinguishing
 * "not printed" from "printed but NaN").
 */
export function readMetric(line: string, key: string): NumericResult | null {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const re = new RegExp(`(?:^|[\\s,|\\[(])${escaped}\\s*[:=]\\s*(${NUMERIC_TOKEN_BOUNDED})`, 'i')
  const m = line.match(re)
  if (!m) return null
  const r = parseNumericToken(m[1])
  // parseNumericToken is definitive for recognised tokens; a value that still
  // isn't finite after matching (can't happen via the regex, but stay safe)
  // is reported non-finite rather than leaking NaN downstream.
  return r
}

/**
 * Records which metric names were printed non-finite so a parser can surface
 * them on `ParsedTrainLog.nonFinite` instead of letting the run look healthy.
 *
 * Usage: `const c = new NonFiniteCollector()`, then `c.read(line, 'loss')`
 * wherever the parser previously ran `parseFloat(regex)`. For aliased keys
 * pass the canonical name: `c.read(line, 'learning_rate', 'lr')`.
 */
export class NonFiniteCollector {
  private hit = new Set<string>()

  /** Read `key` off `line`; absent and finite values behave like plain
   *  parseFloat — only non-finite reads are recorded (under `name`). */
  read(line: string, key: string, name = key): number | null {
    return this.add(readMetric(line, key), name)
  }

  /** Fold an already-obtained readMetric result into the collector. */
  add(result: NumericResult | null, name: string): number | null {
    if (!result) return null
    if (result.nonFinite) this.hit.add(name)
    return result.value
  }

  mark(name: string): void {
    this.hit.add(name)
  }

  /** Canonical names of every non-finite metric seen, or undefined if none. */
  names(): string[] | undefined {
    return this.hit.size > 0 ? [...this.hit] : undefined
  }
}

/**
 * Parse a complete numpy-style `[a b c]` whitespace-separated float array.
 * Returns `null` when the text is not a closed array or contains any token
 * that is not numeric/nan/inf. `nan`/`inf` entries become `null` and set
 * `nonFinite` on the result.
 */
export function parseFloatArray(
  text: string,
): { values: (number | null)[]; nonFinite: boolean } | null {
  const stripped = text.trim()
  if (!stripped.startsWith('[') || !stripped.endsWith(']')) return null
  const inner = stripped.slice(1, -1).trim()
  if (!inner) return null
  const tokens = inner.split(/\s+/)
  if (!tokens.every((tok) => NUMERIC_TOKEN_RE.test(tok))) return null
  let nonFinite = false
  const values = tokens.map((tok) => {
    const r = parseNumericToken(tok)
    if (r.nonFinite) nonFinite = true
    return r.value
  })
  return { values, nonFinite }
}

/**
 * Could `line` be the start (or continuation) of a numpy float array that has
 * not been closed yet? Used to decide whether to keep buffering instead of
 * treating the line as "not part of this EVAL block".
 *
 * Deliberately strict: only numeric/nan/inf tokens and whitespace, plus a
 * single optional leading `[`, so a genuine log line never gets swallowed.
 */
export function isOpenArrayFragment(text: string): boolean {
  const stripped = text.trim()
  if (!stripped || stripped.endsWith(']')) return false
  const body = stripped.startsWith('[') ? stripped.slice(1) : stripped
  if (!body.trim()) return false
  return body
    .trim()
    .split(/\s+/)
    .every((tok) => ARRAY_TOKEN_RE.test(tok))
}
