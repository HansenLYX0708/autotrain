# AGENTS.md

Project knowledge for anyone (human or agent) working on this repo.

## What this is

A Next.js 16 (App Router, Turbopack) + Prisma/SQLite web app that drives
deep-learning training. A "job" is assembled from three independently authored
YAML configs and run as a child process:

```
Dataset config  ─┐
Training config ─┼─► deep-merged YAML ─► tools/train.py ─► stdout parsed into TrainingLog rows
Model config    ─┘
```

Supported frameworks (`project.framework` selects one):

| Framework | Runtime | Task | Repo | Trains by |
|---|---|---|---|---|
| `PaddleDetection` | PaddlePaddle | detection | external, `paddleDetectionPath` | epochs |
| `PaddleClas` | PaddlePaddle | classification | external, `paddleClasPath` | epochs |
| `PaddleSeg` | PaddlePaddle | segmentation | external, `paddleSegPath` | iterations |
| `TorchDet` | PyTorch | detection | bundled `torchtrain/`, `torchPath` | epochs |
| `TorchSeg` | PyTorch | segmentation | bundled `torchtrain/`, `torchPath` | iterations |
| `TorchAnomaly` | PyTorch + anomalib | unsupervised anomaly | bundled `torchtrain/`, `torchPath` | steps |

`src/lib/frameworks.ts` is the **single source of truth**: `FRAMEWORK_META`
declares each framework's repo path field, Python module, CLI dialect, script
names, checkpoint layout, dataset format and step unit. Prefer the predicates
(`isSegmentation`, `isDetection`, `isAnomaly`, `isTorch`, `tracksIterations`,
`frameworkMeta`) over comparing framework names — the string comparisons are what
made adding a framework a whole-codebase grep in the first place.

In particular, **`isSegmentation()` is not a synonym for "counts iterations"**.
It used to be, so it was used for both; `tracksIterations()` (i.e.
`stepUnit === 'iter'`) is the right predicate for anything about progress,
`currentEpoch`/`totalEpochs` or the epoch-named config columns, and
`frameworkMeta(f).datasetFormat` is the right one for dataset layout.

## Commands

```bash
npm run dev          # next dev -p 3000
npm run build        # next build && node copy.js  (output: standalone)
npm run lint         # eslint .
npm run db:push      # prisma db push   <- what this project actually uses
npx tsc --noEmit     # typecheck
```

### Verification before calling a change done

1. `npx tsc --noEmit` — **note the baseline**: there are pre-existing errors in
   `examples/websocket/*`, `mini-services/training-service/*`,
   `src/app/api/dashboard/stats/route.ts`, and `.next/**/validator.ts` (from
   `src/app/api/checkpoints/export/route.ts` returning `Promise<unknown>`).
   Compare counts; do not assume zero.
2. `npx eslint <changed files>`
3. `npx next build`
4. `npx next dev -p 3010` and probe endpoints. `next.config.ts` sets
   `typescript.ignoreBuildErrors: true`, so **a successful build does not mean
   the code typechecks** — always run `tsc` separately.

## Gotchas

### Never edit source files with PowerShell `Set-Content`

On Windows, `Get-Content -Raw` / `Set-Content` (PowerShell 5.1) round-trip through
the **ANSI code page**, which silently corrupts every non-ASCII character in the
file. In practice a UTF-8 sequence loses its trailing byte *and* eats the byte
after it, so `— ` becomes an invalid `E2 80 3F` and `<span>•</span>` becomes
`<span>?/span>` — i.e. broken JSX with a confusing parse error hundreds of lines
from anything you touched. Several files here contain `—`, `·`, `…`, `✓` and `×`
in comments and JSX.

Use an editor tool, or Python with an explicit encoding:

```python
open(path, "w", encoding="utf-8", newline="").write(text)
```

### Database location

The live database lives **outside the repo**:

```
D:\_work\projects\autoTraining\autotrain_db\custom.db
DATABASE_URL=file:D:/_work/projects/autoTraining/autotrain_db/custom.db
```

Backups (including pre-migration snapshots) are in
`D:\_work\projects\autoTraining\autotrain_db_backups\`.

Always use a **fully-qualified** path in `DATABASE_URL`. Two ways to get this
wrong, both of which SQLite handles by silently creating a new empty database:

- A relative `file:./db/custom.db` is resolved against the **`prisma/` folder**,
  not the repo root.
- A rootless absolute path like `file:/home/z/db/custom.db` is resolved against
  the **current drive** on Windows. The project originally shipped exactly this
  (a leftover from the Linux scaffold), so the real data ended up at
  `D:\home\z\my-project\db\custom.db` and launching from another drive would
  have quietly used an empty database.

`src/lib/db.ts` now prints a loud warning at startup when `DATABASE_URL` points
at a non-existent SQLite file, so this fails visibly instead of silently.

**Important, as of 2026-08:** the tracked `.env` still contains
`DATABASE_URL=file:/home/z/my-project/db/custom.db`, so a server started from the
`D:` drive actually uses `D:\home\z\my-project\db\custom.db`, *not* the
`autotrain_db` path documented above. Both files currently hold the same data.
Any schema change must therefore be pushed to **both**:

```bash
$env:DATABASE_URL="file:D:/_work/projects/autoTraining/autotrain_db/custom.db"; npx prisma db push
$env:DATABASE_URL="file:D:/home/z/my-project/db/custom.db";                     npx prisma db push
```

Decide which one is canonical and fix `.env` — but do it deliberately, since it
changes which database the app reads.

Never commit a `.db` file. `db/custom.db` used to be tracked and held a
pre-user-system schema; it has been untracked and `.gitignore` now excludes
`*.db` and `/db/`.

Note that `.env` and `.env.local` **are tracked in git** despite the `.env*`
rule in `.gitignore` (the rule does not apply to already-tracked files).

### Prisma migrations are behind the schema

`prisma/migrations/` stops at `20260403`; `schema.prisma` is much newer. The
project uses `prisma db push`, so `migrate deploy` will not reproduce the
current schema. Regenerate the client (`npm run db:generate`) after editing the
schema, or code will need `as any` casts to reach new columns.

### Config YAML is the source of truth, not the DB columns

`TrainingConfig` and `Model` have flat scalar columns (`epoch`, `baseLr`,
`architecture`, ...). These are a **display cache only**. `yamlConfig` is
authoritative and is what gets merged into the job config and trained on.

Unit convention for the epoch-named columns: they hold values in the
framework's **native training unit**. For a segmentation config (PaddleSeg,
TorchSeg) that means `epoch`/`maxEpochs` carry `iters`, `warmupEpochs` carries
`warmup_iters`, and `snapshotEpoch` carries `save_interval`. `iters`/`saveInterval`
are segmentation-only and null for other frameworks. Use
`countsIterations(framework)` from `src/lib/training-yaml.ts` when the unit
matters. TorchAnomaly also counts steps: `epoch` carries `trainer.max_steps` and
`snapshotEpoch` carries the validation interval (it has no `save_interval`).

The same applies to `TrainingJob.currentEpoch` / `totalEpochs`: for segmentation
frameworks **both hold iterations**. `writeParsedLog` stores `log.iteration` (not
`log.epoch`) for those frameworks, because the Seg log line carries both
(`epoch: 2278, iter: 20500/160000`) and storing the epoch made the progress bar
read 2278/160000 instead of 20500/160000.

If the columns ever drift from the YAML again, re-derive them:

```bash
npx tsx scripts/backfill-config-columns.ts --db "file:D:/.../custom.db"          # dry run
npx tsx scripts/backfill-config-columns.ts --db "file:D:/.../custom.db" --apply
```

That script only writes columns the YAML actually states — it will not overwrite
a stale value with a parser default, and it skips records whose YAML does not
parse. Back up the database first.

Always derive the columns from the YAML via the shared libraries — never the
other way around:

- `src/lib/training-yaml.ts` — `generateTrainingYaml`, `parseTrainingParams`,
  `trainingParamsToColumns`, `totalStepsFor`, `countsIterations`,
  `TRAINING_FIELD_SUPPORT`
- `src/lib/model-yaml.ts` — `generateModelYaml`, `parseModelParams`,
  `modelParamsToColumns`, `validateModelParams`, `SEG_ARCHITECTURES`,
  `TORCH_SEG_ARCHITECTURES`, `TORCH_DET_PRESETS`, `ANOMALY_PRESETS`
- `src/lib/job-commands.ts` — `buildTrainCommand`, `buildEvalCommand`,
  `buildInferCommand`, `buildExportCommand`, `bestWeightsPath`

All three are pure (no `fs`) so the browser can import them. The create/edit
dialogs render their form from `TRAINING_FIELD_SUPPORT` and preview
`generateTrainingYaml`, guaranteeing the previewed YAML is what gets saved; the
job/validation pages preview commands via `job-commands.ts`, which is the same
module the API routes execute, so a previewed command cannot drift from the real
one (it used to, in three separate copies).

### Merging job YAML

Use `mergeYamlConfigs` from `src/lib/yaml-merge.ts`. Do **not** concatenate
config text:

- PaddleDetection configs use custom tags (`!COCODataSet`, `!CosineDecay`).
  `YAML.parse()` silently drops them; only `YAML.parseDocument()` round-trips
  them. The merge therefore operates on the AST.
- Concatenation produces duplicate top-level keys. PyYAML tolerates that
  (last wins) but the document is invalid YAML and `yaml.parse()` throws
  `Map keys must be unique`.

Precedence is dataset → training → model (later wins), merged per-key, so a
training config can refine `train_dataset.transforms` without restating the
dataset block.

### PaddleSeg specifics

- Trains by **iteration** (`iters`), not epoch. `TrainingJob.totalEpochs` stores
  iterations for Seg jobs — use `totalStepsFor(framework, params)`.
- `len(loss.types)` must equal the number of logits the architecture emits
  (main head + aux heads), or training dies with
  `RuntimeError: The length of logits_list should equal to the types of loss config`.
  The counts live in `SEG_ARCHITECTURES[].logits`. **`loss:` belongs in the
  model config**, never the training config.
- `save_dir` / `save_interval` / `log_iters` are CLI arguments, not YAML keys.

### Unsupervised anomaly detection (TorchAnomaly)

New framework, backed by **anomalib** through a thin adapter at
`torchtrain/torchtrain/ad/`. Full rationale and the config shapes are in
`docs/ANOMALY_DETECTION_DESIGN.md`. The parts that will bite:

- **It is the only framework that does not reuse a Paddle schema.** The merged job
  config is anomalib's own `model:` / `data:` / `trainer:` (jsonargparse
  `class_path` + `init_args`), plus an `autotrain:` block that is ours and is
  stripped before anomalib sees it. The three-config split maps onto those three
  blocks one-to-one, which is why this backend was chosen.
- **anomalib's default `Evaluator` registers test metrics only.** A `fit()` run
  therefore logs nothing during validation: no AUROC chart, and
  `ModelCheckpoint(monitor=...)` has nothing to monitor.
  `torchtrain/torchtrain/ad/evaluator.py` exists solely to fix that.
- **`Engine.fit()` writes to a versioned directory of its own**
  (`<root>/<Model>/<dataset>/<category>/vN`), so the adapter copies the best
  checkpoint to `<save_dir>/best_model/model.ckpt` — note `.ckpt`, not `.pt`,
  because only a Lightning checkpoint can be reloaded by anomalib.
- **`trainer.global_step` stays 0 for PatchCore/PaDiM** (no optimizer steps), so
  the progress logger counts batches instead. Their loss is a constant 0 by
  design; a flat loss curve is not a stalled run.
- **`val_check_interval` cannot be set from the YAML.** An int larger than the
  batch count is a hard Lightning error unless `check_val_every_n_epoch=None`,
  and that would disable the only validation a one-epoch model ever gets. The
  training config states `autotrain.val_interval` and
  `ad/config.py::resolve_val_args` converts it once the batch count is known.
- **Tiling only works for PaDiM / PatchCore / ReverseDistillation / STFPM.**
  EfficientAD has no `tiler`; the UI disables the combination and the adapter
  refuses it with a message naming the alternatives.
- **EfficientAD has two hard constraints** enforced by anomalib at train start:
  `train_batch_size` must be 1, and the pre-processing transform must not
  normalise. It also downloads a teacher checkpoint plus ImageNette (~1.5 GB) on
  first run — pre-warm the cache on offline machines.
- **Datasets are `AnomalyFolder`**: four directories (`normalDir` required;
  `normalTestDir` / `abnormalDir` / `maskDir` optional) on the `Dataset` row. Only
  `normalDir` is trained on. anomalib takes half the test split as validation and
  fits the score threshold on it, so metrics from a training run are optimistic —
  keep a separate batch of defect images for an unbiased validation job.
- anomalib needs **its own venv** (`torchtrain/requirements-ad.txt`, pinned to
  `anomalib==2.6.*`) registered under `frameworkPythonMappings.TorchAnomaly`. It
  removes deprecated APIs every minor release, and the adapter depends on
  `Engine.from_config`, `Engine._cache.args` and the model-side
  `configure_pre_processor` / `configure_evaluator` hooks.

### PyTorch frameworks (TorchSeg / TorchDet)

The trainer is **bundled** at `torchtrain/` (see `torchtrain/README.md`). It is
not a wrapper around an external repo: it is a self-contained package that
mirrors the Paddle repo shape (`torchtrain/` package + `tools/train.py`) so the
platform treats it identically.

The core design decision: **torch reuses the Paddle config schemas and log
formats**. `TorchSeg` consumes PaddleSeg-shaped YAML, `TorchDet` consumes
PaddleDetection-shaped YAML, and `torchtrain/torchtrain/logger.py` reproduces the
Paddle stdout formats byte-for-byte. That is why `src/lib/log-parsers/*`,
`yaml-merge`, the config generators and the monitoring charts are shared rather
than forked. If you change a log format on either side, change both.

Things that bite:

- **A torch job must not be wrapped in `paddle.distributed.launch`.** `paddle` is
  not installed in a torch env. `launchPrefix()` in `src/lib/frameworks.ts`
  handles this; the GPU comes from `CUDA_VISIBLE_DEVICES`, which the runner sets.
- **Python environments cannot be shared between runtimes.** `SystemConfig`
  therefore has `frameworkPythonMappings` (`{"TorchSeg": "…/python.exe"}`), which
  `resolvePythonPath()` consults *before* the historical per-GPU
  `gpuPythonMappings`. Without an entry, a torch job fails fast with an
  actionable message rather than importing `torch` from a Paddle env.
- **`num_classes` means different things.** PaddleDetection (and therefore
  TorchDet configs) counts foreground classes only; torchvision wants background
  included. `torchtrain/det/dataset.py` does the `+1` and the label shift.
- **Detection normalisation is a no-op.** torchvision detectors normalise inside
  `GeneralizedRCNNTransform`, so `NormalizeImage`/`Permute`/`PadGT` are accepted
  and ignored, and `TRAINING_FIELD_SUPPORT.TorchDet` omits those form fields
  rather than showing controls that do nothing.
- **Detection metrics have no pycocotools hard dependency.**
  `torchtrain/det/metrics.py` ships a NumPy COCOeval re-implementation that was
  verified to agree with pycocotools to 0.0 on all 12 stats; pycocotools is used
  when importable.
- **Checkpoints are nested** (`<save_dir>/best_model/model.pt`), like PaddleSeg,
  not flat like PaddleDetection. `FRAMEWORK_META[...].checkpointLayout` drives the
  discovery in `/api/checkpoints`.
- Architecture tables must stay in sync in two places:
  `TORCH_SEG_ARCHITECTURES` / `TORCH_DET_PRESETS` in `src/lib/model-yaml.ts`
  (UI + validation) and `torchtrain/torchtrain/{seg,det}/models.py` (what actually
  builds).

### PaddleDetection specifics

The key carrying the detection head differs per meta-architecture:
`yolo_head` (YOLOv3/PP-YOLOE), `head` (RetinaNet/CenterNet/PicoDet),
`bbox_head` (FasterRCNN), `detr_head` (DETR/RT-DETR). Backbone hyper-parameters
are backbone-specific — emitting CSPResNet's `layers`/`use_alpha` under a
`MobileNetV3:` block is a hard failure. `DETECTION_PRESETS` in
`src/lib/model-yaml.ts` encodes valid combinations.

## Conventions

### Dialog widths need a `sm:` prefix

`DialogContent` in `src/components/ui/dialog.tsx` ends its own class list with
`sm:max-w-lg`. Passing an **unprefixed** `max-w-*` does not override it:
tailwind-merge treats `max-w-5xl` and `sm:max-w-lg` as different groups (one is
responsive, one is not), so both survive the merge, and above the `sm`
breakpoint the responsive rule wins. The dialog silently renders at 32rem no
matter what you asked for.

```tsx
<DialogContent className="max-w-5xl">      // ignored above 640px
<DialogContent className="sm:max-w-2xl">   // works
<DialogContent className="sm:max-w-[min(1500px,94vw)]">  // works
```

Several dialogs in the app still have the unprefixed form and are therefore
rendering narrower than intended: `annotation.tsx:954`, `datasets.tsx:1163`,
`1382`, `1529`, `1612`, `jobs.tsx:712`, `users.tsx:311`, `validation.tsx:1760`,
`change-password-dialog.tsx:76`. (Note `max-w-sm`/`max-w-md` ones currently
render *wider* than intended, so fixing those makes them smaller.)

### API routes

- Every `[id]` route must call `requireOwnedScope(request, id)` from
  `src/lib/auth.ts` and query with `findFirst({ where: scope.where })`. Looking
  up by bare `{ id }` is an IDOR — ids are enumerable and rows belong to users.
- Never accept `command` from a request body. Commands are generated
  server-side and executed with `shell: true`.
- Build command strings with `src/lib/job-commands.ts`, never by hand. It is
  imported by both the API routes and the UI previews, which is what keeps the
  two from drifting, and it derives every flag from `FRAMEWORK_META`.
- Resolve interpreters with `resolvePythonPath(framework, gpuId, systemConfig)`.
  Reading `gpuPythonMappings` directly is how three call sites ended up with
  copies of the parsing logic, two of which were wrong (one expected
  `{gpu: {pythonPath}}` when the stored shape is `{gpu: "path"}` and so always
  fell back to a bare `python`).
- Any user string that becomes a path segment must go through `toSafeSlug` from
  `src/lib/safe-path.ts`; any assembled path must be checked with `isInside`
  before a destructive operation.
- Response envelope is inconsistent across the codebase (`{data}` vs
  `{success, data}` vs raw). Prefer `{ success, data | error }` for new code.

### Known remaining issues (not yet fixed)

- `/api/system/environment-check`, `/api/system/gpu`, `/api/system/gpu-usage`,
  `/api/images`, `/api/datasets/image`, `/api/checkpoints`, `/api/annotation/save`
  are unauthenticated. The login screen polls the `system/*` ones, so adding
  auth requires a UI change first.
- Paths are interpolated into a shell string in the TIFF-staging helper in
  `src/app/api/validation-jobs/route.ts`. Should use `spawn` with an argv array
  instead of `exec`. (`checkpoints/export` was converted to argv and is fine.)
- `src/app/api/training-jobs/[id]/route.ts` still contains a dead
  `parseAndUpdateProgress` copy of the old inline log parser; the live path uses
  `src/lib/log-parsers`.
- Sessions live in memory (`src/lib/session-store.ts`) and are lost on restart;
  passwords are unsalted SHA-256; `changePassword` does not invalidate sessions.
- `runningProcesses` is an in-memory map, so a server restart orphans running
  training processes — they keep running but can no longer be stopped.
- `mini-services/training-service` is a **mock** (synthetic logs on a timer) and
  imports Prisma from a hardcoded `/home/z/my-project/...` path. Real training
  runs in `src/app/api/training-jobs/[id]/route.ts`.
