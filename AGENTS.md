# AGENTS.md

Project knowledge for anyone (human or agent) working on this repo.

## What this is

A Next.js 16 (App Router, Turbopack) + Prisma/SQLite web app that drives
PaddlePaddle training. A "job" is assembled from three independently authored
YAML configs and run as a child process:

```
Dataset config  ─┐
Training config ─┼─► deep-merged YAML ─► tools/train.py ─► stdout parsed into TrainingLog rows
Model config    ─┘
```

Supported frameworks: `PaddleDetection`, `PaddleClas`, `PaddleSeg`
(`project.framework` selects one; `src/lib/frameworks.ts` is the single source of
truth for resolving each one's working directory).

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
framework's **native training unit**. For a PaddleSeg config that means
`epoch`/`maxEpochs` carry `iters`, `warmupEpochs` carries `warmup_iters`, and
`snapshotEpoch` carries `save_interval`. `iters`/`saveInterval` are Seg-only and
null for other frameworks. Branch on `project.framework` when the unit matters.

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
  `trainingParamsToColumns`, `totalStepsFor`, `TRAINING_FIELD_SUPPORT`
- `src/lib/model-yaml.ts` — `generateModelYaml`, `parseModelParams`,
  `modelParamsToColumns`, `validateModelParams`, `SEG_ARCHITECTURES`

Both are pure (no `fs`) so the browser can import them; the create/edit dialogs
render their form from `TRAINING_FIELD_SUPPORT` and preview
`generateTrainingYaml`, guaranteeing the previewed YAML is what gets saved.

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
- Paths are interpolated into shell strings in
  `src/app/api/checkpoints/export/route.ts` and
  `src/app/api/validation-jobs/route.ts` (TIFF staging). Should use `spawn`
  with an argv array instead of `exec`.
- Sessions live in memory (`src/lib/session-store.ts`) and are lost on restart;
  passwords are unsalted SHA-256; `changePassword` does not invalidate sessions.
- `runningProcesses` is an in-memory map, so a server restart orphans running
  training processes — they keep running but can no longer be stopped.
- `mini-services/training-service` is a **mock** (synthetic logs on a timer) and
  imports Prisma from a hardcoded `/home/z/my-project/...` path. Real training
  runs in `src/app/api/training-jobs/[id]/route.ts`.
