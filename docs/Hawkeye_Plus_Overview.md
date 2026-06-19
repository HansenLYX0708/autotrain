# Hawkeye+ AutoTrain Platform — Executive Briefing

> Audience: Leadership / Management
> Tone: Concise, business-oriented, outcome-driven
> Structure: Main deck = platform value & workflow; Backup = technical implementation

---

## Main Deck — Platform Overview

### Slide 1 — Project Background

**Title:** From Command Line to One-Stop Platform — Why We Built Hawkeye+

- **Industry context**
  - Custom object-detection models are increasingly demanded by industrial inspection, defect detection, and similar scenarios.
  - Mainstream frameworks such as PaddleDetection are powerful but command-line driven, requiring hand-edited YAML configurations and offering limited training visibility.
  - Data annotation, format conversion, training, evaluation, and deployment are fragmented across disjointed tools.
- **Internal pain points**
  - Algorithm engineers spend significant time on repetitive setup work instead of model improvement.
  - Non-algorithm staff (annotators, QA, business owners) cannot directly participate in the model iteration loop.
  - Shared GPU servers lack governance, leading to resource conflicts and unclear accountability.
  - Training runs are opaque; model versions and experiments are hard to track and reproduce.
- **Our goal**
  - Deliver a **web-based, visualized, multi-user training platform** that turns model development into a configure-and-click experience, while preserving full transparency and control.

---

### Slide 2 — Platform Overview & Key Differentiators

**Positioning (one-liner):**
> Hawkeye+ is an end-to-end, fully visualized training platform built on PaddleDetection, covering the full lifecycle from data to deployment.

**Six Key Differentiators**

| Differentiator | Business Value |
|---|---|
| End-to-end coverage | Projects, datasets, models, training, monitoring, and validation in one place |
| Zero command line | All operations are GUI-driven; YAML configurations can be imported or edited visually |
| Real-time visibility | Live charts for loss, learning rate, mAP, GPU utilization, and ETA |
| Multi-user collaboration | Role-based access (Admin / User), per-user isolation, full activity audit log |
| Built-in data preparation | Integrated annotation tool plus one-click Labelme to COCO conversion |
| Deployment-ready output | TensorRT model export and download directly from the platform |

**User Roles**

- **Administrator** — System settings, user management, data annotation, plus all standard features.
- **User** — Full self-service workflow: project → dataset → model → training → validation.

---

### Slide 3 — End-to-End Workflow at a Glance

**Title:** The Hawkeye+ Workflow in One Picture

```
                ┌──── System Settings & Environment Check (Admin) ────┐
                │                                                      │
[1] Projects → [2] Datasets → [3] Models → [4] Configurations → [5] Jobs → [6] Monitoring → [7] Validation
                ↑                                                                                  │
                │                                                                                  │
                └────── Data Annotation (Admin)            TensorRT Export / Model Download ──────┘

                ──── Dashboard: global overview, GPU & job status in real time ────
```

**Highlights**
- Seven sequential, iterable core steps from data to deployable model.
- Supporting modules: Dashboard, Annotation, User Management, Settings.
- Persistent status indicator (System Ready / GPU Occupied / Training / Environment Error) visible across the platform.

---

### Slide 4 — Step 1: Project Management

**Purpose:** The top-level container that organizes all training assets.

**Capabilities**
- Create, edit, search, and delete projects with unique names and descriptions.
- All datasets, models, configurations, and jobs are scoped to a project.
- Card-based listing for quick navigation.

**Business Value**
- **Clear organization** — every asset is traceable to a business initiative.
- **Strong isolation** — different products, versions, or customers do not interfere with each other.
- **Lifecycle control** — legacy projects can be archived rather than lost.

---

### Slide 5 — Step 2: Dataset Management

**Purpose:** Answer "where does training data come from, and is it healthy?"

**Capabilities**
- Two ingestion paths: **chunked web upload** (large files, resume-on-failure) and **server-side import** of existing datasets with auto-detected paths.
- Supported formats: **COCO** and **Labelme**.
- One-click **Labelme to COCO conversion** with configurable train / validation / test split ratios.
- Dataset insight: annotation counts, class distribution charts, sample preview with bounding boxes, and class filtering.

**Business Value**
- **Large-file friendly** — chunked transfer with resume support.
- **Format agnostic** — built-in conversion eliminates external tooling.
- **Quality assurance up-front** — visual inspection catches data issues before training starts.

---

### Slide 6 — Step 3: Model Configuration

**Purpose:** Decide which network architecture will learn the task.

**Capabilities**
- Three configuration sources:
  - **Default templates** from PaddleDetection.
  - **Reusable user configurations** saved by the team.
  - **Custom YAML** for advanced users.
- Each model is named and bound to a project.

**Business Value**
- **Flexible by skill level** — beginners use templates, experts customize.
- **Knowledge accumulation** — proven configurations become team assets.
- **Full transparency** — underlying YAML is always inspectable and exportable.

---

### Slide 7 — Step 4: Training Configuration

**Purpose:** Define how the model will be trained — hyperparameters and schedules.

**Capabilities**
- Parameter coverage includes epochs, batch size, learning rate, scheduler, warmup, worker count, and more.
- Same three configuration sources as model module: defaults, user history, custom YAML.
- Configurations are independent assets that can be reused across jobs.

**Business Value**
- **Reusable recipes** — proven training setups can power many experiments.
- **Decoupled design** — any model can be paired with any configuration and dataset.

---

### Slide 8 — Step 5: Training Jobs

**Purpose:** Turn "data + model + configuration" into actual GPU work.

**Capabilities**
- Job submission with selectable training configuration and explicit GPU assignment (single or multi-GPU, e.g. `0,1`).
- Training options: AMP (mixed precision), VDL (VisualDL logging).
- Full lifecycle: Pending → Running → Completed / Failed / Stopped.
- Job actions: start, stop, resume from checkpoint, delete, and view live logs.

**Business Value**
- **Resource control** — explicit GPU allocation prevents conflicts.
- **Resilient** — long runs can be stopped and resumed from the latest checkpoint.
- **At-a-glance status** — color-coded badges make queue health obvious.

---

### Slide 9 — Step 6: Training Monitoring

**Purpose:** Eliminate the "black box" during training.

**Capabilities**
- Live charts: loss, learning rate, mAP, iterations per second, ETA.
- Per-GPU resource monitoring: memory and utilization.
- Streaming log view with error highlighting and search.
- Checkpoint management: automatic save by epoch, best-model tracking, resume support, and cleanup.

**Business Value**
- **Real-time insight** — issues are caught early, not after hours of wasted compute.
- **Auditability** — full historical logs and checkpoints are retained.
- **Storage efficiency** — automated cleanup of obsolete checkpoints.

---

### Slide 10 — Step 7: Validation, Inference & Export

**Purpose:** Take a trained model from "completed" to "evaluated, used, and deployable".

**Capabilities**
- Select any completed training job and a specific checkpoint.
- **Evaluation:** mAP@0.5 / 0.75 / 0.5:0.95, AR@1 / 10 / 100, accuracy by object size (S / M / L), shown as interactive charts.
- **Inference:** single image or batch folder, with annotated output images.
- **TensorRT export:** one-click export and download for deployment.
- **Validation history:** complete record of every evaluation and inference run.

**Business Value**
- **Closes the loop to deployment** — no separate toolchain required.
- **Model comparison** — different checkpoints can be evaluated side by side.
- **Visual evidence** — charts and annotated images make model quality intuitive for non-experts.

---

### Slide 11 — Supporting Modules

**Dashboard** — Global overview of projects, jobs, GPUs, and recent activity.

**Annotation (Admin)** — Built-in image annotation tool supporting creation, editing, and review; class selection, drag-to-draw, keyboard navigation, and auto-save every 30 seconds.

**User Management (Admin)** — Create / edit / disable / delete users, assign roles, reset passwords, and review per-user activity logs.

**Settings (Admin)** — Python and PaddleDetection paths, user data and configuration paths, per-GPU Python environment mapping, and one-click **environment health check** (Python, CUDA, PaddleDetection, disk space).

---

### Slide 12 — Summary & Value Delivered

**Value to algorithm engineers**
- Removes repetitive setup work; focus shifts to data and modeling.

**Value to non-algorithm staff**
- Lowers the barrier to participate in annotation, training, and evaluation.

**Value to leadership and operations**
- Visible resource usage, controlled processes, traceable outputs, and a clear path from data to deployable model — all on a single web platform.

---

## Backup — Technical Implementation

### Backup 1 — Overall Technical Architecture

**Frontend**
- Next.js 16 (App Router) + React 19 + TypeScript 5
- Tailwind CSS 4 + shadcn/ui (Radix UI)
- State and data: Zustand + TanStack Query
- Forms and validation: React Hook Form + Zod
- Visualization and UX: Recharts, Framer Motion, next-intl, next-themes

**Backend**
- Next.js API Routes (40+ endpoints under `src/app/api/`)
- Authentication: NextAuth.js with role-based middleware
- Data layer: Prisma ORM over SQLite

**Training Microservice** (`mini-services/training-service/`)
- Bun runtime with Socket.io on port 3003
- Spawns training subprocesses, parses PaddleDetection logs, and streams progress in real time

**Deployment**
- `next build` produces a standalone bundle; Bun serves it
- Caddy reverse proxy; `start.vbs` provides one-click startup on Windows

---

### Backup 2 — Data Model (Prisma Schema)

**Core entities**
- `User`, `SystemConfig`
- `Project` → `Dataset` / `Model` / `TrainingConfig`
- `TrainingJob` → `TrainingLog`
- `ValidationJob`
- `GpuMetric`, `ActivityLog`

**Design notes**
- All training assets are rooted at `Project` for clean ownership.
- `TrainingLog` persists per-epoch / per-iteration metrics for later analysis.
- `ActivityLog` records user actions to support auditing and compliance.

---

### Backup 3 — Real-Time Training Channel (Socket.io)

**Client to Server:** `training:start`, `training:stop`, `training:status`, `training:subscribe`, `training:unsubscribe`

**Server to Client:** `training:started`, `training:log`, `training:progress`, `training:epoch`, `training:complete`, `training:stopped`, `training:error`

**Log pipeline**
- A dedicated parser (`log-parser.ts`) extracts epoch, iteration, loss, learning rate, ETA, and memory metrics from PaddleDetection logs.
- Parsed records are persisted to the database and streamed to subscribed clients.

---

### Backup 4 — Key Engineering Details

- **Multi-GPU, multi-environment:** `SystemConfig.gpuPythonMappings` binds each GPU to its own Python environment.
- **Large-file upload:** chunked transfer with resume-on-failure for datasets.
- **Three configuration sources:** default templates, user history, and custom YAML — unified as YAML strings in the database.
- **Job isolation:** each training job has its own `outputDir`, `weightsPath`, and `vdlLogDir`.
- **Environment health check:** `/api/system/environment-check` verifies Python, PaddleDetection, and GPU status, surfaced as a live status badge in the header.
