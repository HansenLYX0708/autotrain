# AutoTrain Windows application update

This is an update for an existing Windows offline installation with `app/`,
`runtime/`, `frameworks/` and `data/`. It includes the production web application
and actual model/training YAML defaults for all six frameworks. Node.js, Python,
framework environments and model weights are reused from the existing installation.

## Changes included

- Dashboard environment check now probes the `ultralytics` and `anomalib`
  Python modules, so TorchOBB / TorchPose (and any future framework module)
  report correctly instead of always showing a failed import.
- Projects page: "Recent 5 / All projects" toggle (Recent 5 is the default),
  server-side search, and pagination in the All view.
- Datasets uploads moved to a background **Upload Tasks** page: browser uploads
  keep running while you navigate, batches small files into fewer requests
  (much faster for large folders), can be cancelled and resumed after the page
  is closed. A new **From server folder** mode copies files on the server in
  the background and survives closing the browser entirely. Note: a
  server-folder import in progress does not survive an application restart —
  it is marked "Interrupted by server restart" and can be retried.
- Axis-aligned rectangle annotations can now be imported as a **COCO detection
  dataset** for TorchDet / PaddleDetection directly from the localization
  annotation tool; rotated boxes are converted to their enclosing upright box.
- Annotation sets management on the Annotation page: active / archived status,
  rename, export and delete. A set not saved for 7 days is auto-archived on the
  next listing — **existing annotation sets older than 7 days will appear
  archived on first listing**; reactivate them from Annotation → Archived.
- Annotation canvas zoom shortcuts: `+` / `-` / `0` (fit) / `1` (100%),
  `Ctrl+=` / `Ctrl+-` / `Ctrl+0`, and Ctrl+wheel zoom anchored at the cursor.
- Fixed COCO dataset sample preview failing for datasets stored under an
  absolute path.
- Upload API hardening: chunk paths are validated server-side, the target
  directory is taken from the server's own record, and upload routes require
  authentication like the rest of the API.

## Database schema update (automatic)

This update adds one **new table** (`UploadTask`, for the Upload Tasks page).
The installer applies it automatically: before copying the new app it backs up
`<InstallRoot>\data\autotrain.db` (plus any `-wal`/`-shm`/`-journal` files) to
`upgrade-backups\<timestamp>\data\`, runs the additive `CREATE TABLE` /
`CREATE INDEX` statements, and verifies the database matches the shipped
schema. The step is idempotent — re-running the installer on an already
updated database detects the table and skips it. If verification fails the
database files are restored from the backup and the old application is left
untouched. No destructive migration is performed and no existing data is
changed.

For a one-time scheduled update on Windows 11, use the companion
`Schedule-AutoTrainUpdate.cmd` tool. See [SCHEDULED-UPDATE.md](SCHEDULED-UPDATE.md)
for the three-input workflow, logs and recovery behavior.

## Apply

Finish or stop training jobs before upgrading. Extract the ZIP to a directory
outside the installation. Open PowerShell in the extracted package directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apply-MinimalUpgrade.ps1 -InstallRoot "C:\AutoTrain"
```

Replace `C:\AutoTrain` with your actual installation directory. The updater verifies
all package SHA-256 checksums, stops the app, backs up the previous app and launcher
files, copies the update, installs missing default templates, and restarts the app.
Use `-NoStart` to leave it stopped after updating.

The destination for templates is read from `SystemConfig.userConfigsPath` in
`<InstallRoot>/data/autotrain.db`. If unset, it uses `<InstallRoot>/data/configs`.
An explicit override is also available:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Apply-MinimalUpgrade.ps1 -InstallRoot "D:\AutoTrain" -UserConfigsPath "D:\AutoTrainConfig"
```

The override must match the **User Configs Path** setting in the application.
The updater does not change that setting.

## Default YAML files

`payload/default-configs/inventory.json` lists every included file and its SHA-256.
Each framework directory has both `models/` and `training/`:

| Framework | Directory |
|---|---|
| PaddleDetection | `default` |
| PaddleClas | `defaultClas` |
| PaddleSeg | `defaultSeg` |
| TorchDet | `defaultTorchDet` |
| TorchSeg | `defaultTorchSeg` |
| TorchAnomaly | `defaultTorchAnomaly` |

Existing files with the same relative path are preserved. New files are copied.
This protects locally edited templates. To replace an existing template explicitly,
back it up and copy the chosen file from `payload/default-configs` manually.
Personal configuration folders and the installed database are preserved.

After updating, sign in again. Select a project in Models / Configurations and
click **Refresh** beside the default configuration dropdown. Old dataset statistics
can be refreshed with **Parse Dataset Statistics**; previously skipped conversion
images require reconversion.

For an imported COCO dataset whose val statistics are missing, use
**Parse train / val statistics** to recover the converted validation paths and
recompute both splits. Existing jobs keep their original commands: create a new
job with **Validate during training** enabled to record validation results.
Monitoring does not require VisualDL to plot these results.

## Backup and rollback

Backups are under `<InstallRoot>/upgrade-backups/<timestamp>/` (the previous
`app`, launcher files, and a copy of the database under `data/`). Stop the app,
replace `<InstallRoot>/app` with the backup's `app`, and restore the backed-up
launcher files. To also undo the schema addition, restore
`data/autotrain.db` from the backup — the old app works fine with the extra
table in place, so this is only needed for a clean rollback. `added-default-configs.txt` lists templates added by this update;
remove those only if you also want to undo the new defaults.

## Building this package

On the build workstation (Node.js, npm and tsx available), one command does
the typecheck, regression scripts, build, archive verification and a packaged
-server smoke test:

```powershell
npm run package:update -- -Label <name>
# or directly:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows-offline\New-UpdatePackage.ps1 -Label <name>
```

Per-machine paths live in `local-upgrade.config.json` (gitignored; copy
`local-upgrade.config.example.json`): `outputDirectory`, `defaultConfigsPath`,
`tscBaseline`. CLI parameters override the config. `-SkipChecks` skips tsc +
check scripts, `-SkipBuild` reuses an already verified production build,
`-SkipSmokeTest` skips only the packaged-server probe. The manifest records
`sourceRevision` (short git SHA, `-dirty` when the tree is dirty).

The low-level driver remains available for one-off use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows-offline\Build-MinimalUpgrade.ps1 -OutputDirectory "E:\AutoTrainOfflineBuild\output" -DefaultConfigsPath "D:\_work\projects\autoTraining\userConfigBase"
```

Built-in defaults are always exported. `-DefaultConfigsPath` adds YAML files from
the site's framework default folders; it does not copy personal user folders.
`-SkipBuild` reuses an already verified production build and still copies its
static/public assets. The ZIP contains no local `.env`, database or test data.
