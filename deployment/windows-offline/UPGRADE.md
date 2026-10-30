# AutoTrain Windows application update

This is an update for an existing Windows offline installation with `app/`,
`runtime/`, `frameworks/` and `data/`. It includes the production web application
and actual model/training YAML defaults for all six frameworks. Node.js, Python,
framework environments and model weights are reused from the existing installation.

## Changes included

- Labelme image-path handling and COCO train/val statistics.
- Recover omitted COCO validation paths on import/re-parse. Invalid validation
  annotations now produce an error without clearing existing statistics.
- A **Validate during training** switch when creating a job (enabled by default).
  PaddleDetection now receives `--eval`; detection/segmentation jobs check their
  validation annotations before creation when validation is enabled.
- Monitoring reads validation history separately from recent training logs and
  plots detection mAP@0.5:0.95 / mAP@0.5 by epoch, including a single result.
- Select which new Labelme categories to import when merging into COCO.
- Training-job class counts come from the selected dataset.
- Default model/training configuration dropdowns, file refresh and YAML loading.

No database schema migration is required for this update.

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

Backups are under `<InstallRoot>/upgrade-backups/<timestamp>/`. Stop the app,
replace `<InstallRoot>/app` with the backup's `app`, and restore the backed-up
launcher files. `added-default-configs.txt` lists templates added by this update;
remove those only if you also want to undo the new defaults.

## Building this package

On the build workstation (Node.js, npm and tsx available):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows-offline\Build-MinimalUpgrade.ps1 -OutputDirectory "E:\AutoTrainOfflineBuild\output" -DefaultConfigsPath "D:\_work\projects\autoTraining\userConfigBase"
```

Built-in defaults are always exported. `-DefaultConfigsPath` adds YAML files from
the site's framework default folders; it does not copy personal user folders.
`-SkipBuild` reuses an already verified production build and still copies its
static/public assets. The ZIP contains no local `.env`, database or test data.
