# Scheduled updates on Windows 11

The scheduling tool accepts three inputs:

1. The AutoTrain minimal upgrade ZIP.
2. The existing offline installation directory, such as `C:\AutoTrain`.
3. The local execution time in `yyyy-MM-dd HH:mm` format.

Run `Schedule-AutoTrainUpdate.cmd` as administrator. It opens the PowerShell
scheduler, which asks for missing inputs. Command-line use is also supported:

```powershell
.\Schedule-AutoTrainUpdate.ps1 -PackagePath "D:\Updates\AutoTrain-Minimal-Upgrade.zip" -InstallRoot "C:\AutoTrain" -UpdateTime "2026-11-01 02:00"
```

The tool copies the ZIP and execution scripts into the installation's
`update-plans` directory. The original ZIP may then be moved. A Windows Scheduled
Task runs once as SYSTEM at the selected local time, including when no user is
logged in. Keep the computer on; if it is off, the task runs when Windows next
allows it to start. Package, installation, configuration and data locations must
be on local disks accessible to SYSTEM, rather than mapped network drives.

At execution time, the worker:

1. Acquires an installation-specific lock and verifies the staged ZIP hash.
2. Extracts the package and verifies its file checksums before stopping anything.
3. Identifies the installation's background server or associated Windows service.
4. Stops that instance, takes a program backup, and applies the package.
5. Starts the server/service and checks the configured local HTTP port.
6. Records success or failure in the plan directory.

The worker does not indiscriminately terminate `node.exe` processes. For a
standard offline installation, it matches the absolute `app/server.js` path.
For a Windows service, it matches the installation path or the service process
that owns the server. Unrelated applications are left running.

If applying the update or starting the updated app fails after backup, the worker
attempts to restore the previous app and launcher files and restart them. Logs
record both the original error and the recovery result. Database migration
packages and complete offline installation bundles are not supported.

Templates follow the package policy: add missing defaults and preserve existing
files. Program rollback does not remove newly added default templates or alter
the database.

Schedule updates outside training hours. Stopping the web server does not safely
finish an active training job.

## Inspect or cancel

The scheduler prints the task name and plan directory. Use Task Scheduler
(`taskschd.msc`) to inspect the task, run it early, or disable/delete it.
Each installation has one scheduled update task; scheduling again replaces a
pending plan. A currently running update cannot be replaced.

The plan directory contains `plan.json`, `update.log`, and `status.json`.
The final status indicates success, failure, or recovery after failure.

## Manual execution

For troubleshooting, an administrator can execute a staged plan immediately:

```powershell
.\Invoke-AutoTrainUpdate.ps1 -PlanPath "C:\AutoTrain\update-plans\<plan>\plan.json"
```
