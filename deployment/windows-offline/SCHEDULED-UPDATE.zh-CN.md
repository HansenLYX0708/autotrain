# AutoTrain Windows 11 定时更新工具

每次只需输入三项：

1. **Update ZIP path**：更新包 ZIP 的完整路径。
2. **AutoTrain installation directory**：当前安装目录，例如 `C:\AutoTrain`。
3. **Update time**：本机时间，格式为 `yyyy-MM-dd HH:mm`，例如 `2026-11-01 02:00`。至少提前一分钟。

## 使用步骤

1. 解压本工具压缩包，保留所有脚本在同一个文件夹。
2. 右键 `Schedule-AutoTrainUpdate.cmd`，选择 **以管理员身份运行**。
3. 按提示输入上面三项内容，看到 `Update scheduled for ...` 即表示计划已建立。
4. 可以关闭窗口。程序会将更新包复制到安装目录的 `update-plans` 文件夹，之后不再依赖原始 ZIP 所在位置。

工具使用 Windows 任务计划程序，以 SYSTEM 账户运行；退出登录后也可以执行。
请保持电脑开机。关机期间无法更新；错过时间后，会在 Windows 下次允许任务运行时补执行。
安装目录、数据和配置目录需要使用 SYSTEM 可访问的本地磁盘路径。

本工具适用于已有 `app`、`runtime`、`data` 目录的 AutoTrain Windows 离线安装。
更新包使用之前交付的 `AutoTrain-Minimal-Upgrade-*.zip`，不接受完整安装包或数据库迁移包。
当前更新包内的默认模型和训练配置也会按包内规则安装：补充缺少文件，保留同名已有文件。

## 到时间后的流程

校验更新包 → 解压并核对文件 → 检查目标安装的服务/进程 → 如正在运行则停止 →
备份旧程序 → 执行更新 → 启动 → 检查网页服务是否正常。

普通安装按 `app/server.js` 的完整路径识别后台进程，不会直接杀掉所有 `node.exe`。
若通过 Windows 服务运行，则依据服务路径或服务所属进程识别；遇到多个匹配服务会停止更新并记录错误。
没有正在运行的实例时，也会在更新完成后启动。

如果升级或新程序启动失败，会尝试恢复备份的程序和启动脚本，再启动旧程序，并记录恢复结果。
程序回退不会删除新增的默认模板，也不会替换数据库。

请安排在没有训练任务运行的时段；停止 Web 服务不能替代正常结束训练任务。

## 查看结果或取消计划

创建计划后会显示 **Task**（任务名）和 **Plan and logs**（计划及日志目录）。

- 按 `Win + R`，输入 `taskschd.msc` 打开任务计划程序，找到 `AutoTrain-Update-...`。
- 可以查看、提前运行、禁用或删除该任务。
- 同一安装目录再次创建计划会替换之前尚未执行的计划；已在执行的更新不能替换。
- `update.log`：执行日志。
- `status.json`：`scheduled` 等待执行、`running` 执行中、`succeeded` 成功、`failed` 失败。
- 失败时查看 `error` 和 `recovery` 字段，分别记录失败原因和旧程序恢复结果。
- `run-.../backup/`：本次更新前的程序及启动脚本备份。

这些目录由管理员和 SYSTEM 管理，查看日志或手动操作时请使用管理员权限。

## 命令行使用

在管理员 PowerShell 中运行：

```powershell
.\Schedule-AutoTrainUpdate.ps1 `
  -PackagePath "D:\Updates\AutoTrain-Minimal-Upgrade-20261028-datasets-configs.zip" `
  -InstallRoot "C:\AutoTrain" `
  -UpdateTime "2026-11-01 02:00"
```

要手动立即执行已经建立的计划：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "C:\AutoTrain\update-plans\<计划目录>\Invoke-AutoTrainUpdate.ps1" `
  -PlanPath "C:\AutoTrain\update-plans\<计划目录>\plan.json"
```

## 本机验证范围

已在 Windows 11 隔离安装目录测试真实更新包：停止/升级/重启、带空格的安装路径、
旧 PID 文件、其他 Node 进程保护、数据库保留、包校验失败和安装失败后的恢复。
Windows 任务计划程序接受了任务定义。
验证会话未提升管理员权限，未注册或实际触发 SYSTEM 计划任务，也未安装 Windows 服务测试其启停。
