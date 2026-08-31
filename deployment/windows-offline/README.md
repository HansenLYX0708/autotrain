# AutoTrain Windows 全框架离线部署

这套工具生成一个可复制到 Windows 10/11 x64 机器的一键安装包。目标机安装和运行时不需要互联网、Git、Node.js、Bun、npm、系统 Python 或 CUDA Toolkit。

目标机仍需提前安装兼容的 NVIDIA 显卡驱动。GPU wheel 自带大部分 CUDA 用户态运行库，但驱动不能由本项目通用打包。

## 1. 离线包内容

生成后的目录包含：

- Next.js standalone 生产服务和 Prisma Windows 引擎
- Node.js Windows x64 便携运行时
- Python Windows x64 私有便携基础运行时及官方备用安装器
- 全部 Python wheel 及其传递依赖
- PaddleDetection、PaddleClas、PaddleSeg 源码
- TorchDet、TorchSeg、TorchAnomaly 共用的 `torchtrain`
- 当前 Prisma schema 生成的空 SQLite 数据库
- 安装、启动、停止和环境检查脚本
- 可选的 Paddle、PyTorch、anomalib 预训练模型缓存

目标机安装时建立三个互相隔离的环境：

| 环境 | 框架 |
|---|---|
| `runtime/envs/paddle` | PaddleDetection、PaddleClas、PaddleSeg |
| `runtime/envs/torch` | TorchDet、TorchSeg |
| `runtime/envs/anomaly` | TorchAnomaly / anomalib |

不要把 anomalib 安装进普通 torch 环境。它对 Lightning、jsonargparse 和 torch 有独立约束。

## 2. 制作机准备

制作机必须是联网的 Windows x64，且与目标机使用相同的 Python 主次版本和 GPU wheel 类型。

需要准备：

1. 当前项目源码。
2. Node.js Windows x64 ZIP，不要使用 MSI。
3. Python 3.10.x Windows x64 安装器。
4. Microsoft Visual C++ 2015–2022 Redistributable x64 安装器 `VC_redist.x64.exe`。
5. 完整的 PaddleDetection、PaddleClas、PaddleSeg 源码目录。
6. 可访问 npm、PyPI、Paddle wheel 源和 PyTorch wheel 源的网络。
7. 足够磁盘空间。全框架 CUDA 包通常需要 15–30 GB，加入模型缓存后可能更大。

官方入口：

- Node.js 下载：<https://nodejs.org/en/download>
- Python 下载：<https://www.python.org/downloads/windows/>
- Visual C++ Redistributable：<https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist>
- PaddlePaddle 安装命令生成器：<https://www.paddlepaddle.org.cn/install/quick>
- PyTorch 安装命令生成器：<https://pytorch.org/get-started/locally/>

### 版本原则

- `pythonInstaller` 与 `builderPython` 必须具有相同的主次版本，例如都为 Python 3.10。
- 制作机用于构建 Web 的 Node.js 与 `nodeArchive` 必须具有相同主版本，脚本会自动检查。
- TorchAnomaly 使用 anomalib 2.6，要求 Python >= 3.10、torch >= 2.6。
- `torch` 与 `torchvision` 必须使用 PyTorch 官方匹配组合。
- CUDA wheel 必须与目标机 NVIDIA 驱动兼容。
- PaddlePaddle 与 PyTorch 可以使用不同 CUDA runtime，但目标机驱动必须同时满足两者。
- 不要照抄示例版本后直接批量部署；先在一台与目标机显卡相同的机器验证一次真实训练。

## 3. 配置离线包

复制示例配置：

```powershell
cd deployment\windows-offline
Copy-Item bundle.config.example.json bundle.config.json
```

编辑 `bundle.config.json`：

```json
{
  "bundleName": "AutoTrain-Offline-Windows-x64",
  "appSource": "../..",
  "nodeArchive": "D:/offline-sources/node-v24-win-x64.zip",
  "pythonInstaller": "D:/offline-sources/python-3.10.11-amd64.exe",
  "vcRedistInstaller": "D:/offline-sources/VC_redist.x64.exe",
  "builderPython": "C:/Python310/python.exe",
  "frameworkSources": {
    "PaddleDetection": "D:/frameworks/PaddleDetection",
    "PaddleClas": "D:/frameworks/PaddleClas",
    "PaddleSeg": "D:/frameworks/PaddleSeg"
  },
  "paddlePackage": "paddlepaddle-gpu==2.6.2",
  "paddlePipArgs": [],
  "torchPackages": [
    "torch==2.6.0+cu118",
    "torchvision==0.21.0+cu118"
  ],
  "torchIndexUrl": "https://download.pytorch.org/whl/cu118",
  "pythonIndexUrl": "https://pypi.org/simple",
  "cacheSources": {
    "paddle": "",
    "torch": "",
    "anomalib": ""
  }
}
```

`paddlePipArgs` 应填写 Paddle 官方安装命令中除包名以外的参数。例如官方命令给出了额外 wheel 地址或 index，就按字符串数组逐项填写。不要把在线 URL 写进 `paddlePackage`，该字段必须是离线安装时仍可解析的包约束。

示例配置中的版本只是配置格式示例。制作前以官方安装选择器和实际驱动为准。

## 4. 处理真正的完全离线训练

安装依赖后，模型仍可能在第一次训练时下载预训练权重：

- torchvision 的 COCO/ImageNet 权重通常位于用户 torch cache。
- Paddle 模型可能从 Paddle cache 下载权重。
- EfficientAD 首次运行还会下载 teacher checkpoint 和 ImageNette，约 1.5 GB。

可靠做法是在联网的验证机上，用部署包中计划采用的三个环境分别跑一次所需架构，让所有权重下载完成，再把对应缓存目录填入 `cacheSources`。

安装后平台会设置：

```text
TORCH_HOME=<安装目录>\data\cache\torch
PADDLE_HOME=<安装目录>\data\cache\paddle
XDG_CACHE_HOME=<安装目录>\data\cache
HF_HOME=<安装目录>\data\cache\huggingface
```

如果某个第三方库不遵守这些变量，应先在完全断网的测试机进行一次训练验证，并将其实际缓存目录补充到安装包。仅打包 Python wheel 不能保证预训练模型离线可用。

## 5. 生成离线包

在项目根目录运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows-offline\Build-OfflineBundle.ps1 `
  -ConfigPath .\deployment\windows-offline\bundle.config.json `
  -OutputDirectory D:\offline-output `
  -CreateArchive
```

脚本会执行：

1. `npm ci`
2. Prisma Client 生成
3. Next.js standalone 构建
4. 框架源码与便携 Node.js 收集
5. 所有 Python wheel 下载
6. 空数据库模板生成
7. 目录包和 ZIP 生成

已有并确认可用的 `node_modules` 可用 `-SkipNpmInstall` 节省时间：

```powershell
.\deployment\windows-offline\Build-OfflineBundle.ps1 `
  -ConfigPath .\deployment\windows-offline\bundle.config.json `
  -OutputDirectory D:\offline-output `
  -SkipNpmInstall
```

如果输出目录已存在，脚本会拒绝覆盖。确认可删除旧输出后显式加 `-Force`。

完整包很大。若 ZIP 工具或 U 盘文件系统受限，直接复制生成的 `AutoTrain-Offline-Windows-x64` 目录；U 盘建议使用 NTFS 或 exFAT，不要使用有 4 GB 单文件限制的 FAT32。

## 6. 在完全离线目标机安装

### 6.1 安装前检查

在目标机运行：

```powershell
nvidia-smi
```

确认：

- 能识别 NVIDIA GPU。
- 驱动版本满足制作包时选择的 Paddle/PyTorch CUDA wheel。
- 安装盘有足够空间。
- 安装路径和后续数据路径不会被杀毒软件或受控文件夹策略阻止。

### 6.2 一键安装

把整个离线包复制到目标机，双击：

```text
Install-AutoTrain.cmd
```

默认安装到 `C:\AutoTrain`。也可以在 PowerShell 指定目录和端口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Install.ps1 `
  -InstallRoot D:\AutoTrain `
  -Port 3000 `
  -RequireGpu
```

安装器会：

1. 复制 Web 服务和框架源码目录。
2. 安装 Microsoft Visual C++ x64 运行库。
3. 复制包内 Python 便携基础运行时到应用私有目录；如果旧包没有便携运行时才调用备用安装器。
4. 创建 paddle、torch、anomaly 三个 venv。
5. 只从本地 wheelhouse 安装依赖，禁止访问网络。
6. 创建或保留 `data/autotrain.db`。
7. 自动写入所有框架路径和 Python 映射。
8. 检查 Node、数据库、六个框架模块及 GPU。
9. 启动平台并打开浏览器。

不希望安装后立即启动时加 `-NoStart`。没有 NVIDIA GPU、只想测试 CPU 流程时不要加 `-RequireGpu`。

## 7. 启动、停止和访问

安装目录中：

```text
Start-AutoTrain.cmd
Stop-AutoTrain.cmd
Test-Installation.ps1
logs/server.log
logs/server-error.log
```

本机访问：

```text
http://localhost:3000
```

第一次注册的账号自动成为管理员。注册后进入“设置”，确认六个框架和三个 Python 环境均显示正常。

局域网其他机器访问：

```text
http://目标机IP:3000
```

如 Windows 防火墙阻止访问，可在管理员 PowerShell 中显式放行端口：

```powershell
New-NetFirewallRule -DisplayName "AutoTrain TCP 3000" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow
```

平台当前通过 `AUTH_COOKIE_SECURE=false` 支持内网 HTTP 登录。HTTP 不加密，不要暴露到公网。需要跨不可信网络访问时，应在前面部署 HTTPS 反向代理，并把 `Start.ps1` 中该变量改为 `true`。

Windows 自托管必须使用 Next.js 16.3.3 或更高的已修复版本；低于 16.3.3 的 Next.js 16 存在官方披露的 Windows 未认证远程代码执行漏洞，且没有规避方案。最终包的版本记录在 `manifest.json`。

## 8. 数据、备份与升级

所有可变数据集中在：

```text
<安装目录>\data\autotrain.db
<安装目录>\data\configs
<安装目录>\data\users
<安装目录>\data\cache
<安装目录>\logs
```

备份前先运行 `Stop-AutoTrain.cmd`，然后复制整个 `data` 目录。至少必须备份 `data/autotrain.db`、`data/configs` 和 `data/users`。

重新运行同版本安装器时不会覆盖已有 `data/autotrain.db`，但会刷新程序、框架和 Python 包。升级到包含数据库 schema 变化的新版本前，应先备份数据库，并由新离线包提供专门升级步骤；不要用旧数据库模板覆盖现有数据库。

## 9. 验证真实训练

环境检查通过后，还必须逐框架做一个最小任务：

1. PaddleDetection：1 epoch 小型 COCO 数据。
2. PaddleClas：1 epoch 小型分类数据。
3. PaddleSeg：几十 iter 小型分割数据。
4. TorchDet：1 epoch 小型 COCO 数据。
5. TorchSeg：几十 iter 小型分割数据。
6. TorchAnomaly：小型 normal/abnormal 数据。

验证以下功能：

- 训练进度和日志持续入库。
- GPU 利用率正常。
- 最优权重可发现。
- 验证、推理和导出可执行。
- 断开网络后不再产生下载请求。

## 10. 常见故障

### wheel 构建阶段失败

通常是 Python 主次版本、Windows/CUDA wheel 或包版本不兼容。制作脚本使用 `pip wheel`，会把纯 Python 源码包先构建成 wheel，确保目标机不需要联网构建。若某个含本机扩展的包只有源码发行版，制作机需要安装相应 C/C++ 构建工具，或者改用已有 Windows wheel 的兼容版本；不要把未构建的源码包直接交给目标机。

### 安装时提示找不到 wheel

离线包制作不完整，或 `manifest.json` 中版本与 wheelhouse 不一致。重新制作包，不要在目标机临时联网修补。

### `torch.cuda.is_available()` 为 false

检查：

1. `nvidia-smi` 是否正常。
2. wheel 是否为目标 CUDA 版本而不是 CPU 版。
3. 驱动是否满足该 wheel。
4. 目标 GPU 是否被其他策略隐藏。

### Paddle 可导入但框架模块失败

确认三个源码目录完整，至少分别包含 `ppdet`、`ppcls`、`paddleseg` 和 `tools/train.py`。安装器通过 `.pth` 把源码根目录注册到 paddle venv，不需要在目标机执行 `pip install -e`。

### 平台无法启动

查看：

```text
<安装目录>\logs\server-error.log
```

并运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File <安装目录>\Test-Installation.ps1 -RequireGpu
```

### 局域网能打开页面但不能登录

确认服务确实由安装目录中的 `Start.ps1` 启动，并检查 `AUTH_COOKIE_SECURE=false`。如果使用 HTTPS，则应改为 `true`。

## 11. 当前限制

- 安装包面向 Windows 10/11 x64，不跨操作系统。
- GPU 驱动不随包分发。
- 不同 GPU/驱动组合可能需要分别制作 CUDA 版本不同的离线包。
- 平台训练进程状态保存在 Web 进程内存中；训练期间不要重启平台，否则训练子进程可能继续运行但平台无法再控制它。
- 登录会话在重启后失效，需要重新登录。
- 当前认证实现不适合直接暴露到公网；推荐仅在受信任内网使用。
