# Windows 11 环境安装指南

本文档介绍如何在 Windows 11 上安装和运行 Auto Training Platform。

## 目录

1. [安装 Node.js](#1-安装-nodejs)
2. [安装 Bun](#2-安装-bun)
3. [安装 Python](#3-安装-python)
4. [安装 PaddlePaddle](#4-安装-paddlepaddle)
5. [安装 PaddleDetection](#5-安装-paddledetection)
6. [配置项目](#6-配置项目)
7. [运行项目](#7-运行项目)

---

## 1. 安装 Node.js

### 方法一：使用官方安装包

1. 访问 [Node.js 官网](https://nodejs.org/)
2. 下载 **LTS 版本**（推荐 20.x 或更高）
3. 运行安装程序，按照提示完成安装
4. 打开 **PowerShell** 或 **命令提示符**，验证安装：

```powershell
node --version
npm --version
```

### 方法二：使用 winget（Windows 包管理器）

```powershell
winget install OpenJS.NodeJS.LTS
```

---

## 2. 安装 Bun

Bun 是本项目使用的 JavaScript 运行时和包管理器。

### 方法一：使用 PowerShell 安装

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
```

### 方法二：使用 npm 安装

```powershell
npm install -g bun
```

### 验证安装

```powershell
bun --version
```

---

## 3. 安装 Python

PaddleDetection 需要 Python 3.8-3.12。

### 方法一：使用官方安装包

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 Python 3.10 或 3.11（推荐）
3. **重要**：安装时勾选 **"Add Python to PATH"**
4. 验证安装：

```powershell
python --version
pip --version
```

### 方法二：使用 winget

```powershell
winget install Python.Python.3.10
```

### 方法三：使用 Anaconda/Miniconda（推荐用于深度学习）

1. 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. 安装后创建专用环境：

```powershell
# 创建新的 conda 环境
conda create -n paddle python=3.10
conda activate paddle
```

---

## 4. 安装 PaddlePaddle

### CPU 版本

```powershell
pip install paddlepaddle
```

### GPU 版本（需要 NVIDIA GPU 和 CUDA）

首先确保已安装：
- [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit 11.8 或 12.x](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

然后安装 GPU 版本：

```powershell
# CUDA 11.8
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# CUDA 12.x
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
```

### 验证安装

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

---

## 5. 安装 PaddleDetection

### 克隆仓库

```powershell
# 选择一个目录
cd D:\_work\projects

# 克隆 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
```

### 安装依赖

```powershell
pip install -r requirements.txt
pip install pycocotools-windows  # Windows 特定
```

### 编译安装

```powershell
python setup.py install
```

### 验证安装

```powershell
python -c "import ppdet; print(ppdet.__version__)"
```

### 创建必要的配置目录

```powershell
# 在 PaddleDetection 目录下创建 autotrain 配置目录
mkdir configs\autotrain\jobs
mkdir configs\autotrain\training\default
mkdir configs\autotrain\training\user
mkdir configs\autotrain\models
```

---

## 6. 配置项目

### 克隆项目

```powershell
git clone https://github.com/HansenLYX0708/autotrain.git
cd autotrain
```

### 安装依赖

```powershell
bun install
```

### 配置数据库

项目使用 SQLite，无需额外安装。初始化数据库：

```powershell
bunx prisma generate
bunx prisma db push
```

### 创建环境文件

创建 `.env` 文件（如果不存在）：

```env
DATABASE_URL="file:./db/custom.db"
```

---

## 7. 运行项目

### 启动开发服务器

```powershell
bun run dev
```

项目将在 `http://localhost:3000` 运行。

### 在浏览器中打开

打开浏览器访问 `http://localhost:3000`

### 配置系统路径

1. 进入 **Settings** 页面
2. 配置以下路径：
   - **Python Path**: `C:\Python310\python.exe` 或 conda 环境的 python 路径
   - **PaddleDetection Path**: `D:\_work\projects\PaddleDetection`
   - **PaddleClas Path**: （如果使用 PaddleClas）

---

## 常见问题

### Q1: Bun 安装失败

尝试使用 npm 安装：
```powershell
npm install -g bun
```

或手动下载：https://github.com/oven-sh/bun/releases

### Q2: PaddlePaddle GPU 版本无法识别 GPU

1. 确认 NVIDIA 驱动已正确安装
2. 确认 CUDA 版本与 PaddlePaddle 版本匹配
3. 运行诊断：
```powershell
python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### Q3: pycocotools 安装失败

Windows 上需要使用特殊版本：
```powershell
pip install pycocotools-windows
```

或：
```powershell
pip install pycocotools -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html
```

### Q4: Prisma 命令失败

确保已安装 Prisma CLI：
```powershell
bun add prisma
bunx prisma generate
```

### Q5: 端口被占用

检查并结束占用端口的进程：
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## 推荐的目录结构

```
D:\_work\projects\
├── autotrain\              # 本项目
│   ├── src\
│   ├── prisma\
│   ├── db\
│   └── ...
├── PaddleDetection\        # PaddleDetection
│   ├── configs\
│   │   └── autotrain\
│   │       ├── jobs\
│   │       ├── training\
│   │       └── models\
│   ├── output\
│   └── ...
└── PaddleClas\             # PaddleClas（可选）
```

---

## 开发工具推荐

1. **VS Code** - 代码编辑器
   - 扩展：ESLint, Prettier, Prisma, Python

2. **PowerShell 7** - 更好的终端体验
   ```powershell
   winget install Microsoft.PowerShell
   ```

3. **Windows Terminal** - 多标签终端
   ```powershell
   winget install Microsoft.WindowsTerminal
   ```

4. **Git for Windows** - 版本控制
   ```powershell
   winget install Git.Git
   ```

---

## 一键安装脚本

创建 `install.ps1` 文件：

```powershell
# install.ps1 - Windows 11 环境安装脚本

Write-Host "=== Auto Training Platform 环境安装 ===" -ForegroundColor Green

# 检查 Node.js
Write-Host "检查 Node.js..." -ForegroundColor Yellow
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "安装 Node.js..." -ForegroundColor Yellow
    winget install OpenJS.NodeJS.LTS
} else {
    Write-Host "Node.js 已安装: $(node --version)" -ForegroundColor Green
}

# 检查 Bun
Write-Host "检查 Bun..." -ForegroundColor Yellow
if (!(Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Host "安装 Bun..." -ForegroundColor Yellow
    powershell -c "irm bun.sh/install.ps1 | iex"
} else {
    Write-Host "Bun 已安装: $(bun --version)" -ForegroundColor Green
}

# 检查 Python
Write-Host "检查 Python..." -ForegroundColor Yellow
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "安装 Python 3.10..." -ForegroundColor Yellow
    winget install Python.Python.3.10
} else {
    Write-Host "Python 已安装: $(python --version)" -ForegroundColor Green
}

# 安装项目依赖
Write-Host "安装项目依赖..." -ForegroundColor Yellow
bun install

# 初始化数据库
Write-Host "初始化数据库..." -ForegroundColor Yellow
bunx prisma generate
bunx prisma db push

Write-Host "`n=== 安装完成 ===" -ForegroundColor Green
Write-Host "运行 'bun run dev' 启动项目" -ForegroundColor Cyan
```

运行脚本：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install.ps1
```
