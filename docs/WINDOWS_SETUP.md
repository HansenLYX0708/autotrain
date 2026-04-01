# Windows 环境安装指南

本文档介绍如何在 Windows 上安装和运行 Auto Training Platform。

## 目录

1. [环境要求](#1-环境要求)
2. [安装 Node.js](#2-安装-nodejs)
3. [安装 Bun](#3-安装-bun)
4. [安装 Python](#4-安装-python)
5. [安装 PaddlePaddle](#5-安装-paddlepaddle)
6. [安装 PaddleDetection](#6-安装-paddledetection)
7. [配置项目](#7-配置项目)
8. [配置系统路径](#8-配置系统路径)
9. [运行项目](#9-运行项目)
10. [常见问题](#10-常见问题)

---

## 1. 环境要求

- Windows 10/11 (64位)
- 至少 8GB 内存（推荐 16GB）
- 至少 10GB 可用磁盘空间
- 可选：NVIDIA GPU（用于 GPU 训练）

---

## 2. 安装 Node.js

项目需要 Node.js 18.x 或更高版本。

### 方法一：使用官方安装包

1. 访问 [Node.js 官网](https://nodejs.org/)
2. 下载 **LTS 版本**（推荐 20.x 或更高）
3. 运行安装程序，按照提示完成安装
4. 验证安装：

```powershell
node --version
npm --version
```

### 方法二：使用 winget（推荐）

```powershell
winget install OpenJS.NodeJS.LTS
```

---

## 3. 安装 Bun

Bun 是本项目使用的 JavaScript 运行时和包管理器。

### 方法一：使用 PowerShell 安装（推荐）

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

## 4. 安装 Python

PaddleDetection 需要 Python 3.8-3.12（推荐 3.10）。

### 方法一：使用官方安装包

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 Python 3.10
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
2. 创建专用环境：

```powershell
conda create -n paddle python=3.10
conda activate paddle
```

---

## 5. 安装 PaddlePaddle

### CPU 版本

```powershell
pip install paddlepaddle
```

### GPU 版本（需要 NVIDIA GPU 和 CUDA）

前置要求：
- [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit 11.8 或 12.x](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

安装命令：

```powershell
# CUDA 11.8
pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 12.x
pip install paddlepaddle-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

---

## 6. 安装 PaddleDetection

### 6.1 克隆仓库

打开 PowerShell，选择一个目录（如 `D:\_work\projects`）：

```powershell
cd D:\_work\projects

# 克隆 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection

# 切换到稳定版本（推荐）
git checkout release/2.6
```

### 6.2 安装依赖

```powershell
pip install -r requirements.txt
pip install pycocotools-windows
```

### 6.3 验证安装

```powershell
python -c "import ppdet; print(ppdet.__version__)"
```

### 6.4 创建必要的配置目录

在 PaddleDetection 目录下执行：

```powershell
mkdir configs\autotrain\jobs
mkdir configs\autotrain\training\default
mkdir configs\autotrain\training\user
mkdir configs\autotrain\models
```

---

## 7. 配置项目

### 7.1 拷贝项目

### 7.2 安装项目依赖

```powershell
bun install
```

### 7.3 初始化数据库

项目使用 SQLite，无需额外安装。

```powershell
bunx prisma generate
bunx prisma db push
```

### 7.4 创建环境文件

在项目根目录创建 `.env.local` 文件：

```env
DATABASE_URL="file:./db/custom.db"
NEXT_PUBLIC_API_URL="http://localhost:3000"
```

---

## 8. 配置系统路径

启动项目后，需要在 Web 界面中配置以下路径：

1. 打开浏览器访问 `http://localhost:3000`
2. 登录系统（首次使用需注册）
3. 进入 **Settings** 页面
4. 配置以下路径：

| 配置项 | 示例值 | 说明 |
|--------|--------|------|
| Python Path | `C:\Python310\python.exe` | Python 可执行文件路径 |
| PaddleDetection Path | `D:\_work\projects\PaddleDetection` | PaddleDetection 根目录 |
| PaddleClas Path | （可选） | 如使用 PaddleClas |

### 路径查找方法

**Python 路径**：
```powershell
where python
# 或
(Get-Command python).Source
```

**PaddleDetection 路径**：
即克隆的 PaddleDetection 文件夹路径，如 `D:\_work\projects\PaddleDetection`

---

## 9. 运行项目

### 启动开发服务器

```powershell
bun run dev
```

项目将在 `http://localhost:3000` 运行。

### 生产环境部署

```powershell
bun run build
bun run start
```

---

## 10. 常见问题

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

### Q4: Prisma 命令失败

确保在项目目录下运行：
```powershell
bun add prisma
bunx prisma generate
bunx prisma db push
```

### Q5: 端口被占用

检查并结束占用端口的进程：
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Q6: 提示 "无法加载文件，因为在此系统上禁止运行脚本"

以管理员身份运行 PowerShell，执行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 推荐的目录结构

```
D:\_work\projects\
├── autotrain\              # 本项目
│   ├── src\
│   ├── prisma\
│   ├── db\
│   ├── docs\
│   └── ...
├── PaddleDetection\        # PaddleDetection
│   ├── configs\
│   │   └── autotrain\      # 训练配置目录
│   ├── ppdet\
│   └── ...
└── PaddleClas\             # PaddleClas（可选）
```

---

## 开发工具推荐

1. **VS Code** - 代码编辑器
   
- 推荐扩展：ESLint, Prettier, Prisma, Python
   
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

