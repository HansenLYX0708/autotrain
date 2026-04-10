# Auto Training Platform 用户手册

## 目录

1. [系统简介](#1-系统简介)
2. [环境要求](#2-环境要求)
3. [安装教程](#3-安装教程)
   - 3.1 [安装 Node.js](#31-安装-nodejs)
   - 3.2 [安装 Bun](#32-安装-bun)
   - 3.3 [安装 Python](#33-安装-python)
   - 3.4 [安装 PaddlePaddle](#34-安装-paddlepaddle)
   - 3.5 [安装 PaddleDetection](#35-安装-paddledetection)
   - 3.6 [配置项目](#36-配置项目)
4. [快速开始](#4-快速开始)
5. [功能使用指南](#5-功能使用指南)
   - 5.1 [用户管理](#51-用户管理)
   - 5.2 [项目管理](#52-项目管理)
   - 5.3 [数据集管理](#53-数据集管理)
   - 5.4 [模型管理](#54-模型管理)
   - 5.5 [训练任务](#55-训练任务)
6. [系统设置](#6-系统设置)
7. [常见问题](#7-常见问题)
8. [故障排除](#8-故障排除)

---

## 1. 系统简介

Auto Training Platform 是一个基于 Web 的目标检测与图像分类模型训练平台，基于以下技术栈构建：

- **前端**: Next.js 16 + TypeScript 5 + Tailwind CSS 4 + shadcn/ui
- **后端**: Next.js API Routes + Prisma ORM + SQLite
- **深度学习框架**: PaddlePaddle + PaddleDetection/PaddleClas
- **运行时**: Bun (JavaScript/TypeScript 运行时)

### 主要功能

- **用户管理**: 支持多用户，基于角色的访问控制（管理员/普通用户）
- **数据集管理**: 支持 COCO 和 Labelme 格式数据集的上传和转换
- **项目管理**: 组织和管理不同的训练项目
- **模型管理**: 预置多种目标检测和分类模型架构
- **训练任务**: 可视化创建和管理训练任务，支持 GPU 训练
- **实时监控**: 训练过程实时日志和指标监控
- **存储配额**: 用户级别的存储空间管理

---

## 2. 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11 (64位) | Windows 11 (64位) |
| CPU | 4核处理器 | 8核或更高 |
| 内存 | 8GB | 16GB 或更高 |
| 磁盘空间 | 20GB 可用空间 | 50GB+ SSD |
| GPU | 可选（CPU训练较慢） | NVIDIA GPU (CUDA 11.8+) |

### 软件要求

- Node.js 18.x 或更高版本
- Bun 1.0 或更高版本
- Python 3.8 - 3.12（推荐 3.10）
- Git

---

## 3. 安装教程

### 3.1 安装 Node.js

#### 方法一：使用 winget（推荐）

```powershell
winget install OpenJS.NodeJS.LTS
```

#### 方法二：官方安装包

1. 访问 [Node.js 官网](https://nodejs.org/)
2. 下载 **LTS 版本**（推荐 20.x 或更高）
3. 运行安装程序，按提示完成安装
4. 验证安装：

```powershell
node --version
npm --version
```

---

### 3.2 安装 Bun

Bun 是本项目使用的 JavaScript 运行时和包管理器。

```powershell
powershell -c "irm bun.sh/install.ps1 | iex"
```

验证安装：

```powershell
bun --version
```

---

### 3.3 安装 Python

#### 方法一：使用官方安装包

1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载 **Python 3.10**
3. **重要**：安装时勾选 **"Add Python to PATH"**
4. 选择 **"Customize installation"** 并确保 **pip** 被勾选
5. 验证安装：

```powershell
python --version
pip --version
```

#### 方法二：使用 Anaconda/Miniconda（推荐用于深度学习）

1. 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. 创建专用环境：

```powershell
conda create -n paddle python=3.10
conda activate paddle
```

---

### 3.4 安装 PaddlePaddle

#### CPU 版本

```powershell
pip install paddlepaddle
```

#### GPU 版本（需要 NVIDIA GPU）

前置要求：
- [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

安装命令：

```powershell
# CUDA 11.8
pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA 12.x
pip install paddlepaddle-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证安装：

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

---

### 3.5 安装 PaddleDetection

#### 步骤 1：克隆仓库

选择一个目录（如 `D:\_work\projects`）：

```powershell
cd D:\_work\projects

# 克隆 PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection

# 切换到稳定版本（推荐）
git checkout release/2.6
```

#### 步骤 2：安装依赖

```powershell
pip install -r requirements.txt
pip install pycocotools-windows
```

#### 步骤 3：验证安装

```powershell
python -c "import ppdet; print(ppdet.__version__)"
```

#### 步骤 4：创建配置目录

```powershell
mkdir configs\autotrain\jobs
mkdir configs\autotrain\training\default
mkdir configs\autotrain\training\user
mkdir configs\autotrain\models
```

---

### 3.6 配置项目

#### 步骤 1：获取项目代码

```powershell
cd D:\_work\projects
git clone <项目仓库地址> autotrain
cd autotrain
```

#### 步骤 2：安装项目依赖

```powershell
bun install
```

#### 步骤 3：初始化数据库

```powershell
bunx prisma generate
bunx prisma db push
```

#### 步骤 4：创建环境文件

在项目根目录创建 `.env.local` 文件：

```env
DATABASE_URL="file:./db/custom.db"
NEXT_PUBLIC_API_URL="http://localhost:3000"
```

#### 步骤 5：启动项目

```powershell
bun run dev
```

项目将在 `http://localhost:3000` 运行。

#### 步骤 6：配置系统路径

1. 打开浏览器访问 `http://localhost:3000`
2. 注册第一个管理员账号（第一个注册的用户自动成为管理员）
3. 登录系统
4. 进入 **Settings** 页面
5. 配置以下路径：

| 配置项 | 示例值 | 说明 |
|--------|--------|------|
| Python Path | `C:\Python310\python.exe` | Python 可执行文件路径 |
| PaddleDetection Path | `D:\_work\projects\PaddleDetection` | PaddleDetection 根目录 |
| PaddleClas Path | （可选） | 如使用 PaddleClas |
| User Database Path | `D:\_work\database\users` | 用户数据存储路径 |
| User Configs Path | `D:\_work\configs\users` | 用户配置存储路径 |

**查找 Python 路径**：

```powershell
where python
# 或
(Get-Command python).Source
```

---

## 4. 快速开始

### 4.1 创建数据集

1. 进入 **Datasets** 页面
2. 点击 **Upload Dataset** 按钮
3. 选择数据集格式（COCO 或 Labelme）
4. 输入数据集名称
5. 选择或拖放数据集文件/文件夹
6. 点击上传

### 4.2 创建项目

1. 进入 **Projects** 页面
2. 点击 **Create Project** 按钮
3. 输入项目名称和描述
4. 选择框架（PaddleDetection 或 PaddleClas）
5. 点击创建

### 4.3 创建训练任务

1. 进入 **Jobs** 页面
2. 点击 **Create Job** 按钮
3. 填写任务信息：
   - 选择项目
   - 选择数据集
   - 选择模型
   - 配置训练参数（epoch, batch size, 学习率等）
   - 选择 GPU
4. 点击创建

### 4.4 启动训练

1. 在 Jobs 列表中找到创建的任务
2. 点击 **Start** 按钮
3. 在 **Training** 页面查看实时日志和指标

---

## 5. 功能使用指南

### 5.1 用户管理

#### 管理员功能

- 进入 **Users** 页面（仅管理员可见）
- 可以：
  - 创建新用户
  - 编辑用户信息
  - 重置用户密码
  - 设置用户存储配额
  - 启用/禁用用户

#### 普通用户

- 可修改自己的密码
- 只能查看自己的数据

#### 存储配额

管理员可为每个用户设置存储空间配额。当用户空间不足时：
- 无法上传新数据集
- 无法执行 labelme→coco 转换
- 无法创建新的训练任务

系统会提示用户联系管理员扩容或删除数据。

### 5.2 项目管理

项目是组织训练任务的容器。

**创建项目**：
1. 点击 **Create Project**
2. 输入名称和描述
3. 选择框架：
   - **PaddleDetection**: 目标检测
   - **PaddleClas**: 图像分类

**项目详情**：
- 查看项目关联的所有训练任务
- 查看项目使用的数据集
- 编辑或删除项目

### 5.3 数据集管理

支持两种格式：

#### COCO 格式

标准的目标检测数据集格式，包含：
- `data/train/` - 训练图片
- `data/val/` - 验证图片
- `data/annotations/` - 标注文件（instance_train.json, instance_val.json）

#### Labelme 格式

Labelme 标注工具生成的格式，包含：
- `data/imgs/` - 图片文件
- `data/jsons/` - 标注文件（每张图片对应一个 .json）

#### Labelme → COCO 转换

1. 在 Datasets 页面点击 **Convert Labelme to COCO**
2. 选择要转换的 Labelme 数据集
3. 输入新数据集名称
4. 设置训练/验证/测试比例（默认 70%/20%/10%）
5. 点击转换

**注意**：转换前系统会检查存储配额。

### 5.4 模型管理

预置了多种模型架构：

#### 目标检测模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| PP-YOLOE+ | 高性能实时检测 | 通用检测 |
| RT-DETR | 端到端检测 | 高精度需求 |
| YOLOv8 | 社区流行模型 | 快速部署 |

#### 图像分类模型

| 模型 | 描述 |
|------|------|
| ResNet | 经典分类网络 |
| PP-LCNet | 轻量级分类网络 |

**创建自定义模型配置**：
1. 在 Models 页面点击 **Create Model**
2. 选择架构、Backbone、Neck、Head
3. 设置类别数量
4. 保存配置

### 5.5 训练任务

#### 创建训练任务

1. 进入 Jobs 页面
2. 点击 **Create Job**
3. 配置参数：
   - **基础信息**: 名称、项目、数据集、模型
   - **训练配置**: Epoch、Batch Size、学习率等
   - **GPU 设置**: 选择 GPU 设备（支持多卡训练）
   - **高级选项**: 混合精度 (AMP)、VisualDL 日志

#### 监控训练

1. 在 Training 页面查看所有运行中的任务
2. 实时查看：
   - 训练日志
   - Loss 曲线
   - 学习率变化
   - GPU 内存使用
   - ETA 预估时间

#### 模型导出

训练完成后，可以导出模型：
1. 在 Job 详情页点击 **Export**
2. 选择导出格式（Paddle/ONNX）
3. 下载模型文件

---

## 6. 系统设置

进入 **Settings** 页面配置系统级参数：

### 路径配置

| 配置项 | 说明 |
|--------|------|
| Python Path | Python 解释器路径 |
| PaddleDetection Path | PaddleDetection 安装路径 |
| PaddleClas Path | PaddleClas 安装路径（可选） |
| User Database Path | 用户数据根目录 |
| User Configs Path | 用户配置根目录 |
| Conda Environment | Conda 环境名称（如使用） |

### GPU 配置

配置每个 GPU 对应的 Python 路径（多环境场景）：

```json
{
  "0": "C:\\Python310\\python.exe",
  "1": "C:\\Python310\\python.exe"
}
```

---

## 7. 常见问题

### Q1: 如何检查存储空间使用情况？

管理员可以在 Users 页面查看每个用户的存储使用情况。
普通用户尝试超出配额的操作时会收到提示。

### Q2: 训练任务显示 "存储空间不足"

训练任务需要预留 5GB 空间存储模型和日志。请：
1. 删除不需要的数据集或旧模型
2. 联系管理员增加存储配额

### Q3: Labelme 转换失败

可能原因：
- 存储空间不足
- Labelme 图片路径配置错误
- JSON 文件格式不正确

检查方法：
1. 确认 storage quota 足够
2. 检查 labelmeImagesPath 和 labelmeAnnotationsPath 是否正确
3. 验证 JSON 文件可以被正常解析

### Q4: 上传数据集时提示空间不足

上传前系统会检查剩余空间。如需上传，请：
1. 删除旧数据集
2. 或联系管理员增加配额

### Q5: GPU 训练无法启动

检查步骤：
1. 确认 NVIDIA 驱动已安装：`nvidia-smi`
2. 确认 CUDA 版本与 PaddlePaddle 匹配
3. 检查 GPU Python 路径配置是否正确
4. 查看 Training 页面日志获取详细错误

---

## 8. 故障排除

### 启动问题

**问题**: `bun run dev` 失败

**解决**:
```powershell
# 清除缓存
bun pm cache rm
# 重新安装依赖
bun install
# 重新生成 Prisma 客户端
bunx prisma generate
```

### 数据库问题

**问题**: 数据库 Schema 错误

**解决**:
```powershell
# 重置数据库（会删除数据）
bunx prisma migrate reset

# 或推送 Schema 变更
bunx prisma db push
```

### 训练问题

**问题**: 训练进程无法启动

**检查清单**:
1. Python 路径配置正确
2. PaddleDetection 路径配置正确
3. GPU Python 映射配置正确（如使用 GPU）
4. 有足够的存储空间（至少 5GB）

### 端口占用

```powershell
# 查找占用 3000 端口的进程
netstat -ano | findstr :3000
# 结束进程
taskkill /PID <PID> /F
```

### PowerShell 执行策略

**问题**: "无法加载文件，因为在此系统上禁止运行脚本"

**解决**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 附录

### 推荐目录结构

```
D:\_work\
├── projects\
│   ├── autotrain\              # 本项目
│   │   ├── src\
│   │   ├── prisma\
│   │   ├── db\
│   │   └── docs\
│   ├── PaddleDetection\        # PaddleDetection
│   │   ├── configs\autotrain\   # 训练配置
│   │   └── ppdet\
│   └── PaddleClas\              # PaddleClas（可选）
├── database\                    # 用户数据
│   └── users\
└── configs\                     # 用户配置
    └── users\
```

### 有用的命令

```powershell
# 检查 Python 路径
where python
(Get-Command python).Source

# 检查 GPU
nvidia-smi

# 检查 PaddlePaddle
python -c "import paddle; paddle.utils.run_check()"

# 检查端口占用
netstat -ano | findstr :3000

# 查看磁盘空间
Get-PSDrive C | Select-Object Used,Free

# 计算目录大小
(Get-ChildItem -Path D:\_work\database -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
```

### 技术支持

遇到问题？请：
1. 查看本手册的故障排除章节
2. 检查应用程序日志
3. 联系系统管理员

---

**文档版本**: 1.0  
**最后更新**: 2026-04-10  
**适用系统版本**: Auto Training Platform v0.2.0+
