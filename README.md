# Hawkeye+ AutoTrain Platform

面向工业视觉的深度学习训练平台。用浏览器完成 **数据集管理 → 模型/训练配置 → 启动训练 → 实时监控 → 评估/推理 → 模型发布** 的全流程，无需手写 YAML、无需记 CLI 参数。

平台本身不实现训练算法：它把三份独立编写的 YAML 配置深度合并成一份任务配置，然后以子进程方式调用对应框架的 `tools/train.py`，再把子进程 stdout 解析成结构化的训练日志入库、绘图。

```
数据集配置 ─┐
训练配置   ─┼─► 深度合并的 YAML ─► <框架>/tools/train.py ─► stdout ─► TrainingLog 行 ─► 图表/进度
模型配置   ─┘
```

## 支持的框架

`project.framework` 选定一个框架，其余行为（配置模式、CLI 方言、检查点布局、数据集格式、进度单位）全部由 `src/lib/frameworks.ts` 中的 `FRAMEWORK_META` 推导。

| Framework | 运行时 | 任务 | 代码库 | 训练长度单位 |
|---|---|---|---|---|
| `PaddleDetection` | PaddlePaddle | 目标检测 / 实例分割 | 外部，`paddleDetectionPath` | epoch |
| `PaddleClas` | PaddlePaddle | 图像分类 | 外部，`paddleClasPath` | epoch |
| `PaddleSeg` | PaddlePaddle | 语义分割 | 外部，`paddleSegPath` | iter |
| `TorchDet` | PyTorch | 目标检测 | 内置 `torchtrain/`，`torchPath` | epoch |
| `TorchSeg` | PyTorch | 语义分割 | 内置 `torchtrain/`，`torchPath` | iter |
| `TorchAnomaly` | PyTorch + anomalib | 无监督异常检测 | 内置 `torchtrain/`，`torchPath` | step |

## 技术栈

- **Next.js 16**（App Router + Turbopack）、React 19、TypeScript 5
- **Tailwind CSS 4** + shadcn/ui（Radix UI）、Recharts、Framer Motion
- **Prisma 6 + SQLite** 作为唯一持久层
- **yaml**（`parseDocument` AST 级合并，保留 PaddleDetection 自定义标签）
- Node `child_process.spawn` 驱动 Python 训练进程

前端是**单页应用**：只有 `src/app/page.tsx` 一个路由，侧边栏在 `src/components/pages/*` 之间切换，所有数据经 `/api/*` 读写。

## 快速开始

```bash
npm install                 # 或 bun install
npx prisma generate         # 生成 Prisma Client
npm run db:push             # 按 schema 建/更新表（本项目用 db push，不用 migrate）
npm run dev                 # http://localhost:3000
```

首次启动后：

1. 注册第一个账号 —— **第一个注册的用户自动成为 admin**。
2. 以 admin 登录，进入 **Settings** 填写框架仓库路径、Python 解释器映射、用户配置目录与用户数据目录。
3. 可选：调用 `POST /api/init-defaults` 或在 Dashboard 点击初始化，生成一套示例项目/数据集/模型/训练配置。

> Windows 10/11 全框架、完全离线的一键部署流程与打包脚本见
> [deployment/windows-offline/README.md](deployment/windows-offline/README.md)。

### 常用命令

```bash
npm run dev          # next dev -p 3000
npm run build        # next build && node copy.js（standalone 产物）
npm run lint         # eslint .
npm run db:push      # prisma db push
npm run db:generate  # prisma generate
npx tsc --noEmit     # 单独类型检查
```

注意 `next.config.ts` 设置了 `typescript.ignoreBuildErrors: true`——**构建成功不代表类型检查通过**，务必单独跑 `tsc`。

### 数据库位置

`DATABASE_URL` 必须是**完全限定路径**（如 `file:D:/path/to/custom.db`）。相对路径 `file:./x.db` 会相对 `prisma/` 目录解析，而 `file:/foo/bar` 在 Windows 上相对**当前盘符**解析——两种情况 SQLite 都会静默新建一个空库。`src/lib/db.ts` 会在文件不存在时打印醒目告警。详情见 [AGENTS.md](AGENTS.md#database-location)。

## 目录结构

```
src/
├── middleware.ts             # 仅用于吞掉 /socket.io 轮询请求
├── app/
│   ├── page.tsx              # 唯一页面：侧边栏 + 页面切换
│   ├── layout.tsx            # 根布局、主题
│   └── api/                  # REST 路由（见 docs/API_REFERENCE.md）
├── components/
│   ├── pages/                # 每个侧边栏条目一个大组件
│   ├── ui/                   # shadcn/ui 基础组件
│   └── *.tsx                 # 跨页共享组件
├── contexts/auth-context.tsx # 登录态
└── lib/                      # 框架注册表、YAML 生成/合并、日志解析、鉴权
torchtrain/                   # 内置 PyTorch 训练器（TorchDet/TorchSeg/TorchAnomaly）
prisma/schema.prisma          # 数据模型
tools/                        # 独立 Python 工具（napari 标注、TEM 分析）
scripts/                      # 一次性维护/校验脚本
deployment/windows-offline/   # 离线部署打包与安装脚本
docs/                         # 全部文档
```

## 文档

完整文档索引见 **[docs/README.md](docs/README.md)**。最常用的几篇：

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构与实现原理：配置三段合并、训练进程生命周期、日志解析、路径与存储、鉴权模型 |
| [docs/FEATURES.md](docs/FEATURES.md) | 按页面讲每个功能模块能做什么、怎么操作 |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | 全部 REST 接口：方法、入参、鉴权要求、返回结构 |
| [docs/DATA_MODEL.md](docs/DATA_MODEL.md) | Prisma 数据模型逐字段说明与单位约定 |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | 面向使用者的安装与操作手册 |
| [AGENTS.md](AGENTS.md) | 开发者必读：踩坑清单、验证流程、代码约定 |
