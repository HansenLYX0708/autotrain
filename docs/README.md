# 文档索引

Hawkeye+ AutoTrain Platform 的全部文档。项目总览见 [../README.md](../README.md)。

## 按角色查阅

### 我要**使用**这个平台

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [USER_MANUAL.md](USER_MANUAL.md) | 用户手册：环境要求、安装、快速开始、逐模块操作、常见问题 |
| 2 | [FEATURES.md](FEATURES.md) | 功能模块说明：每个页面能做什么、背后发生了什么、故障速查表 |

应用内顶栏的 **Manual** 按钮也能直接打开一份同源手册。

### 我要**部署**这个平台

| 文档 | 适用场景 |
|---|---|
| [../deployment/windows-offline/README.md](../deployment/windows-offline/README.md) | **Windows 10/11 全框架完全离线部署**：打包、安装、启停、备份、验证。生产环境首选 |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | 在线环境下逐步安装 Node/Bun/Python/PaddlePaddle/PaddleDetection |
| [../USER_SYSTEM_SETUP.md](../USER_SYSTEM_SETUP.md) | 用户系统的数据库初始化与首个 admin 账号 |

### 我要**开发/修改**这个平台

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [../AGENTS.md](../AGENTS.md) | **必读**。踩坑清单、验证流程、代码约定、单位约定、各框架陷阱 |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | 架构与实现原理：框架抽象层、三段配置合并、命令生成、训练进程生命周期、日志解析、鉴权、路径与配额、已知限制 |
| 3 | [DATA_MODEL.md](DATA_MODEL.md) | Prisma 数据模型逐字段说明、实体关系、删除语义、单位约定 |
| 4 | [API_REFERENCE.md](API_REFERENCE.md) | 全部 46 个 REST 路由：方法、鉴权、入参、行为、返回结构 |

### 我要动某个**具体框架**

| 文档 | 内容 |
|---|---|
| [../torchtrain/README.md](../torchtrain/README.md) | 内置 PyTorch 训练器（TorchDet / TorchSeg / TorchAnomaly）：CLI、支持的架构、与 Paddle 的有意差异、指标实现 |
| [ANOMALY_DETECTION_DESIGN.md](ANOMALY_DETECTION_DESIGN.md) | 无监督异常检测（TorchAnomaly）完整设计：anomalib 适配、数据集约定、配置形状、日志格式、实现状态与设计的差异 |

### 我要做**汇报**

| 文档 | 内容 |
|---|---|
| [Hawkeye_Plus_Overview.md](Hawkeye_Plus_Overview.md) | Executive briefing（幻灯片式）：项目背景、平台定位、端到端流程、价值总结，附技术实现备份页 |

> 注意：该文档的 "Real-Time Training Channel (Socket.io)" 一节描述的是**设想中的**实时通道。当前代码**没有** WebSocket，实时性靠前端轮询（Monitoring 5 秒、Dashboard 30 秒）实现，`mini-services/training-service/` 只是一个生成合成日志的未接入原型。

## 辅助工具

| 文档 | 内容 |
|---|---|
| [../tools/napari_seg/readme.md](../tools/napari_seg/readme.md) | 基于 napari 的分割标注工具 |
| [../tools/tem_analysis/readme.md](../tools/tem_analysis/readme.md) | TEM 图像几何测量与报告生成 |
| [../tools/tem_analysis/PLAN.md](../tools/tem_analysis/PLAN.md) | TEM 分析工具的设计与推进计划 |
| [../mini-services/training-service/README.md](../mini-services/training-service/README.md) | Socket.io 训练服务原型（**mock，未接入主应用**） |

## 其它

- [../worklog.md](../worklog.md) —— 开发工作日志，记录各阶段改动的来龙去脉
- [../prisma/schema.prisma](../prisma/schema.prisma) —— 数据模型源文件

## 文档维护约定

1. **文档描述代码的当前行为，而不是设想。** 若某处已知与代码不符（如上面 Overview 的 Socket.io 一节），必须就地标注，而不是留给读者踩坑。
2. **单位与字段名以源码为准。** 涉及 epoch / iter 的地方一律写清单位；字段名用代码里的原始拼写。
3. 改动行为时，**同步更新对应文档**：
   - 改 `src/lib/frameworks.ts`、配置合并、进程管理 → [ARCHITECTURE.md](ARCHITECTURE.md)
   - 改 `prisma/schema.prisma` → [DATA_MODEL.md](DATA_MODEL.md)
   - 增删改 API 路由 → [API_REFERENCE.md](API_REFERENCE.md)
   - 改界面交互 → [FEATURES.md](FEATURES.md)
   - 新发现的坑 → [../AGENTS.md](../AGENTS.md)
4. **不要用 PowerShell 的 `Set-Content` 写这些文件。** PowerShell 5.1 会按 ANSI 代码页往返，静默破坏所有非 ASCII 字符（本目录下的文档大量使用中文与 `—`、`·`、`✓`）。用编辑器，或用 Python 显式指定编码：
   ```python
   open(path, "w", encoding="utf-8", newline="").write(text)
   ```
