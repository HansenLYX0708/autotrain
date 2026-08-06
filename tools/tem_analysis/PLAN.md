# TEM 分割图几何分析 —— 实现计划

> **状态：已实现。** 落地过程中有四处偏离本计划，都是因为实测发现原方案不成立，
> 详见文末「§8 实现中的修正」。使用说明见 <readme.md>。


对 PaddleSeg 输出的伪彩色分割图做几何量测。参考样本：

```
G:\datasets\TEM\1300kgroup1\temp_out\pseudo_color_prediction\751845_RMF 24(12,15)_1300kx_FOV_76.37nm.png
```

## 0. 前置事实（已核对）

### 0.1 类别与调色板

调色板来自 `tools/napari_seg/annotate.py::voc_colormap()`：PASCAL VOC 色表**去掉首个黑色项**，
因此 `class_id == 调色板索引`，`class 0` 是 `(128,0,0)` 而不是黑色。
`pseudo_color_prediction/` 写出的是 8-bit 调色板 PNG，像素值即 class id。

| id | 类别 | RGB | 样本中的位置 |
|----|------|-----|------|
| 0  | `_background_` | (128,0,0) 暗红 | 大面积底色 |
| 1  | `SAF_Ru_L` | (0,128,0) 绿 | 左侧水平带 |
| 2  | `SAF_Ru_R` | (128,128,0) 橄榄 | 右侧水平带 |
| 3  | `MgO_L` | (0,0,128) 深蓝 | 左侧折弯带 |
| 4  | `MgO_R` | (128,0,128) 紫 | 右侧折弯带 |
| 5  | `MgO_C` | (0,128,128) 青 | 中心细横条 |
| 6  | `Non_mag` | (128,128,128) 灰 | 中心灰块 |
| 7  | `Milling_L` | (64,0,0) 深红 | 左侧曲线 |
| 8  | `Milling_R` | (192,0,0) 亮红 | 右侧曲线 |
| 9  | `Leveling` | (64,128,0) 黄绿 | 底部长直线 |
| 10 | `Block_U` | (192,128,0) 橙 | 顶部块 |
| 11 | `Block_D` | (64,0,128) 靛 | 中下块 |

### 0.2 标尺校准存疑（需确认）

`76.37 nm / 1024 px = 0.07458 nm/px`，与给定的 `0.0755 nm/px` 相差 1.2%。
`0.0755` 对应有效宽度 ~1011.5 px（可能 FOV 只覆盖裁剪后的宽度）。
实现上 `--scale-nm` 默认取 **0.0755**（按需求），但该值线性影响所有 nm 量。
JSON 里会把 `scale_nm_per_px` 与 `image_width_px` 一并记录，便于事后换算。

### 0.3 坐标与符号约定

图像坐标系 y 向下。为消除歧义，所有角度同时给出两种：

- `angle_deg_image`：`atan2(dy, dx)`，直接用图像坐标（y 向下），顺时针为正
- `angle_deg_math`：`atan2(-dy, dx)`，数学坐标（y 向上），逆时针为正

所有 y 方向偏差：**正 = 更靠下（图像中更低）**。

---

## 1. 目录结构

```
tools/tem_analysis/
  PLAN.md              本文件
  readme.md            使用说明
  requirements.txt
  analyze.py           CLI 入口 + 编排
  labelmap.py          掩膜 -> class id -> 各类布尔掩膜
  skeleton.py          Guo-Hall 细化 / 剪枝 / 有序折线 / 端点
  geometry.py          直线拟合、角度、R2、弧长样条、解析曲率
  rectify.py           Leveling 角度 -> 最近邻旋转 + 坐标变换
  measures/
    __init__.py        MEASURES 注册表
    leveling.py        需求 1
    interfaces.py      需求 2 + 3
    mgo_c.py           需求 4
    milling.py         需求 5
    non_mag.py         需求 6
    corner_offsets.py  需求 7
  report.py            JSON schema / CSV 扁平化 / 叠加图渲染
  selftest.py          合成几何回归校验
```

模块化原则：每个需求一个模块，导出统一签名

```python
def measure(ctx: AnalysisContext) -> dict   # 返回可 JSON 序列化的结果
```

`AnalysisContext` 持有：旋转后 label 数组、各类掩膜访问器（带缓存）、骨架缓存、
`scale_nm`、参数 namespace、坐标反变换、`warn()` 收集器。
`analyze.py` 逐个调用并 try/except，单项失败只写 `null` + warning，不影响批处理。

---

## 2. 模块细节

### 2.1 `labelmap.py`

- `load_label_map(path) -> (ids: uint8[H,W], note: str)`
  - `P` / `L` 模式：`np.array(img)` 直接当 id
  - `RGB` / `RGBA`：吸附到调色板最近色；离色率 > 5% 则报错（这是
    `added_prediction/` 那种混合预览图，不是标签图）
  - 复用 `tools/napari_seg/annotate.py` 的 `voc_colormap` / `build_palette` /
    `rgb_to_indices`，通过 `importlib.util.spec_from_file_location` 按路径加载
    （该文件的 `import napari` 都在函数体内，顶层导入安全），避免调色板逻辑二次实现
- `CLASS_NAMES`：默认写死本项目 12 类顺序；`--classes` 可覆盖；若掩膜同目录/上级
  存在 `class_names.txt` 则优先读取
- `class_mask(ids, name, *, min_area, largest_only, prefer_near) -> bool[H,W]`
  （`prefer_near` 见 §8.1e）
  - `min_area`（默认 30 px）：`scipy.ndimage.label` + 面积过滤去噪点
  - `largest_only`（默认 True）：只留最大连通域；丢弃任何成分都记 warning
    并附上被丢面积，便于发现分割崩坏的图
- `255`（ignore）与旋转填充值统一视为无效像素

### 2.2 `skeleton.py`

- `guo_hall(mask) -> bool[H,W]`
  - 优先 `cv2.ximgproc.thinning(src, thinningType=cv2.ximgproc.THINNING_GUOHALL)`
  - 无 `opencv-contrib-python` 时走纯 NumPy 回退：Guo-Hall 两个子迭代，
    用 8 个平移邻居平面向量化计算 `C, N1, N2, O` 条件，循环至无像素被删。
    1024² 掩膜约 0.2 s，输出与 OpenCV 一致（selftest 中做一致性校验）
- `prune(skel) -> bool[H,W]` + `longest_path(skel) -> Path`
  - 建 8 邻域像素图，边权 1 / √2
  - 双向 BFS（Dijkstra，因为边权不等）取两个 degree-1 端点间最长测地路径，
    剔除毛刺与分叉；若骨架含环，退化为去环后再取最长路径
- `Path` 数据类：`pts: float[N,2] (x,y)`、`s: float[N]` 累积弧长（px）、
  `tangent(i)`（中心差分 + 局部平滑）、`endpoint('xmin'|'xmax'|'ymin'|'ymax')`
- `walk_from(endpoint, max_s)`：从指定端点沿**弧长**取窗口（供需求 3 用）
- 端点内缩的已知偏差：形态学骨架会从区域尖端回缩约半个局部带宽，故 `a1` 略在
  SAF_Ru_L 真实尖端内侧。提供可选 `--endpoint-extend region`：沿端点局部切线
  外推到该区域的极值列。默认 `none`（忠实 Guo-Hall），使用哪种模式写进 JSON

### 2.3 `geometry.py`

- `fit_line_tls(pts) -> Line`：PCA / 全最小二乘，方向无偏，用于**所有角度**
- `fit_line_ols(pts, axis='y~x') -> (slope, intercept, r2, rmse)`：常规 R²，
  用于需求 4 的 `fit R2`；若点云更接近竖直（TLS 主轴与 x 轴夹角 > 45°）则改用
  `x~y` 并在结果里标注 `r2_axis`，同时发 warning
- `Line`：`point`, `direction`, `angle_deg_image/math`, `project(p)`,
  `signed_distance(p)`, `clip_to(p1, p2)`
- `fit_spline(pts, smooth) -> Spline`：按弧长重采样后 `scipy.interpolate.splprep(k=3, s=smooth)`
- `Spline.curvature(u)`：解析式 `κ = (x'y'' − x''y') / (x'² + y'²)^{3/2}`（带符号），
  `splev(der=1/2)` 求导，不做数值差分
- `trimmed_line_fit(pts, sigma=2.0, iters=5)`：迭代 2σ 剔除离群点后再拟合（需求 6 用）

### 2.4 `rectify.py`

1. `Leveling` 掩膜 → Guo-Hall → 剪枝 → `fit_line_tls` → `angle`
2. 绕图像中心旋转 `-angle`，`scipy.ndimage.rotate(order=0, reshape=True, cval=255)`
   （最近邻，保持 label 语义；填充 255 = ignore，与真实 background 0 区分开）
3. 保存正/反仿射矩阵，`to_orig(p)` 把任意点映射回原图坐标
4. `--no-rotate`：跳过旋转，`to_orig` 为恒等，下游全部在原图上跑

### 2.5 `measures/leveling.py`（需求 1）

输出：骨架点数、拟合直线（两端点）、`angle_deg_image`、`angle_deg_math`、
`r2`、`rmse_px/nm`、`rotation_applied`。旋转后重测一次作为自检
（残余角度应 ≈ 0，超过 0.05° 发 warning）。

### 2.6 `measures/interfaces.py`（需求 2 + 3）

**端点**（均取剪枝后骨架）

- `a1` = `SAF_Ru_L` 骨架 x 最大端点
- `a2` = `SAF_Ru_R` 骨架 x 最小端点
- `b1` = `MgO_C` 骨架 x 最小端点，`b2` = x 最大端点

**偏差**：`(b1 − a1)` 与 `(b2 − a2)` 的 `dx, dy`，px 与 nm 双份，带符号
（`dy > 0` = b 比 a 低）；另给 `dist`（欧氏距离）。

**5 nm 窗口内最大向下偏差**（基准 = 端点本身，按确认的口径）

- 从 `a1` 沿骨架**弧长**回走，取 `s ≤ 5 nm / scale`（`--window-nm`，默认 5）
- 每点 `dev_i = y_i − y_{a1}`（正 = 更低）
- 输出 `max_dev`（带符号最大值）、对应点坐标、该点弧长偏移 `s_nm`、
  窗口点数、以及 `max_abs_dev`（若全部在端点上方，`max_dev` 为负，此项便于对比）
- `a2` 侧镜像处理（向右回走）

### 2.7 `measures/mgo_c.py`（需求 4）

- `MgO_C` 剪枝骨架 → `fit_line_ols` 得 `r2` / `rmse`，`fit_line_tls` 得角度
- 输出拟合线段（投影 `b1`/`b2` 到直线上得两端点）、长度 nm、
  `angle_deg_image/math`、`r2`、`r2_axis`、`rmse_px/nm`、`max_residual_nm`
- 残差序列一并写入 JSON，便于判断是"整体倾斜"还是"局部起伏"

### 2.8 `measures/milling.py`（需求 5）

对 `Milling_L` / `Milling_R` 各跑一次：

1. **取轮廓**：`cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)`
   得到**有序**外轮廓（弧长样条需要顺序，简单的 erosion 差集给不出顺序）
2. **按骨架分侧**（对应"用骨架分割边缘，离骨架正向距离的保留"）
   - 剪枝骨架 → `cKDTree` 查每个轮廓点的最近骨架点索引 `i`
   - 该点局部切线 `t_i`，法向 `n_i = rot90(t_i)`，符号统一取**指向 `Non_mag` 质心**
     的一侧（即靠近器件中心的内侧）
   - 保留 `dot(p − skel_i, n_i) > 0` 的轮廓点
3. **取连续段**：在轮廓顺序上取保留点的最长连续 run，得到一条干净的开曲线
   （直接取全部保留点会在两端混入跨越点，破坏弧长参数化）
4. **拟合**：按弧长等间距重采样 → `splprep(k=3, s=smooth)`，
   `--milling-smooth` 默认 `n_pts * (1.0 px)²`（对应 ~1 px 标注噪声）
5. **曲率**：解析 `κ(s)`；在 `s ∈ [margin, L − margin]`（`--milling-edge-margin-nm`，
   默认 5 nm，避开样条端点伪影）且落在中段
   （`--milling-middle-frac`，默认 0.6，防止左右直段胜出）的范围内取 `|κ|` 最大
6. **输出**：`m1` / `m2` 点（旋转后 + 原图坐标）、`kappa_max_per_nm`、
   `radius_nm = 1/κ`、该点弧长位置、边缘点数、拟合 RMSE，
   以及**完整 `κ(s)` 采样序列**（便于不重跑就调参复核）

说明："中间曲线"没有严格定义，这里用"边距 + 中段占比"两个显式参数控制，
参数与 κ(s) 全量落盘，方便你按实际形貌重新划定。

> §2.9 已按需求变更改写，见 §8.1c / §8.1d。以下为原始计划，保留以对照。

### 2.9 `measures/block_d.py`（需求 6，已改为 `non_mag.py`）

- `Block_D` 下边缘：对每个有前景的列取 `max(y)`（比轮廓分侧更直接、更稳）
- `trimmed_line_fit`（迭代 2σ 剔离群列，`--blockd-trim` 可关）
- `blockD1` / `blockD2` = 拟合直线在**保留列**的最小 / 最大 x 处取值
- 输出两端点、`angle_deg_image/math`、长度 nm、`r2`、`rmse_nm`、剔除列数

---

## 3. `report.py`

**JSON**（`<out>/<stem>.json`）

```json
{
  "image": "...png",
  "image_size_px": [1024, 1024],
  "scale_nm_per_px": 0.0755,
  "tool_version": "1.0.0",
  "params": { "window_nm": 5.0, "endpoint_extend": "none", "...": "..." },
  "rotation": { "applied": true, "angle_deg_image": 0.31,
                "matrix": [[...],[...]], "output_size_px": [1030, 1030] },
  "results": {
    "leveling":    { "...": "..." },
    "interfaces":  { "a1": {}, "a2": {}, "b1": {}, "b2": {},
                     "a1_b1": {}, "a2_b2": {},
                     "saf_ru_l_dip": {}, "saf_ru_r_dip": {} },
    "mgo_c":       { "...": "..." },
    "milling_l":   { "...": "..." },
    "milling_r":   { "...": "..." },
    "non_mag":     { "Non_mag1": {}, "Non_mag2": {}, "...": "..." },
    "corner_offsets": { "non_mag1_m1": {}, "non_mag2_m2": {} }
  },
  "warnings": ["..."]
}
```

- 每个点统一为 `{ "x_px", "y_px", "x_orig_px", "y_orig_px" }`
- 每个距离统一给 `_px` 与 `_nm` 两个字段
- 失败项为 `null`，原因进 `warnings`

**CSV**（`--csv summary.csv`）：一行一图，列顺序固定（由 `report.CSV_COLUMNS`
声明，缺失填空），只含标量指标，序列类数据不进 CSV。

**叠加图**（`<out>/<stem>_overlay.png`）：在旋转后的伪彩图上画

- 各类剪枝骨架（白细线）
- 拟合直线 / 样条（虚线）
- 标注点 `a1 a2 b1 b2 m1 m2 blockD1 blockD2` 带文字标签
- 5 nm 偏差窗口 + 最大偏差处的竖向标注箭头
- 右上角数值图例框（角度、R²、各偏差、κmax）

`--overlay-on <dir>`：按文件名 stem 匹配原始 TEM 灰度图，画在灰度图上（可选）。

---

## 4. CLI

```powershell
# 单图
.\.venv-napari\Scripts\python tools\tem_analysis\analyze.py `
    --mask "G:\datasets\TEM\1300kgroup1\temp_out\pseudo_color_prediction\751845_RMF 24(12,15)_1300kx_FOV_76.37nm.png" `
    --out  G:\datasets\TEM\1300kgroup1\analysis

# 批量 + 汇总 CSV
.\.venv-napari\Scripts\python tools\tem_analysis\analyze.py `
    --mask-dir G:\datasets\TEM\1300kgroup1\temp_out\pseudo_color_prediction `
    --out      G:\datasets\TEM\1300kgroup1\analysis `
    --csv      summary.csv
```

主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mask` / `--mask-dir` | — | 单图或目录（二选一） |
| `--out` | — | 输出目录 |
| `--scale-nm` | `0.0755` | nm/px |
| `--no-rotate` | off | 不做 Leveling 校平 |
| `--endpoint-extend` | `none` | `none` \| `region` |
| `--window-nm` | `5.0` | 需求 3 的窗口长度 |
| `--min-area` | `30` | 连通域最小面积(px) |
| `--keep-all-components` | off | 关闭"只留最大连通域" |
| `--milling-smooth` | 自动 | 样条平滑量 |
| `--milling-middle-frac` | `0.6` | 曲率搜索的中段占比 |
| `--milling-edge-margin-nm` | `5.0` | 样条端点排除边距 |
| `--blockd-trim` | on | Block_D 拟合的 2σ 剔除 |
| `--csv` | — | 批量汇总 CSV 文件名 |
| `--no-overlay` | off | 不生成叠加图 |
| `--overlay-on` | — | 原始图目录，叠加画在灰度图上 |
| `--classes` | 内置 12 类 | 逗号分隔覆盖类别顺序 |

---

## 5. 依赖

`tools/tem_analysis/requirements.txt`

```
numpy>=1.24,<3
Pillow>=10,<12
scipy>=1.10
opencv-contrib-python>=4.8      # ximgproc.thinning；缺失时走 NumPy 回退
matplotlib>=3.7                 # 叠加图
```

可直接复用已有的 `.venv-napari`（Python 3.11）。

---

## 6. 验证方案

- [ ] `selftest.py` 合成几何回归（不依赖真实数据）
  - 倾斜 3.0° 的直带 → Leveling 角度误差 < 0.05°，R² > 0.999
  - 半径 200 px 圆弧带 → `κ_max` 误差 < 2%
  - 两条已知 Δx/Δy 偏移的带 → 端点偏差精确复现（容差 1 px，含骨架内缩量断言）
  - Guo-Hall：OpenCV 实现与 NumPy 回退输出逐像素一致
  - 旋转往返：`to_orig(rotate(p))` 误差 < 1 px
- [ ] 在样本图上跑通，肉眼核对叠加图上 8 个标注点位置合理
- [ ] JSON 数值合理性：Leveling 角接近 0；MgO_C R² 接近 1；
      milling 曲率半径量级在几十 nm
- [ ] `--no-rotate` 复跑，各角度应恰好相差一个 Leveling 角
- [ ] 全新 venv 装 `requirements.txt`（不装 contrib）确认回退分支可用
- [ ] 批量跑整个 `pseudo_color_prediction/` 目录，确认无异常中断、CSV 列齐整

## 7. 已知风险

1. **标尺** 0.0746 vs 0.0755（见 0.2），线性影响全部 nm 结果
2. **骨架端点内缩**：Guo-Hall 固有特性，`a1`/`b1` 内偏约半个带宽；
   `--endpoint-extend region` 可缓解，但数值会与"逐列中线"口径不同
3. **`cv2.ximgproc`** 需 contrib 包；NumPy 回退较慢但结果一致
4. **"中间曲线"边界**无客观定义，靠 `middle-frac` / `edge-margin` 两个参数约定，
   κ(s) 全量落盘以便复核
5. **分割噪声**（碎片、孤立斑点）会挪动端点；`--min-area` + 最大连通域可控，
   且任何丢弃都会写 warning

---

## 8. 实现中的修正

计划里有四处设想被实测推翻，记录原因以免以后又改回去。

### 8.1 Milling 边缘分侧：骨架法 → 邻接法，且取朝真空的外侧

原方案（§2.8 步骤 2）用骨架法向的正负分侧。在样本图上直接测量每段骨架两侧的类别：

```
i= 411 p=(420,682)  +n -> _background_   -n -> MgO_L
i= 651 p=(431,443)  +n -> Block_U        -n -> _background_
```

器件在低处位于法向负侧、在高处位于法向正侧，**没有哪个符号能沿整条带选中同一侧**，
第一版叠加图确实中途翻面了。改为邻接判据：只看边界像素旁边是什么类，与法向无关。
`--milling-edge-method skeleton` 保留原方法供对比。

取哪一侧则按需求确认后定为 **`outer`**（朝真空、离子研磨真正加工出的表面）：保留"不与
任何其它类相邻"的边界像素。`inner`（贴器件侧）保留为选项。实测 `outer` 侧形状正是
「线段-曲线-线段」：左尾 −4.7°(rmse 0.053 nm)、右尾 −55.1°(rmse 0.607 nm)，拐点在弧长
85% 处，密切圆 rms 0.80 px。

### 8.1b 画幅切边必须排除

Milling 带跑出视场左右边界，`outer` 侧于是包含一段裁剪产生的直边和两个 90° 拐角——那
是整条边上曲率最大的位置。切换到 `outer` 后 `m2` 立刻落到了图像右边界上。现在丢弃距画
幅边界或旋转填充 `--milling-frame-margin-px`（默认 4）以内的边界像素。

这也解释了为什么旋转填充必须用 `255`(ignore) 而不是背景：校平后那条切边变成图像内部的
一条斜线，唯一还能表明它是"裁剪"而非真实表面的信息，就是它外侧属于原画幅之外。自检里
特意保留了这一点（`write_scene` 不把填充压成背景），否则该用例的曲率最大值就会落在切边
拐角上——最初写成压成背景时确实如此。

### 8.1c 需求 6 的量测对象改成 Non_mag

演进了两步：先是把 `Block_D` 的下边缘（每列最大 y）改成"不与 `Non_mag` 相邻的那条边"，
随后需求进一步明确为**量 `Non_mag` 本身**——取它**不与 `Block_D` 相邻**的那条横向边缘，
端点改名 `Non_mag1`/`Non_mag2`。`Block_D` 不再单独量测，只作为选边的参照类。

实现上不写死 "bottom"，而是分别统计上下两条边与 `Block_D` 的接触列数并取接触少的那条
（样本图 `top: 172 / bottom: 0` → 取下边缘），这样样品倒装或类别重标时不会悄悄量错界面。
`--nonmag-edge top|bottom` 可强制。原 `--blockd-*` 参数随之改名为 `--nonmag-*`。

### 8.1d 新增需求 7：corner_offsets

`Non_mag1`→`m1`、`Non_mag2`→`m2` 的 dx/dy。这是唯一一个需要组合其它量测结果的模块，为此
给 `AnalysisContext` 加了 `results` 字典（`analyze.py` 按注册表顺序边算边写入）以及
`offset()` / `measured_point()` 两个共享方法（`interfaces.py` 里原本私有的 `_offset` 也
收敛到 `ctx.offset`）。注册表顺序保证它排在 `non_mag` 与 `milling_*` 之后。

自检只能验证"报出的 dx/dy 精确等于它所命名的两点之差"：在恒定曲率圆弧上，曲率最大值的
位置由像素噪声而非几何决定，绝对值无法预言。

### 8.2 曲率搜索的「中段」限制默认关闭

原方案默认 `--milling-middle-frac 0.6`。实测拐点在弧长的 **73%**（44.97 nm 边缘上
s=33.01 nm），中段 60% 的窗口会把真正的拐弯排除掉。直段的 κ≈0 本来就抢不到最大值，
所以默认改成 `1.0`（不限制），只保留两端 2 nm 的样条端点余量。

### 8.3 曲率幅值：样条解析导数不够准，增加密切圆拟合

splprep 会把节点数削减到残差与 `s` 匹配，少数几段三次曲线撑不住恒定曲率。在真值
200 px 的合成弧上：

| 估计方式 | 半径 | 偏差 |
|---------|------|------|
| 样条解析 κ（`sigma` 0.5→6，已饱和） | 168.4 px | −15.8% |
| 密切圆拟合（原始边缘点，半窗 0.8R） | 190.0 px | **−5.0%** |

另外两条弯路也记下来：**(a)** 全局扫描「局部半径最小处」不可行——Kása 代数圆拟合在近
共线点上条件数极差，会在直段返回 R≈10 px、rms 5–17 px 的垃圾解并赢得 argmin。改为
只在样条已定位的拐点处拟合一次，并把点先中心化（`geometry.fit_circle`）。
**(b)** 在样条采样点上拟合圆没用（仍带样条本身 ±15% 的形状误差），必须用原始边缘点。

结论：`radius`/`kappa_*` 保留样条值（需求指定的方法，定位可靠），另出
`local_circle.radius` 作为幅值口径，两者都进 JSON / CSV / 叠加图。

### 8.4 R² 不能作为质量门限

`R² = 1 − SS_res/SS_tot`，水平层的 `SS_tot` 极小，所以 1029 px 长、rmse 1.94 px 的
`Leveling` 只得 R² = 0.021，`Block_D`（rmse 0.42 px）只得 0.078。第一版据此报警，结果
26 张图几乎全部触发——纯噪声。改为：

- 保留 `fit_r2`（需求要求），附带 `fit_r2_degenerate` 标记与 `fit_r2_note` 说明
- 平直度看 `rmse` / `max_residual`
- 校平角度的可靠性用斜率标准误 `leveling.angle_stderr_deg`
  （首版误用 `arctan(3.46·rmse/length)`，漏掉了 √N 平均，给出 0.374° —— 实测校平后
  残余角只有 0.027°，正确的标准误是 0.012°）
- `local_circle` 的 rms 门限由 1.0 px 放宽到 3.5 px：真实拐弯不是圆弧，1–2 px 属正常，
  26 张图实测最大 2.4 px，而条件数失效的垃圾解在 5–17 px

### 8.1e 「最大连通域」不是安全的选法

26 张图批量结果里 `non_mag.length` 出现一个 33.98 nm 的离群值（中位数 10.4 nm），
`corner_offsets` 也跟着到 −37 nm。查到那张图的预测在画幅底部长出一条 `Non_mag` 带，面积
11128 px，**比真正的 `Non_mag` 方块还大**，于是 §2.1 的"只留最大连通域"选中了伪迹。

改为：`class_mask` 接受 `prefer_near`，先只在与该类相邻的连通域里挑，挑不到再退回按面积；
`Non_mag` 锚定在 `Block_D` 上（那正是它选边时已经依赖的邻居，不需要额外信息）。修复后该图
9.14 nm，26 张图的 dx/dy 符号也全部一致（左侧 dx 恒负、右侧恒正、dy 恒正）。

教训：这种错误不会抛异常、也不会让叠加图明显出错，只能靠**看整批结果的分布**发现。合成
自检里加了对应回归用例（塞一条大 6 倍且远离 `Block_D` 的假带）。

### 8.6 后续需求变更（第三轮）

1. **`b3`/`b4`**：除了 `MgO_C` 骨架的原始端点 `b1`/`b2`，再给出**拟合线段**的两个端点
   （极值点投影到拟合直线），并加 `a1→b3`、`a2→b4` 两组偏差。为此把 `mgo_c` 在注册表里
   提到 `interfaces` 之前，由 `interfaces` 从 `ctx.results` 读回 `b3`/`b4`，这样全部
   a↔b 偏差仍集中在一个模块里。实测 dx 两者几乎一致（中位 5.332 / 5.333 nm），
   dy 的离散度 `b3` 更小——符合"拟合线不受局部起伏影响"的预期。

2. **标尺改为从文件名解析**：`--scale-nm` 默认 `auto`，正则取 `..._FOV_76.37nm.png` 里的
   视场，再除以**图像实际宽度**（而非写死 1024，以免掩膜换尺寸导出后算错；宽度非 1024 时
   warning）。26 张图全部解析成功 → `76.37 / 1024 = 0.074580 nm/px`。解析失败且未显式给
   `--scale-nm` 时直接报错，不静默用错标尺。§0.2 记录的 0.0755 与 0.0746 之争就此关闭。

3. **`Non_mag1`/`Non_mag2` 改为拟合直线外延到区域 x 极值处的交点**。原先停在参与拟合的
   列上，而被 σ 剔除的恰恰是左右两端圆角掉的列，端点会随剔除阈值内缩。改后端点是区域的
   属性而不是阈值的属性；y 仍取直线值，所以圆角不污染纵坐标。样本图上外延了 11 px，
   `non_mag.length` 的分布从 7.56–14.19 nm 变为 9.30–14.69 nm。

### 8.5 其它

- `MgO_C` 的拟合线段改用与 `interfaces` 相同的（可外推）骨架，否则 `fit_length`
  与 `b1..b2` 的跨度对不上（160 px vs 170 px）
- `extend_to_region` 的切线窗口放宽到 `half=14`：端点是骨架最毛糙的地方，短窗口的
  切线误差会被外推长度放大
- `guo_hall()` 先给掩膜补 2 px 背景边再细化：OpenCV 不处理最外一圈像素，而这些掩膜里
  有好几层是跑出画幅左右边界的，不补边会在边界留下整条未细化的残桩
- 轮廓追踪（Jacob 终止条件）会把起点收进列表两次，需去掉重复点
- 叠加图补画：5 nm 搜索窗口（加粗骨架段 + 两端竖线 + 端点基准虚线 + 偏差箭头与数值）、
  密切圆、`Block_D` 选中的整条边缘像素，便于目视核对而不是只看数字
- 合成自检场景必须与真实拓扑同构，否则测的是脚手架而不是工具。踩过两次：
  **(a)** `Non_mag` 原本放在 `Block_D` 上方且不相邻，自动选边规则无从验证；
  **(b)** 圆环扇形的径向端帽面向背景，于是被算进 `outer` 侧，其 90° 拐角赢得曲率最大值
  （半径报 46 px 而真值 215 px）。改为让 `MgO_L/R` 在角度上包住端帽即可
- 26 张图批量实测：全部成功，无异常中断，warning 只剩预测质量相关的 7 条
