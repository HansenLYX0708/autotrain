# TEM 分割图几何分析

对 PaddleSeg 输出的伪彩色分割图（`pseudo_color_prediction/`）做几何量测：用
`Leveling` 层校平，然后逐项量测层间偏差、层平直度、Milling 边缘曲率和 Block_D 下边缘。

实现计划与算法取舍见 <PLAN.md>。

## 安装

```powershell
cd D:\_work\projects\autoTraining\Autotrain_g
.\.venv-napari\Scripts\pip install -r tools\tem_analysis\requirements.txt
```

## 用法

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

# 批量 + 汇总 CSV
.\.venv-napari\Scripts\python tools\tem_analysis\analyze.py `
    --mask-dir G:\datasets\TEM\1300kgroup1\temp_out\pseudo_color_prediction `
    --out      G:\datasets\TEM\1300kgroup1\analysis `
    --overlay-on G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm`
    --csv      summary.csv

# 批量 + 汇总 CSV + 叠加
.\.venv-napari\Scripts\python tools\tem_analysis\analyze.py `
    --mask-dir G:\datasets\TEM\1300kgroup2\normal_test_predict\pseudo_color_prediction `
    --out      G:\datasets\TEM\1300kgroup2\normal_test_analysis `
    --overlay-on G:\datasets\TEM\1300kgroup2\normal_test`
    --csv      summary.csv

# 不校平；叠加图画在原始 TEM 灰度图上
.\.venv-napari\Scripts\python tools\tem_analysis\analyze.py `
    --mask-dir ... --out ... --no-rotate `
    --overlay-on G:\datasets\TEM\1300kgroup1\1300kx_FOV_76.37nm

# 自检（合成几何，已知真值）
.\.venv-napari\Scripts\python tools\tem_analysis\selftest.py
```

必须用 `pseudo_color_prediction/`。`added_prediction/` 是原图与掩膜的混合预览，不是
标签图，工具会拒绝（离色像素超过 5% 即报错）。

## 输出

每张图三个产物（叠加图可用 `--no-overlay` 关闭）：

| 文件 | 内容 |
|------|------|
| `<stem>.json` | 全部指标。每个点同时给旋转后与原图坐标，每个距离同时给 px 与 nm；另含曲率曲线、残差曲线等序列，便于不重跑就改参数复核 |
| `<stem>_overlay.png` | 骨架、拟合线、密切圆、`a1 a2 b1 b2 m1 m2 blockD1 blockD2` 标注点与数值图例 |
| `summary.csv` | 批量模式下一行一图，列顺序固定（`report.CSV_COLUMNS`） |

单项量测失败只会让该项变成 `null` 并写入 `warnings`，不会中断整批。

## 量测项

| JSON 键 | 需求 | 说明 |
|---------|------|------|
| `leveling` | 1 | `Leveling` 骨架拟合直线 → 倾角，定义整图校平 |
| `interfaces` | 2, 3 | `a1/a2/b1/b2`、`a1→b1` 与 `a2→b2` 的 dx/dy、5 nm 内最大向下偏差 |
| `mgo_c` | 4 | `MgO_C` 拟合线段、fit R²、角度 |
| `milling_l` / `milling_r` | 5 | 被离子研磨出来的外侧边缘、样条解析曲率、`m1`/`m2`、密切圆半径 |
| `non_mag` | 6 | `Non_mag` 背离 `Block_D` 那侧的横向边缘拟合直线、`Non_mag1`/`Non_mag2` |
| `corner_offsets` | 7 | `Non_mag1`↔`m1` 与 `Non_mag2`↔`m2` 的 dx/dy |

端点定义：`a1` = `SAF_Ru_L` 骨架 x 最大端，`a2` = `SAF_Ru_R` x 最小端，
`b1`/`b2` = `MgO_C` x 最小/最大端。偏差符号：`dx>0` 表示 b 在 a 右侧，
`dy>0` 表示 b 在图像中比 a 更低。5 nm 窗口沿**弧长**从端点回走，偏差以端点自身为基准
（`dev = y - y(endpoint)`，正值 = 更低）。叠加图上该窗口用洋红色画出：加粗的骨架段是
被搜索的范围，两端竖线是窗口边界，虚线是端点基准线，箭头指向最大偏差点并标出数值。

**Milling 取哪一侧**（`--milling-edge-method`）：一条 Milling 带有两个侧面，一侧贴器件
（`MgO_L`/`MgO_R`/`Block_*`），一侧朝真空。默认 `outer` = 朝真空那侧，也就是离子研磨
真正加工出来的表面：保留"不与任何其它类相邻"的边界像素，再取轮廓顺序上最长连续段。
`inner` 取贴器件那一侧。`skeleton` 是最初设想的"骨架法向分侧"，仅供对比，原因见下。

**Non_mag 取哪条边**（`--nonmag-edge`）：默认 `auto`，取**不与 `Block_D` 相邻**的那条
横向边缘——分别统计上下两条边与 `Block_D` 接触的列数，取接触少的那条。样本图上
`top: 172 / bottom: 0`，于是取下边缘。这样写而不是写死 "bottom"，是为了样品倒装或类别
重标时不会悄悄量错界面。`edge_side` 与接触计数都写进 JSON/CSV。

**`corner_offsets`** 给出 `Non_mag1`→`m1` 和 `Non_mag2`→`m2` 的 dx/dy（px 与 nm）和直线
距离，符号约定与 `interfaces` 一致。它是唯一一个组合其它量测结果的模块，因此在注册表里
排在 `non_mag` / `milling_*` 之后，从 `ctx.results` 读回它们而不重算；任一侧依赖的量测失
败时该侧单独变成 `null`。叠加图上用青绿色折线画出，横竖两段分别对应 dx 和 dy。

角度一律给两个：`angle_deg_image`（图像坐标，y 向下，顺时针为正）和
`angle_deg_math`（数学坐标，逆时针为正）。

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--scale-nm` | `0.0755` | nm/px，见下方"标尺"一节 |
| `--no-rotate` | off | 不做 `Leveling` 校平 |
| `--endpoint-extend` | `none` | `region` 把 `a1/a2/b1/b2` 外推回区域尖端，补偿骨架内缩 |
| `--window-nm` | `5.0` | 需求 3 的弧长窗口 |
| `--min-area` | `30` | 连通域最小面积(px) |
| `--keep-all-components` | off | 关闭"每类只留最大连通域" |
| `--thinning-backend` | `auto` | `cv2` / `numpy`，两者输出逐像素一致 |
| `--milling-edge-method` | `outer` | `outer`(朝真空/研磨面) / `inner`(贴器件) / `skeleton` |
| `--milling-frame-margin-px` | `4` | 丢弃距画幅边界或旋转填充这么近的边界像素 |
| `--milling-smooth` | `1.0` | 假定边缘噪声(px)，决定样条平滑量 |
| `--milling-middle-frac` | `1.0` | 曲率搜索的中段占比，默认不限制 |
| `--milling-edge-margin-nm` | `2.0` | 边缘两端排除的弧长 |
| `--milling-circle-window-frac` | `0.8` | 密切圆拟合半窗 = 该系数 × 样条半径 |
| `--milling-tail-frac` | `0.25` | 两端各取多少弧长拟合直线（线-曲线-线校验） |
| `--nonmag-edge` | `auto` | `auto` 取不与 `Block_D` 相邻的那条；也可强制 `top`/`bottom` |
| `--nonmag-sigma` / `--no-nonmag-trim` | `2.0` / on | `Non_mag` 拟合的 σ 剔除 |
| `--classes` | 内置 12 类 | 逗号分隔覆盖类别顺序 |

类别顺序默认用内置的 12 类（`labelmap.DEFAULT_CLASSES`）；若掩膜目录或其上两级存在
`class_names.txt` 则优先读取。调色板逻辑直接复用
<../napari_seg/annotate.py>，二者不会走偏。

## 需要知道的几个坑

**标尺对不上。** `76.37 nm / 1024 px = 0.0746 nm/px`，而需求给的是 `0.0755`
（对应有效宽度约 1011.5 px）。`--scale-nm` 默认按需求取 `0.0755`，但它线性影响所有
nm 数值（差 1.2%），建议核对标定。JSON 里同时记了 `scale_nm_per_px` 与
`image_size_px`，事后可换算。

**R² 对近水平的层没有意义。** `R² = 1 − SS_res/SS_tot`，而水平层的 `SS_tot`（y 的方
差）本身极小，所以一条 1029 px 长、rmse 仅 1.9 px 的优秀直线也只能得到 R² ≈ 0.02。
`Leveling` / `MgO_C` / `Block_D` 都是这种情况。结果里带 `fit_r2_degenerate` 标记和
`fit_r2_note` 说明，**判断平直度请看 `rmse` 与 `max_residual`**。校平角度的可靠性用
`leveling.angle_stderr_deg`（斜率标准误）衡量，不用 R²。

**骨架端点会内缩。** Guo-Hall（任何形态学细化都一样）会从区域尖端回缩约半个带宽，所以
`a1`/`b1` 默认偏内侧。合成算例里 13 px 厚的层内缩正好 6 px。用
`--endpoint-extend region` 沿端点切线外推到区域尖端可以补偿（自检里该模式下 `a1`
误差为 0）；默认关闭以保持"忠实于 Guo-Hall"。dx/dy 是两个端点之差，两侧内缩量相近时
会大部分抵消。

**Milling 边缘不能用骨架法分侧。** 需求原本设想"用骨架分割边缘、保留正向距离一侧"。在
样本图上逐点实测两侧的相邻类别：

```
i= 411 p=(420,682)  +n -> _background_   -n -> MgO_L
i= 651 p=(431,443)  +n -> Block_U        -n -> _background_
```

器件在低处位于法向负侧、在高处位于法向正侧，**没有哪个符号能沿整条带选中同一侧**，
结果会中途翻面。改用邻接判据（`outer`/`inner`），只看边界像素旁边是什么类，与骨架法向
无关。`--milling-edge-method skeleton` 保留原方法仅供对比。

**画幅切边会伪造出最尖的拐角。** Milling 带跑出视场左右边界，外侧边缘于是包含一段由裁
剪产生的直边和两个 90° 拐角——那是整条边上曲率最大的地方，加上这条判据之前 `m2` 就落
在图像右边界上。现在会丢弃距画幅边界或旋转填充 `--milling-frame-margin-px`(默认 4) 以
内的边界像素。旋转填充用 `255`(ignore) 而非背景正是为此：校平后那条切边变成图像内部的
一条斜线，唯一还能表明它是"裁剪"而非真实表面的信息，就是它外侧属于原画幅之外。

**曲率幅值是尺度相关的，样条值偏小。** 1 px 量化的边缘上，曲率必然依赖测量尺度。
splprep 会把节点数削减到残差与 `s` 匹配，而少数几段三次曲线撑不住恒定曲率：在真值
半径 200 px 的合成弧上，样条报 168 px（−16%），而密切圆拟合报 190 px（−5%）。因此：

- `radius` / `kappa_*` 来自样条解析导数（需求指定的方法），**定位** `m1`/`m2` 可靠
- `local_circle.radius` 是在同一点对原始边缘点拟合的圆，**幅值**请用它
- `local_circle.rms` 是拟合残差；真实拐弯不是圆弧，1–2 px 属正常，超过 3.5 px 才告警

"中间曲线"没有客观定义。实测拐点在弧长的约 73%（不是中点），所以
`--milling-middle-frac` 默认 `1.0`（不限制中段）——直段的 κ≈0 本来就抢不到最大值。
完整 κ(s) 序列写进 JSON，改参数不必重跑。

**"最大连通域"不等于"对的那个"。** 26 张图里有一张的预测在画幅底部长出一条 `Non_mag`
带，面积（11128 px）比真正的 `Non_mag` 方块还大，于是按面积选就量到了那条伪迹上——边缘
长度报 33.98 nm 而正常值约 10 nm。现在 `Non_mag` 的连通域锚定在 `Block_D` 上：先只在
"与 `Block_D` 相邻"的连通域里挑，挑不到再退回按面积。修复后该图为 9.14 nm，回到正常区
间。其余类别仍按面积选；任何丢弃都会写 warning，并注明是否用了锚定。

**旋转会引入重采样噪声。** 校平用最近邻插值（类别 id 不能做插值平均），填充值为
`255`（ignore），以便区分"画幅外"和"预测为背景"。代价是层骨架会毛糙 1–2 px：自检中
校平后 `Leveling` 残余角 0.007°，但 `dy` 误差可达约 5 px。`--no-rotate` 可完全避开。

## 自检覆盖

`selftest.py` 全部为合成几何 + 已知真值，无需真实数据：

- Guo-Hall：OpenCV 与 NumPy 两条实现对直带/折弯带/随机团逐像素一致
- 旋转正反变换往返误差 < 1e-9 px；按拟合角旋转后方向恰好水平
- 精确直线的 TLS 角度误差 < 1e-6°；R² 退化标记按预期触发
- 精确圆的解析曲率半径误差 < 1%（50 / 200 / 600 px）
- 轮廓追踪：矩形环长度精确，环绕连续段选取正确
- 未倾斜整图端到端：`a1/a2/b1/b2`、dx/dy、`blockD1/2`、角度、rmse **误差为 0**；
  骨架内缩量落在预期区间（13 px 厚的层正好内缩 6 px）
- `Non_mag` 自动选边：合成图与真实图同构（`Block_D` 紧贴 `Non_mag` 上方），必须选中下边缘
- `corner_offsets`：报出的 dx/dy 必须精确等于它所命名的两个点之差（1e-9 px），nm 换算同样精确
- 干扰连通域：往合成图里塞一条比真 `Non_mag` 大 6 倍（57000 vs 9150 px）且远离 `Block_D`
  的假带，量测必须仍落在真方块上（复现真实批次里的那张坏预测）
- Milling 分侧：合成图内外两侧半径已知（200 / 215 px），`outer` 与 `inner` 各须落在对应
  半径上；选中的边缘不得包含画幅边界像素
- 倾斜 3° 再校平：恢复倾角误差 < 0.05°，残余角 < 0.05°，密切圆半径误差 < 10%
  （其中左侧圆弧被画幅切断，专门用来验证切边排除；右侧完整，用来验证端帽排除）
- NumPy 细化后端与 cv2 后端给出完全相同的角度
