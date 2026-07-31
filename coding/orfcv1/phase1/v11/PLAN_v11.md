# 阶段一计划 v11：从 ORFC 热启动的非均匀联合学习

日期：2026-07-31  
状态：预注册草案；正式执行前冻结为 Git commit/tag。  
前置节点：N10（分配算法桥接通过）、N12（v10 联合训练设计失败，holdout 未开封）。

---

## 0. 本轮唯一目标

本轮回答两个递进问题：

1. 在固定名义总码率下，联合优化正交表示、码本和组级非均匀分配，能否严格优于原始 ORFC？
2. 若优于 ORFC，收益中是否存在由非均匀分配带来的独立贡献？

完整 hard tail 失真为

\[
D(U,\Theta,\mathbf m)
=
\mathbb E_x
\left\|
h\!\left(\widehat F_x(U,\Theta,\mathbf m)\right)-h(F_x)
\right\|_2^2 ,
\]

其中 \(U\) 为正交表示，\(\Theta\) 为三模式分组码本，
\(\mathbf m\in\mathcal A_R\) 为固定总名义码率 \(R\) 下的离散分配。
目标问题是

\[
\min_{U,\Theta,\mathbf m\in\mathcal A_R}
D(U,\Theta,\mathbf m).
\]

v11 采用交替优化：

\[
(U,\Theta)\ \xleftarrow{\text{inner hard-STE}}\ 
D(U,\Theta,\mathbf m_t),
\qquad
\mathbf m_{t+1}\ \xleftarrow{\text{outer hard search}}\ 
\arg\min_{\mathbf m\in\mathcal N(\mathbf m_t)}
D(U,\Theta,\mathbf m).
\]

内层更新 \(U,\Theta\)，外层在完整 hard 失真上更新非均匀分配。
因此本轮检验的确是 \(U\)、码本和分配的联合优化；外层搜索不是训练后的附加分析。

本轮不研究真实熵率、ECVQ 主结论、token 条件熵、分类或分割。
这些内容只在 v11 的 hard tail 主结论成立后进入下一阶段。

---

## 1. v10 失败与 v11 修复边界

v10 有两条相互独立的设计缺陷：

1. 均匀分配训练只更新 mode-1；mode-0/mode-2 在 \(U\) 已变化后仍保持初始值，
   外层搜索比较的是与当前表示不匹配的陈旧码本。
2. 200 步短探针选择的最大步长被外推到 3,096 步，训练约在 step 350 达到最低点后
   持续恶化，终点不是有效解。

v11 分别用“全菜单覆盖训练”和“全长稳定性选参”修复。
N12 的六个终点均不进入 v11，也不用于 holdout 判断。

`holdout-500` 尚未被 v10/N12 读取，继续作为 v11 唯一确认集。
历史 `measure-3000` 已在早期实验及 v9 中使用，只保留为历史结果，不再承担确认性声称。

---

## 2. 固定问题设定

共同设置：

```text
model       DINOv2 ViT-L/14
split       blk20，tail = blocks[21:]
G × d       32 × 32
norm        per_image
TF32        off
dtype       float32
R64         (K2,K4,K8)，均匀 K4，64 bit/token
R96         (K4,K8,K16)，均匀 K8，96 bit/token
```

所有候选分配必须逐项满足

\[
\sum_{g=1}^{32}\log_2 K_{m_g}=R
\]

且由整数运算核验误差为零。

### 2.1 数据职责

| 数据 | 数量 | 唯一职责 |
|---|---:|---|
| train-fit | 3,500 | \(U,\Theta\) 的梯度训练、侧模式校准 |
| train-val | 500 | 全长超参数选择、稳定性与菜单对齐闸门 |
| cal | 500 | A2 外层 hard 分配搜索 |
| dev | 500 | A0 选择、Stage B 与开封前效力闸门 |
| holdout | 500 | 冻结后一次性确认 |

`train-fit/train-val` 由原 train-core 4,000 张按 `TRAINVAL_SEED` 一次冻结。
五个集合必须 basename 两两零交集。
holdout 为 500 类各 1 张，从未进入 v8/v9/v10 的新 ImageNet-val 图像。

任何训练、选参、搜索和 checkpoint 选择代码读取 holdout，均记为
`INVALID_EXPERIMENT`。

---

## 3. 三个实验对象

### A0：最强 hard-NN ORFC

对 R64、R96 分别枚举远端全部合法同码率 ORFC 检查点：

```text
ORFC/coding/orfc/checkpoints/dinov2_vitl14/
```

每个检查点必须包含：

```text
R          (1024,1024)
codebooks  (32,K,32)
pmf        (32,K)
layer      blk20
norm       per_image
```

在 dev-500 上使用 hard nearest-neighbour、固定名义码率和完整 tail 失真评估。
每个码率选择 dev hard 失真最低者为 A0；候选集合、排序规则和 tie-break 在评估前冻结。

不同码率可以选择不同 \(\lambda\)、epoch 或 seed。对称性来自相同的
“每个码率选择最强合法 ORFC”规则。

ECVQ 不进入 A0。各检查点自己的 ECVQ 结果与实际 rANS 码率单独记录，
留给真实熵率阶段。统一的 `s42/λ=0.5/ep100` 只作匹配配置描述项。

### A1：uniform-joint

- 从 A0 的 \(U\) 和主档码本热启动；
- 固定均匀分配；
- 联合训练 \(U\) 与三模式码本；
- 不执行外层分配更新。

### A2：nonuniform-joint

- 与 A1 逐元素相同地热启动；
- 使用相同 inner 步数、batch 流、学习率表和菜单辅助流；
- 每 \(T_{\rm outer}\) 步在 cal 上执行一次完整 hard 一比特邻域搜索；
- 接受的分配立即成为后续 inner 主分配。

A1/A2 的唯一本质差别是 A2 允许外层改变 \(\mathbf m_t\)。
A2 的搜索 FLOPs 与 cal 使用量作为方法成本报告，不用额外训练步数补偿。
v10 的等算力 A3 退出主流程。

---

## 4. ORFC 热启动与三模式菜单

每个锚点分别构建：

\[
U_0:=R_{A0},\qquad
\Theta_{1,0}:=\Theta_{A0}.
\]

mode-0/mode-2 使用与 v9 相同的 k-means++/Lloyd 实现，
在 \(U_0\) 旋转后的 train-fit 特征上独立拟合。

### W-I：初始化同一性

任何一条失败均为 `INVALID_EXPERIMENT`：

1. `array_equal(U_init, R_A0)`；
2. `array_equal(codebook_1, codebooks_A0)`；
3. A0、A1@0、A2@0 的 hard 量化索引在 dev-500 和固定 100 张 train-fit 上逐元素相同；
4. A0、A1@0、A2@0 的逐图 tail 失真相对差不超过
   `REPLAY_REL_TOL = 8·2^-23`；
5. \(\|U_0^\top U_0-I\|_F\le10^{-4}\)；
6. `layer=20`、`tail=blocks[21:]`、`norm_mode=per_image` 一致。

参数张量和索引要求逐元素相同；tail 归约只要求落在已知 fp32 replay 容差内。

---

## 5. Stage B：在 ORFC 工作点重放非均匀收益

固定新建的 \((U_0,\Theta_0)\)，从均匀分配开始：

1. 枚举所有合法 hard 一比特交换；
2. 在 cal-500 上选择完整 tail 失真最低者；
3. 改善超过预注册数值噪声带才接受；
4. 重复至无改善或达到 `S_OUTER_MAX`；
5. 在 dev-500 上一次性比较最终非均匀分配与均匀分配。

统计使用逐图配对差。Stage B 只决定锚点是否进入后续联合训练，不产生最终论文结论。

### Stage B exit

| 结果 | 动作 |
|---|---|
| R64、R96 均有 dev 改善 | 两锚点进入 Stage C |
| 仅一个锚点改善 | 只保留该锚点，另一锚点结束 |
| 两点均无改善 | v11 结束，holdout 保持封存 |
| 任一同一性/码率/重放不变量失败 | `INVALID_EXPERIMENT`，只修实现，不作科学结论 |

Stage B 未通过只说明当前 ORFC 工作点及当前三档菜单的一比特邻域没有可确认收益，
不外推为所有非均匀分配或所有联合训练均无收益。

---

## 6. Inner：hard 主目标与全菜单覆盖

### 6.1 hard-STE 主前向

\[
z=yU,\qquad
\hat c_g=
\Theta_{g,m_g}\!\left[
\arg\min_k\|z_g-\Theta_{g,m_g,k}\|^2
\right],
\]

\[
\tilde z_g=\hat c_g+z_g-\operatorname{sg}(z_g),
\qquad
\widehat F=\operatorname{invnorm}(\tilde zU^\top).
\]

前向与终点 hard nearest-neighbour 完全一致；STE 只定义反向。
主损失为当前分配的完整 tail 失真

\[
\mathcal L_{\rm main}
=D(U,\Theta,\mathbf m_t).
\]

### 6.2 菜单覆盖辅助项

辅助项的职责是使外层可能调用的全部组×模式码本持续跟随当前 \(U\)。
它不被解释为候选分配内层最优值的精确计算。

固定一个由 `TRAIN_SEED` 决定的组置换。每三个 inner step 循环三个
同码率辅助分配：

1. \(\mathbf a^{(1)}=(1,\ldots,1)\)：全部组使用 mode-1；
2. \(\mathbf a^{(02)}\)：前 16 组 mode-0，后 16 组 mode-2；
3. \(\overline{\mathbf a}^{(02)}\)：前后两半交换 mode-0/mode-2。

每个辅助分配的模式索引和均为 32，因此 R64/R96 均严格保持名义码率。
每个三步周期内，每个组的 mode-0/1/2 恰好各被辅助前向覆盖一次。
每个周期重新使用冻结种子生成置换，流完全落盘且 A1/A2 相同。

辅助损失为

\[
\mathcal L_{\rm menu}
=
D\!\left(\operatorname{sg}(U),\Theta,\mathbf a_s\right),
\]

\[
\boxed{
\mathcal L
=
\mathcal L_{\rm main}
+\beta\mathcal L_{\rm menu},
\qquad \beta=1.
}
\]

辅助支路对 \(U\) 的梯度恒为零，只校准码本；主支路决定 \(U\) 的更新方向。
A1/A2 使用完全相同的辅助分配流和 dead-codeword 处理。

### 6.3 Outer：仅 A2

每 \(T_{\rm outer}\) 个 inner step：

1. 固定当前 \(U,\Theta\)；
2. 在 cal-500 上评估当前分配及全部合法一比特邻居的 hard 完整失真；
3. 取唯一最小邻居，按冻结字典序破平；
4. 只有其改善超过数值噪声带才接受；
5. 核验整数名义码率后进入下一 inner block。

这是有限邻域的交替下降，不声称找到完整 \(\mathcal A_R\) 的全局最优。

---

## 7. 全长选参与稳定性

### 7.1 数据隔离

- 训练：train-fit；
- 超参和稳定性：train-val；
- 外层搜索：cal；
- 开封前效力：dev；
- 最终确认：holdout。

### 7.2 学习率

\(U\) 与 \(\Theta\) 均使用预注册余弦退火：

\[
\eta(t)
=
\eta_0\frac{1+\cos(\pi t/N_{\rm inner})}{2},
\qquad t=0,\ldots,N_{\rm inner}.
\]

短探针不再直接决定正式学习率。

候选初始网格与 shortlist 规则在代码执行前写入 `config.py`：

1. 5×5、200 步只用于排除数值无效格并给出短程排序；
2. shortlist 包含短程最优格、至少一个 `lr_theta` 低一数量级的格，
   以及一个预注册的内部参考格；
3. 三个 shortlist 格全部按正式 \(N_{\rm inner}\)+余弦退火跑满；
4. 仅在通过稳定性闸门的格中，按 train-val 最后 25 步均值选择终点最低者；
5. 若边界格再次获胜，不在 v11 内扩网格，只记录。

若一个锚点没有任何全长稳定候选，该锚点退出，不进入正式两臂训练。

---

## 8. 开封前闸门

所有闸门均只使用 train-fit/train-val/cal/dev。

### G1：终点确实改善

A1/A2 的 train-val 最后 25 步 hard 失真均值，相对各自 step-0 必须下降超过
`100 × REPLAY_REL_TOL`。

### G2：完整训练稳定

令 \(\bar D_t\) 为 train-val 上 25 步滑动均值，要求

\[
\arg\min_t\bar D_t\ge0.9N_{\rm inner},
\qquad
\bar D_{N_{\rm inner}}
\le1.01\min_t\bar D_t.
\]

该闸门要求正式终点本身有效，不允许事后取中途最优 checkpoint 替代终点。

### G3：菜单覆盖与当前 \(U\) 对齐

G3 同时包含机械覆盖和效果核验：

1. 每个三步周期内，每个 \((g,\text{mode})\) 的辅助调用次数必须恰好为 1；
2. 外层每次搜索前，保存上一 inner block 起点的旧码本
   \(\Theta_{\rm stale}\)；
3. 固定当前 \(U_t\)，在 train-val 的固定菜单探针
   \(\{\mathbf a^{(1)},\mathbf a^{(02)},\overline{\mathbf a}^{(02)}\}\) 上比较

\[
\overline D_{\rm fresh}
=
\overline D(U_t,\Theta_t),
\qquad
\overline D_{\rm stale}
=
\overline D(U_t,\Theta_{\rm stale});
\]

4. 要求

\[
\overline D_{\rm fresh}
\le
\overline D_{\rm stale}
+10\cdot\mathrm{REPLAY\_REL\_TOL}\cdot\overline D_{\rm stale}.
\]

因此 G3 验证的是“更新后的菜单在当前 \(U\) 下不劣于陈旧菜单”。
梯度范数、参数漂移和 dead-codeword 次数继续记录，但不再单独作为对齐结论。

### G4：方法在 dev 上具备可测效力

每个锚点必须满足：

1. A2 至少接受一次真实 hard 交换，最终 \(\mathbf m_{A2}\ne\mathbf m_{\rm unif}\)；
2. dev 上 \(D(A2)-D(A0)\) 的配对单侧上置信界小于 0；
3. dev 上 \(D(A2)-D(A1)\) 的配对单侧上置信界小于 0。

未通过 G4 的锚点不进入 holdout 确认族。

若 G1–G3 通过但 A2 接受零次交换，只能表述为：

> 当前训练器和一比特搜索在该锚点未产生可测的非均匀方法臂。

不得据此声称 \(U\) 已吸收全部非均匀收益。

---

## 9. 节点、输入、产物与 exit

| 节点 | 输入 | 动作与产物 | Exit |
|---|---|---|---|
| N14 A0 选择 | 全部合法 ORFC 同码率检查点、dev | 冻结候选表、hard-NN 最强 A0；ECVQ/匹配配置描述表 | 无合法检查点或 provenance 不一致：`INVALID_EXPERIMENT` |
| N15 菜单初始化 | A0、train-fit | 构建三模式菜单，执行 W-I1…W-I6 | 任一 W-I 失败：只修实现；全部通过才继续 |
| N16 ORFC 工作点桥接 | 冻结菜单、cal/dev | hard 一比特搜索与 dev 验收，冻结 eligible anchors | 空集：v11 结束；单点/双点按实际继续 |
| N17 全长 profiling | eligible anchors、train-fit/train-val | 吞吐、shortlist、三格全长退火，冻结 `profile.json` | 无稳定 profile 的锚点退出；空集则结束 |
| N18 两臂联合训练 | A0、profile、train/cal/dev | A1/A2 两臂，完整轨迹、分配轨迹、G1–G4、冻结确认族 | G1–G3 失败：设计/实现失败；G4 失败：该锚点不进确认族；空集不开封 |
| N19 holdout 确认 | 冻结代码、checkpoint、分配、确认族 | 写封条后一次开封，运行正式统计 | 按第 10 节判定；开封后不得修改 v11 |

每个节点完成后写入
`节点记录/节点记录/N*.md`，包括输入 provenance、运行命令、结果、耗时和 exit。

---

## 10. Holdout 统计与最终判定

### 10.1 确认性假设

每个 eligible anchor 有两个必须同时成立的比较：

\[
H^{\rm ORFC}_{0,R}:
\mathbb E[D(A2)-D(A0)]\ge0,
\]

\[
H^{\rm alloc}_{0,R}:
\mathbb E[D(A2)-D(A1)]\ge0.
\]

确认 family 由开封前通过 G1–G4 的全部
`(anchor, comparison)` 组成，最多四个假设。
统一使用 Holm 控制 FWER=0.05。

逐图差使用预注册的 null-centered studentized paired bootstrap，
100,000 次，固定 seed；Monte Carlo p 值使用 add-one 修正

\[
p=\frac{1+\#\{T_b^\ast\le T_{\rm obs}\}}{B+1},
\]

并报告单侧上置信界、均值差、相对差和改善图像数。
holdout 为 500 类各 1 张，逐图重采样同时也是按类重采样。

### 10.2 成功定义

一个锚点只有在 `A2-A0` 与 `A2-A1` 两个 Holm 校正检验均通过时，
才支持“联合非均匀方法超过 ORFC，且非均匀分配具有独立贡献”。

| A2 vs A0 | A2 vs A1 | 结论 |
|---|---|---|
| 通过 | 通过 | 该码率 v11 成功 |
| 通过 | 未通过 | 方法超过 ORFC，但不能把收益归因于非均匀分配 |
| 未通过 | 通过 | 非均匀机制改善内部训练，但尚未超过 ORFC |
| 未通过 | 未通过 | 当前联合方法未成立 |

两个码率均成功：可声称在两个低码率锚点复现。  
仅一个成功：结论严格限定到该码率。  
没有成功：停止当前 v11 方法线，holdout 结果按事实报告。

### 10.3 描述项

- \(D(A1)-D(A0)\)；
- A2 外层轨迹、接受交换数和搜索 FLOPs；
- s42/\(\lambda=0.5\)/ep100 匹配 ORFC；
- s43/s44 seed 波动；
- 最强 ECVQ 失真及其实际 rANS 码率；
- 三模式码本存储量。

描述项不参与成功判定。

---

## 11. 代码冻结与执行入口

新代码位于：

```text
phase1/v11/
  config.py
  orfc_baseline.py
  init_menu.py
  qhard.py
  search.py
  stats.py
  bridge.py
  profile.py
  train.py
  verify.py
  unseal.py
```

v9 的 `phase1/*.py` 与 v10 的 `phase1/v10/*.py` 保持不变。
可复用实现通过 import 复用，不复制第二份搜索器。

正式 N14 前必须：

1. 将 v11 计划和源代码纳入 Git；
2. 记录 commit 与 tag；
3. 冻结数据清单、ORFC 候选清单、统计 seed 和 config；
4. 不使用额外逐文件 SHA 协议；
5. N18 后冻结 checkpoint、分配和确认 family，再写 holdout 封条。

执行入口：

```bash
# N15：ORFC 同一性与菜单合法性
python3 -m phase1.v11.verify --stage init

# N16：ORFC 工作点 Stage B
python3 -m phase1.v11.bridge --anchor R64 --decide
python3 -m phase1.v11.bridge --anchor R96 --decide

# N18：联合训练后的全部开封前闸门
python3 -m phase1.v11.verify --stage arms

# N19：冻结后一次开封
python3 -m phase1.v11.unseal --seal
python3 -m phase1.v11.unseal --confirm-unseal
```

运行约束：

```text
GPU                   2 / 3 / 6
GPU Python            /home/user/anaconda3/envs/featcodec2/bin/python
CPU Python            /home/user/anaconda3/bin/python3
PYTHONPATH            /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
SSH                   -o ClearAllForwardings=yes
```

---

## 12. 总 exit 图

\[
\boxed{
\begin{aligned}
&\text{N14 A0 合法}
\rightarrow
\text{N15 step-0 等价}
\rightarrow
\text{N16 ORFC 点存在非均匀收益}
\\
&\rightarrow
\text{N17 全长稳定 profile}
\rightarrow
\text{N18 菜单对齐且 A2 在 dev 优于 A0/A1}
\\
&\rightarrow
\text{N19 新 holdout 上同时检验 A2<A0 与 A2<A1}.
\end{aligned}
}
\]

任一节点失败均在该节点退出；不得越过失败节点打开 holdout。
