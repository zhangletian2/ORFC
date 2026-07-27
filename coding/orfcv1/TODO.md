# ORFC-v1：基于有限扰动弹性的表示学习

## 0. 本轮目标

在现有 ORFC 的编码与推理结构保持不变的条件下，仅扩展训练损失，验证正交表示 \(U\) 能否：

1. 降低固定均匀 PQ 工作点的完整 ViT tail 失真；
2. 平衡不同 PQ 组的实际误差响应；
3. 改变量化难度、有限幅度非线性和组间交互的经验指标；
4. 为下一阶段的真实相邻码率边际训练提供依据。

本轮采用固定 \(K\) 的均匀 PQ。训练后部署的 codec 仍为：

\[
F \xrightarrow{U} Z \xrightarrow{\mathrm{PQ}(K)}
\widehat Z \xrightarrow{U^\top} \widehat F .
\]

本轮输出是“固定工作点误差弹性是否可由 \(U\) 控制”的算法结论。

---

## 1. 实现前的基础修复

- [ ] 从现有 `coding/orfc/` 读取实现，所有新增代码放在 `coding/orfcv1/`。
- [ ] 修复训练输入生命周期：当前 `soft_pq.py` 堆叠 `features_train` 后删除该变量，但 epoch 末仍读取 `features_train[0]`。
- [ ] 建立互斥的 train/validation/test 划分。
  - 当前低码率配置使用全部 5000 张训练图，validation 又从同一训练池抽取。
  - 保存三个 split 的 feature ID manifest。
- [ ] 固定随机种子并保存：
  - 数据划分种子；
  - minibatch 顺序种子；
  - 组采样种子；
  - \(U\)、码本和先验初始化种子。
- [ ] 为每次运行保存完整配置、数据划分、随机种子和输出路径。
- [ ] 建立 `beta=0` 回归测试，确认新入口复现原 ORFC 的硬前向结果。
- [ ] 检查并记录每个 epoch 的
  \[
  \|U^\top U-I\|_F.
  \]

---

## 2. 暴露每组原空间量化误差

令归一化特征为 \(Y\)，旋转特征为

\[
Z=YU.
\]

PQ 的组误差为

\[
r_g=P_g(\widehat Z-Z).
\]

映射回归一化特征空间：

\[
\widetilde e_g=r_gU^\top.
\]

经过逆归一化的线性缩放后得到原特征空间误差 \(e_g\)，并验证：

\[
\widehat F-F
=
\sum_{g=1}^{G}e_g
\]

在数值精度内成立。

- [ ] 在 codec 训练前向中可选地返回：
  - clean rotated feature \(Z\)；
  - hard reconstruction \(\widehat Z\)；
  - 每组 rotated residual \(r_g\)；
  - 每组 original-space residual \(e_g\)；
  - hard labels。
- [ ] 增加 reconstruction audit：
  \[
  \frac{\|\widehat F-F-\sum_ge_g\|_2}
       {\|\widehat F-F\|_2+\epsilon}
  <10^{-5}.
  \]
- [ ] 保持默认推理接口与现有 ORFC 完全一致。

---

## 3. 完整 tail 工作点失真

令 frozen tail 为 \(h_\ell\)，clean teacher 输出为

\[
Y_0=h_\ell(F).
\]

均匀 PQ 工作点的完整失真：

\[
D_0(U)
=
\mathbb E
\left\|
h_\ell\left(F+\sum_ge_g\right)-Y_0
\right\|_2^2.
\]

- [ ] 沿用现有 teacher cache。
- [ ] 明确 loss reduction：
  - 先对 token/channel 求和；
  - 再对 batch 求均值；
  - 日志同时报告 per-image、per-token 和 per-element 版本。
- [ ] 第一轮固定长度实验关闭 learned entropy prior 对训练的影响。
  - 固定 \(K\) 后名义码率为 \(G\log_2K\) bits/token。
  - rANS 和 empirical entropy 作为 post-hoc 指标保存。

---

## 4. 有限扰动弹性探针

对实际工作点的第 \(g\) 组误差作对称缩放：

\[
D_g^\pm(U)
=
\mathbb E
\left\|
h_\ell\left(
F+\sum_{j\ne g}e_j+(1\pm\alpha)e_g
\right)-Y_0
\right\|_2^2.
\]

定义无量纲有限差分弹性：

\[
\varepsilon_g(U)
=
\frac{D_g^+(U)-D_g^-(U)}
{2\alpha\,[\operatorname{stopgrad}(D_0(U))+\epsilon]}.
\]

它对应当前量化误差方向上的工作点响应：

\[
\varepsilon_g(U)
\approx
\frac{\partial\log D}{\partial\log a_g}
\bigg|_{a_g=1}.
\]

- [ ] 支持 `alpha` 配置，首轮扫描：
  \[
  \alpha\in\{0.05,0.1,0.2\}.
  \]
- [ ] 每个 minibatch 轮换抽取组子集 \(S\)。
- [ ] 首轮比较每批抽取 4 组和 8 组的计算成本与方差。
- [ ] 一个 epoch 内保证每组被抽取次数近似一致。
- [ ] 保存每组 \(\varepsilon_g\) 的 mean、std、CV、min、max 和 span。

---

## 5. 第一轮训练损失

使用平滑极值：

\[
\operatorname{smax}_\tau(x)
=
\tau\log\sum_g\exp(x_g/\tau),
\]

\[
\operatorname{smin}_\tau(x)
=
-\tau\log\sum_g\exp(-x_g/\tau).
\]

弹性离散损失：

\[
\mathcal L_{\mathrm{elastic}}
=
\operatorname{smax}_\tau\{\varepsilon_g:g\in S\}
-
\operatorname{smin}_\tau\{\varepsilon_g:g\in S\}.
\]

总损失：

\[
\boxed{
\mathcal L_{\mathrm{v1}}
=
D_0
+
\beta\,
\operatorname{stopgrad}(D_0)\,
\mathcal L_{\mathrm{elastic}}
}
\]

使用 `stopgrad(D0)` 只做量纲恢复，使 \(\beta\) 的含义不随原始 loss reduction 改变。

- [ ] 保留 `D0` 主目标。
- [ ] 增加 `beta` 配置。
- [ ] 第一轮扫描：
  \[
  \beta\in\{0,0.01,0.03,0.1\}.
  \]
- [ ] 增加弹性 soft-extreme 温度 `elastic_tau`。
- [ ] loss 中使用当前抽样组；epoch 末在独立 calibration subset 上计算全部 32 组。
- [ ] 记录 `D0` 梯度、elastic loss 梯度及二者在 \(U\) 参数上的梯度范数和夹角。

---

## 6. 第一轮优化协议

首个受控实验：

- backbone：DINOv2 ViT-L/14；
- split：`blk20`；
- \(D=1024\)；
- \(G=32\)；
- group dimension \(d=32\)；
- \(K=8\)；
- fixed rate：128 bits/token；
- 初始化：同一个 OPQ 或同一个已训练 ORFC checkpoint。

阶段 A：

- [ ] 固定码本与先验；
- [ ] 仅训练正交 \(U\)；
- [ ] 对照：原始 \(D_0\) loss；
- [ ] 实验：\(\mathcal L_{\mathrm{v1}}\)；
- [ ] 所有对照共享数据顺序、初始化和训练步数。

阶段 B（阶段 A 有正结果后执行）：

- [ ] 交替更新：
  - \(U\)-step：固定码本，更新 \(U\)；
  - codebook-step：固定 \(U\)，更新码本；
- [ ] 与现有全参数联合训练比较。

---

## 7. 机制诊断：先记录，再决定是否加入损失

### 7.1 量化难度

\[
q_g(U)=\mathbb E\|e_g\|_2^2.
\]

- [ ] 保存 \(q_g\) 的 mean、CV、span。
- [ ] 保存 \(\operatorname{corr}(q_g,\varepsilon_g)\)。

### 7.2 工作点曲率

\[
\kappa_g(U)
=
\frac{D_g^++D_g^--2D_0}
{\alpha^2[\operatorname{stopgrad}(D_0)+\epsilon]}.
\]

- [ ] 保存 \(\kappa_g\) 的 signed/absolute mean、CV 和 span。
- [ ] 比较不同 \(\alpha\) 下 \(\kappa_g\) 的稳定性。

### 7.3 组间混合差分

对抽样组对 \((g,h)\)：

\[
I_{gh}
=
D_{gh}^{++}-D_g^+-D_h^++D_0.
\]

- [ ] 每个 batch 抽样少量组对。
- [ ] 保存 normalized interaction：
  \[
  \widetilde I_{gh}
  =
  \frac{I_{gh}}{D_0+\epsilon}.
  \]
- [ ] 比较原始 loss 与 elastic loss 训练后交互分布的变化。

首轮训练只使用 \(\mathcal L_{\mathrm{elastic}}\)。\(q_g\)、\(\kappa_g\) 和 \(I_{gh}\) 用于判断 \(U\) 通过哪一种机制改善结果。

---

## 8. 评估与结果文件

必须报告：

- [ ] hard-codec tail distortion \(D_0\)；
- [ ] classification accuracy；
- [ ] nominal fixed rate；
- [ ] empirical entropy；
- [ ] train-histogram rANS rate；
- [ ] learned-prior rANS rate（若存在）；
- [ ] \(\varepsilon_g\) mean/CV/span；
- [ ] \(q_g\) mean/CV/span；
- [ ] \(\kappa_g\)；
- [ ] sampled \(I_{gh}\)；
- [ ] \(\|U^\top U-I\|_F\)；
- [ ] train/validation/test 全部独立结果；
- [ ] 至少 3 个随机种子。

结果 JSON 至少包含：

```text
schema_version
created_at_utc
git_commit
git_dirty
config
split_manifest
history
per_group_metrics
heldout_metrics
```

核心比较：

1. OPQ；
2. 原始 ORFC；
3. 相同初始化、只训练 \(U\)、原始 \(D_0\)；
4. 相同初始化、只训练 \(U\)、elastic loss；
5. 后续交替优化版本。

---

## 9. 第一轮结论边界

本轮弹性量满足：

\[
\varepsilon_g
=
\text{实际量化误差幅度方向上的有限响应}.
\]

它描述当前固定 \(K\) 工作点的误差响应均衡程度。

真实码率边际需要：

\[
\delta_g(k)
=
D_g(k-1)-D_g(k).
\]

因此第一轮结果用于回答：

> 正交表示 \(U\) 能否在完整 ViT tail 下平衡实际 PQ 误差响应，并在固定均匀模式下保持或改善 RD？

---

## 10. 第二阶段 TODO：真实相邻码率模式

第一轮验证完成后，建立训练期辅助模式：

\[
K_-=8,\qquad K_0=16,\qquad K_+=32.
\]

三种模式共享同一个 \(U\)，每组分别拥有经过公平训练的模式码本。

固定总预算的真实交换：

\[
m_{ij}(U)
=
D_U(K_i=8,K_j=32,K_{-ij}=16)
-
D_U(K_g=16,\forall g).
\]

- [ ] 建立共享 \(U\) 的三模式 PQ。
- [ ] 先固定 \(U\)，分别优化三种模式码本。
- [ ] 再固定模式码本，优化 \(U\) 的交换余量。
- [ ] 使用周期性交替更新避免辅助模式退化。
- [ ] 训练时硬挖掘危险 donor/receiver。
- [ ] held-out 上枚举全部 \(32\times31=992\) 个单 bit 交换。
- [ ] 报告：
  \[
  \min_{i\ne j}m_{ij},\quad
  \operatorname{median}m_{ij},\quad
  \Pr(m_{ij}<0).
  \]
- [ ] 分解并估计：
  \[
  \mu_{\rm eq}(U),\qquad
  \eta_U,\qquad
  \chi_U=\eta_U/\mu_{\rm eq}(U).
  \]

若 held-out 上所有 \(m_{ij}>0\)，可以表述为：

> 对给定硬 codec、给定经验分布和固定 128 bits/token，均匀 K=16 是全部单 bit 交换邻域中的严格局部最优。

全局最优结论继续结合离散边际递减与全可行域交互控制进行判定。

---

## 11. 代码审查后的正式运行前阻断清单

### 11.1 恢复 elastic loss 到 \(U\) 的梯度

当前 `codec_v1.py` 返回的 `Z`、`Z_hat`、`r_g` 和 `e_g` 均经过
`detach()`；由它们构造的 \(D_g^\pm\) 不向 \(U\) 传播梯度。

- [ ] 将训练路径与诊断路径分开：
  - 训练路径保留被抽样组误差到 \(U\) 的计算图；
  - 日志与 held-out 诊断路径使用 detached tensor。
- [ ] 增加单元检查：
  ```text
  L_elastic.requires_grad == True
  ||grad_U L_elastic|| > 0
  ```
- [ ] 在同一初始化、同一 minibatch 上执行 beta=0 与 beta>0 的单步更新，
  验证：
  \[
  \Delta U_{\beta>0}\ne\Delta U_{\beta=0}.
  \]

### 11.2 以实际 hard reconstruction 为弹性中心

float32 Cayley 变换满足数值意义上的近似正交，因此
\(X+\sum_ge_g\) 与实际 `X_hat` 之间可能存在正交往返数值残差。

弹性探针统一改为：

\[
D_g^\pm
=
\left\|
h_\ell(X_{\mathrm{hat}}\pm\alpha e_g)-h_\ell(X)
\right\|_2^2.
\]

- [ ] `compute_elasticity` 显式接收实际 `X_hat`。
- [ ] 使用：
  ```python
  X_plus = X_hat + alpha * eg_orig
  X_minus = X_hat - alpha * eg_orig
  ```
- [ ] 增加中心一致性检查：
  \[
  X_g^\pm(\alpha=0)=X_{\mathrm{hat}}.
  \]
- [ ] 曲率中的 \(D_0\) 与 \(D_g^\pm\) 使用同一个 hard reconstruction 中心。

### 11.3 降低残差张量和 tail 反向图显存

当前完整 `e_g` 的形状为 \([G,BT,D]\)。在
\(G=32,B=32,T=257,D=1024\) 下，单个 float32 张量约占 1.08 GB。

- [ ] 训练时只为本批抽样组构造可微原空间残差。
- [ ] 使用旋转矩阵对应块直接映射：
  \[
  e_g^{\mathrm{norm}}
  =
  r_g
  \left[R^\top\right]_{gd:(g+1)d,:}.
  \]
- [ ] 全组 `e_g` 仅用于小 batch、no-grad 的诊断。
- [ ] smoke test 使用 batch size 1--2、每批 2 个组，记录 peak GPU memory。
- [ ] 再逐步测试每批 4 组和 8 组。
- [ ] 必要时对 frozen tail 使用 gradient checkpointing 或分组反向累积。

### 11.4 在 held-out 数据上计算核心弹性指标

当前全组弹性来自训练集前 200 张。正式结果增加独立评估：

- [ ] 训练期间在 validation subset 上周期性计算全 32 组
  \(\varepsilon_g\)。
- [ ] 最终在完整 validation 上计算：
  \[
  \varepsilon_g,\quad q_g,\quad\kappa_g,\quad I_{gh}.
  \]
- [ ] 模型和超参数固定后，在 test 上计算一次。
- [ ] 填充结果 JSON 中的：
  ```text
  per_group_metrics
  heldout_metrics
  ```
- [ ] 训练集指标、validation 指标和 test 指标分别存储。

### 11.5 修正 K=8 的码率说明

本轮配置为：

\[
G=32,\qquad K=8,
\]

因此固定长度码率为：

\[
R=32\log_2 8=96\ \text{bits/token}.
\]

- [ ] 将 `run_exp_v1.sh` 中的 “K=8, 128 bits/token” 改为
  “K=8, 96 bits/token”。
- [ ] 第二阶段 \(K_0=16\) 的 128 bits/token 说明继续保留。

### 11.6 完整实现梯度诊断

当前代码记录的是 \(D_0\) 梯度与总梯度的夹角，尚未读出
elastic-only 梯度。

- [ ] 分别计算：
  \[
  g_0=\nabla_U D_0,
  \qquad
  g_{\mathrm{el}}=\nabla_U\mathcal L_{\mathrm{elastic}}.
  \]
- [ ] 保存：
  \[
  \|g_0\|,\quad
  \|\beta\,\operatorname{sg}(D_0)g_{\mathrm{el}}\|,\quad
  \cos(g_0,g_{\mathrm{el}}).
  \]
- [ ] 记录 gradient clipping 前后范数和 clipping 触发比例。
- [ ] 删除未使用的 `grad_elastic_norms` 占位逻辑。

### 11.7 明确量化难度的坐标空间

当前 \(q_g\) 由归一化空间的 `e_g` 计算，TODO 中定义的是原特征空间量。

- [ ] 原特征空间指标使用：
  \[
  q_g^{\mathrm{orig}}
  =
  \mathbb E\|e_g^{\mathrm{norm}}\odot\mathrm{Std}\|_2^2.
  \]
- [ ] 同时保存归一化空间指标时，命名为：
  ```text
  q_g_normalized
  q_g_original
  ```

### 11.8 加强可复现性审计

- [ ] 使用 `git status --porcelain` 检测 tracked 与 untracked 变化。
- [ ] 先生成唯一的 OPQ initialization artifact，所有 beta 实验共同加载。
- [ ] 在 OPQ artifact 内保存 split、归一化方式、训练样本数和 OPQ
  超参数；加载时逐字段检查与当前实验配置一致。
- [ ] 并行任务使用各自的 manifest 文件或先生成只读公共 manifest。
- [ ] checkpoint 和结果文件名加入：
  ```text
  lr
  elastic_tau
  n_groups_per_batch
  result_suffix
  ```

### 11.9 修正 elastic loss 的数值基线与温度

对 \(|S|\) 个完全相等的弹性值，当前 smooth range 等于
\(2\tau\log|S|\)。该项为常数，梯度为零，但会影响不同采样规模间的
loss 数值比较。

- [ ] 使用零基线版本：
  \[
  \mathcal L_{\mathrm{elastic}}^{0}
  =
  \operatorname{smax}_\tau(\varepsilon)
  -
  \operatorname{smin}_\tau(\varepsilon)
  -
  2\tau\log|S|.
  \]
- [ ] 先用 beta=0 pilot 测量 held-out \(\varepsilon_g\) 的尺度。
- [ ] 根据弹性 std/span 选择 `elastic_tau`，并保存选择依据。
- [ ] signed mean 接近零时以 std、span、MAD 和 absolute mean 为主，
  CV 作为辅助指标。

### 11.10 正式运行前验收顺序

1. Python 语法与 import 检查；
2. differentiable elasticity 单元测试；
3. \(\alpha=0\) 中心一致性测试；
4. beta=0 与 beta>0 单步梯度差异测试；
5. 8--16 张图、1 epoch、单 GPU smoke；
6. reconstruction audit；
7. held-out 全组弹性评估；
8. peak memory 与训练时间审计；
9. beta=0 完整回归；
10. 四 GPU beta sweep。

---

## 12. 训练阶段定位与联合训练对照

只训练 \(U\) 的 Phase A 用于隔离表示变换的因果贡献；最终 ORFC-v1
算法同时评估联合训练与交替训练。

### Phase A：只训练 \(U\)

- [ ] 固定 OPQ 码本；
- [ ] 比较 \(D_0\) 与 \(D_0+\beta\mathcal L_{\mathrm{elastic}}\)；
- [ ] 回答 elastic loss 是否能独立改变 \(U\) 和 held-out 弹性。

### Phase B：\(U\) 与码本联合训练

- [ ] 从与 Phase A 相同的 OPQ artifact 初始化；
- [ ] 训练 \(U\) 和所有组码本；
- [ ] 比较：
  \[
  \mathcal L=D_0
  \quad\text{与}\quad
  \mathcal L=D_0+\beta\mathcal L_{\mathrm{elastic}}.
  \]
- [ ] 保持 K=8、96 bits/token、数据划分、训练步数和随机种子一致。
- [ ] 报告联合训练相对只训练 \(U\) 的 \(D_0\)、Acc、弹性和机制指标。

### Phase C：交替训练

- [ ] \(U\)-step：固定码本，更新 \(U\)；
- [ ] codebook-step：固定 \(U\)，更新码本；
- [ ] 比较不同 step ratio；
- [ ] 当直接联合训练出现梯度竞争或码本快速吸收 elastic 信号时，
  使用交替训练保持 \(U\) 的可辨识贡献。

最终受控矩阵至少包含：

1. OPQ；
2. 原始 ORFC：联合训练 \(U,C\)，使用 \(D_0\)；
3. U-only：使用 \(D_0\)；
4. U-only：使用 \(D_0+\beta\mathcal L_{\mathrm{elastic}}\)；
5. Joint：使用 \(D_0+\beta\mathcal L_{\mathrm{elastic}}\)；
6. Alternating：使用 \(D_0+\beta\mathcal L_{\mathrm{elastic}}\)。

Phase A 判定“损失是否作用于 \(U\)”；Phase B 判定“是否改进完整 ORFC”；
Phase C 判定“联合优化中的梯度竞争是否需要显式解耦”。

---

## 13. 代码审查后的正式运行前修改

本节记录 2026-07-26 审查中仍需完成的内容。弹性损失到 \(U\) 和码本的
可微路径已经通过最小 GPU 梯度测试；以下项目不改变训练目标，只修复实验
协议、统计估计和执行效率。

### 13.1 正式运行阻断项

- [ ] 消除 checkpoint 命名冲突。文件名必须加入：
  ```text
  step_mode
  alt_u_steps
  alt_c_steps
  result_suffix
  ```
  Phase B 和 Phase C 不得覆盖同一路径，不同交替比例的并发任务也不得
  写入同一路径。
- [ ] checkpoint 使用“临时文件写完后原子重命名”的保存方式，避免并发
  或异常中断留下半写文件。
- [ ] 将 OPQ 初始化拆成独立的串行准备阶段；所有并发训练任务只读该
  artifact。不得由多个进程同时执行“若不存在则创建”。
- [ ] OPQ artifact 文件名或内部元数据至少区分：
  ```text
  layer
  K
  embedding_dim
  norm_mode
  n_train
  split_seed
  OPQ/K-means iteration settings
  ```
- [ ] Phase C 的 beta 必须由 Phase A/B 的 validation 指标选择。删除脚本中
  未经选择直接固定 `beta=0.03` 并称为 best beta 的逻辑；test 不能参与选择。
- [ ] 增加 OPQ 到 Cayley 参数化的初始化检查：
  \[
  \frac{\|U_{\mathrm{Cayley}}-R_{\mathrm{OPQ}}\|_F}
  {\|R_{\mathrm{OPQ}}\|_F},
  \qquad
  \operatorname{cond}(R_{\mathrm{OPQ}}+I),
  \qquad
  \|U^\top U-I\|_F.
  \]
  若 Cayley 图不稳定，应停止运行或改用覆盖范围更完整的正交参数化。

### 13.2 held-out 指标的统计定义

- [ ] held-out \(\varepsilon_g\) 不再计算“各 batch 比率的非加权平均”。
  应先保存逐图的 \(D_0,D_g^+,D_g^-\)，再按预先定义的数据集级公式聚合；
  结果不得依赖 batch size、最后一个 batch 的大小或样本分组顺序。
- [ ] \(q_g\)、\(\kappa_g\) 和 interaction 的 held-out 汇总使用图像数加权，
  同时报告样本数和置信区间或 bootstrap 区间。
- [ ] 明确训练时 4 组随机 smooth-range 是完整 32 组 smooth-range 的
  随机代理，通常不是无偏估计。训练日志命名为
  `sampled_elastic_loss`，完整 32 组值仅在 held-out 路径报告。
- [ ] 增加固定 batch 划分与改变 batch size 的一致性测试，确保 held-out
  结论不因执行参数改变。

### 13.3 训练热路径的显存与吞吐

- [ ] `FeatureCodecV1.forward` 在训练时只构造采样组的原空间残差。
  当前 \(G=32,B=32,T=257,D=1024\) 的完整 `e_g` 约占 1.004 GiB，
  4 个采样组只需约 0.125 GiB。
- [ ] 训练热路径不返回未使用的完整 `Z`、`Z_hat`、`e_g` 和 labels；
  通过显式的 `detail_level=train|diagnostic` 区分输出。
- [ ] 归一化空间 \(q_g\) 直接由旋转空间 `r_g` 计算。正交变换保持范数，
  `per_image` 标准差又是逐图标量，因此无需把全部组提升回 1024 维。
- [ ] 将采样组残差的多个小矩阵乘法改为 batched GEMM。
- [ ] 将
  \[
  X_{\mathrm{hat}}\pm\alpha e_g
  \]
  沿 batch 维堆叠后批量通过 frozen tail；增加可配置
  `probe_group_chunk`，在 24GB 显存下比较 1、2、4 组分块。
- [ ] held-out 全 32 组探针使用同一分块实现。当前串行实现每 batch 约有
  68 次 tail 调用，不能作为正式高效实现。
- [ ] interaction 不在每个训练 batch 计算，只在固定 calibration batch
  或每隔若干 step/epoch 计算。
- [ ] 梯度分解诊断每个诊断 epoch 只取 1--3 个固定 batch。不得在该 epoch
  的所有 batch 上重复执行 \(D_0\)、scaled elastic、raw elastic 和 combined
  四次 backward。
- [ ] 使用 `torch.cuda.max_memory_allocated()` 和 CUDA event 分别报告：
  peak memory、images/s、tokens/s、每步 tail 调用数。优化前后必须在相同
  batch 和相同数值精度下比较。
- [ ] 先完成 FP32 正确性验证，再评估 TF32/BF16。中心差分的
  \(D_g^+-D_g^-\) 必须使用 FP32 累加，并验证混合精度不会改变弹性排序。

### 13.4 多 GPU 调度

- [ ] OPQ artifact 串行生成完成后，Phase A 和 Phase B 的 8 个独立配置
  可同时分配到 8 张 GPU。
- [ ] Phase C 必须等待 validation 选择完成后运行；记录总 optimizer step
  以及 U-step、codebook-step 的各自数量，避免把相同 epoch 数误称为相同
  参数更新预算。
- [ ] 后台任务失败时主脚本必须收集所有退出码并生成完整失败汇总，不能因
  `set -e` 或单个 `wait` 返回值提前丢失其他任务状态。

### 13.5 理论结论的边界

- [ ] 本轮固定 K8 的目标严格表述为：
  “均衡当前硬量化残差方向的有限幅度响应，并检验是否改善 ORFC 的
  \(D_0\)、准确率和 held-out 弹性。”
- [ ] 当前实验未构造相邻非均匀模式，因而不直接测量
  \[
  \mu_{\mathrm{eq}}(U),\qquad
  \eta_U,\qquad
  \chi_U=\eta_U/\mu_{\mathrm{eq}}(U).
  \]
  在引入 \(k_0-1,k_0,k_0+1\) 的真实相邻模式和单步交换实验以前，不把
  elastic loss 的改善表述为“已证明 \(\chi_U<1\)”或“均匀分配全局最优”。
- [ ] 保留 Phase A/B/C 的因果定位：Phase A 识别 \(U\) 的独立贡献，
  Phase B 判断是否改进完整 ORFC，Phase C 只在确认存在梯度竞争时作为
  解耦方案。

---

## 14. ORFC-v1 交付验收要求（Definition of Done）

本节是正式交付门槛。第 13 节描述“应如何修改”，本节规定“满足什么条件
才算完成”。任一“阻断项”未通过时，不启动或不接受正式 Phase A/B/C 结果。
按照项目约定，交付不要求任何文件摘要校验。

### 14.1 2026-07-26 审查基线

已通过：

- [x] `detail_level=train` 不再构造完整 `[G,BT,D]` 的 `e_g`。
- [x] 训练弹性探针支持 `probe_group_chunk`，正负扰动在 batch 维合并。
- [x] 最小 GPU 测试中，chunk=1 与 chunk=4 的
  \(\varepsilon_g\) 和 loss 完全一致；
  \(U\) 与码本梯度相对差异分别低于 \(10^{-8}\)。
- [x] 弹性损失可同时向 \(U\) 和码本反传。
- [x] checkpoint 名称已包含训练阶段信息和 `result_suffix`，并采用原子保存。
- [x] OPQ artifact 已加入锁、原子保存和部分配置元数据。
- [x] 梯度分解诊断已限制为诊断 epoch 的前 3 个 batch。
- [x] 当前 Phase A 单卡观测到约 92%--98% GPU utilization，
  弹性任务显存约 15--18.5 GiB，未立即触发 OOM。

当前阻断：

- [x] `evaluate_heldout_elasticity` 删除 group×image 内的逐图
  `tail.forward_nograd`。当前实现先执行合批探针，又为每张图重复执行
  正负探针；完整 test 会额外产生约 \(2NG\) 次 batch=1 tail 前向。
- [x] 合批探针直接返回 `[group, image]` 的 \(D_g^+\) 与 \(D_g^-\)，
  held-out 逐图统计必须由这些张量得到，不得重新前向计算。
- [x] 修正 held-out \(q_g\)。当前
  `compute_quantisation_difficulty` 返回 batch 均值，随后每个 batch
  只 append 一次，却被命名为 per-image。现有 500 图 smoke 中每组
  `q_g` 的 `n=125`，应为 500。
- [x] 修正 held-out interaction。当前循环结束后仅保留最后一个 group
  chunk 的 `Dplus_ch`，预设 pair 不在该 chunk 中，因此正式结果
  `interaction=None`。必须跨 chunk 保存需要的 \(D_g^+\)，并保证所有
  预设 pair 都被实际计算。
- [x] 修正 reconstruction audit 的数学对象。现有 warmup 和 smoke 分别为
  \(7.72\times10^{-5}\) 与 \(2.89\times10^{-5}\)，均为 FAIL。
  审计应分别报告：
  \[
  \widehat Y-Y_{\rm roundtrip}
  =\sum_g e_g,
  \qquad
  Y_{\rm roundtrip}-Y
  =Y(UU^\top-I),
  \]
  不能把 Cayley 的有限精度 round-trip 误差归入 PQ 组残差。
- [x] 当前正在运行的 Phase A 任务启动早于最新 `train_v1.py` 和
  `run_v1.py` 修改时间，运行进程使用的是旧版已导入代码。这批结果只用于
  调试，不得作为最新版正式交付结果。代码冻结后必须重新运行。

### 14.2 必须自动通过的单元与契约测试

交付必须提供可重复执行的测试入口，失败时返回非零退出码。至少包含：

1. [x] Python 编译、模块 import 与 shell 语法测试。
2. [x] `detail_level=train` 契约：
   - 不含完整 `e_g`、`Z`、`Z_hat` 和 labels；
   - 仅含采样组可微 `e_g_diff`；
   - `r_g` 的 shape、dtype、device 正确。
3. [x] `detail_level=diagnostic` 契约：完整组残差可用且重构关系成立。
4. [x] 分块等价性：
   \[
   \max_g|\varepsilon_g^{\rm chunk1}
   -\varepsilon_g^{\rm chunk4}|\le10^{-6},
   \]
   loss 绝对差不超过 \(10^{-6}\)，\(U/C\) 梯度相对差不超过 \(10^{-5}\)。
5. [x] 可微性：
   `L_elastic.requires_grad=True`，
   \(\|\nabla_U L_{\rm elastic}\|>0\)；Joint 模式还要求
   \(\|\nabla_C L_{\rm elastic}\|>0\)，且全部有限。
6. [x] beta 单步差异：相同初始化、相同 minibatch 下，
   beta=0 与 beta>0 的 \(U\) 更新不同。
7. [x] held-out batch 不变性：固定相同图像，使用至少两种 batch size
   和两种样本分组顺序，数据集级 \(\varepsilon_g,q_g,\kappa_g\)
   的相对差不超过 \(10^{-4}\)。
8. [x] held-out 计数契约：每个组的
   `eps_g.n == q_g.n == kappa_g.n == n_images`；
   每个预设 interaction pair 的计数大于 0。
9. [x] reconstruction audit 分离量化残差与正交 round-trip 误差；
   量化残差恒等式相对误差不超过 \(10^{-5}\)，round-trip 误差单独报告。
10. [x] checkpoint 原子保存、重新加载和一次硬前向完全可执行；并发配置的
    checkpoint/result 路径两两不同。
11. [x] OPQ artifact 缺少任一必需元数据时拒绝加载。加载检查必须覆盖
    `opq_iter`、`kmeans_iter`、`kmeans_max_samples` 及完整 split feature ID
    列表或等价的非摘要式 manifest 标识。

### 14.3 smoke 交付门槛

- [x] 提供独立 `--smoke` 或等价参数，限制 train/val/test 图像数和 epoch，
  不允许 smoke 默认跑完整 500 图 test。
- [x] smoke 至少覆盖 beta=0、beta>0、U-only、Joint、Alternating。
- [x] 所有 smoke：
  - 退出码为 0；
  - 无 Traceback、OOM、NaN、Inf；
  - reconstruction audit 通过；
  - checkpoint 可加载；
  - result JSON 可解析；
  - held-out `q_g` 图像计数正确；
  - interaction 非空；
  - beta>0 时 \(U\) 弹性梯度非零。
- [x] smoke 保存每步或每 epoch 的：
  `peak_memory_mb`、images/s、tokens/s、常规训练 tail 调用数和
  held-out tail 调用数。

### 14.4 GPU 与性能交付门槛

- [x] 常规弹性训练 batch 在 \(S=4\)、`probe_group_chunk=4` 时，
  tail 调用数应为：
  \[
  1\ \text{次基础前向}
  +1\ \text{次合并正负探针前向},
  \]
  calibration interaction 除外。
- [x] held-out 探针 tail 调用数应随 group chunk 数增长，不得随
  \(N\times G\) 的逐图循环增长；代码中不得存在 group×image 嵌套的
  tail 前向。
- [x] training 模式不得分配完整 `[G,BT,D]` tensor；interaction 接口接受
  sampled dict 或紧凑 tensor，不得为少数 pair 创建约 1 GiB 的
  `e_g_ixn_full`。
- [x] 记录 `probe_group_chunk` 为 1、2、4 的 peak memory 和吞吐，
  最终值以“不 OOM 且 images/s 最高”为准，不仅依据 GPU utilization。
- [x] 正式 batch=32 的 peak allocated memory 必须低于 22 GiB，给
  24GB GPU 保留运行时余量。
- [ ] 正式训练阶段单卡 median GPU utilization 目标不低于 80%，同时报告
  images/s；仅有 utilization 而无吞吐不算性能交付。
- [ ] OPQ 与共享 feature/teacher cache 只生成一次。Phase A/B 的独立配置
  在资源允许时使用 8 张 GPU 并发，Phase C 在 validation 选择后启动。

### 14.5 实验协议交付门槛

- [x] 正式运行前冻结代码并记录一个明确的版本标签、Git 状态、配置和
  文件清单。运行过程中不得修改已被进程导入的源文件；修改后所有受影响
  的结果重新生成。
- [x] 每次正式 sweep 使用独立 `run_id` 或独立结果目录，Phase C 选择器
  只能读取本次运行且已成功完成的 Phase A/B 结果，不能读取历史残留 JSON。
- [x] split manifest 原子保存或按 `run_id` 分离；并发进程不得覆盖包含
  不同 `created_utc` 的同一路径。
- [x] OPQ metadata 校验必须是严格 schema：预期字段缺失即失败，不能因为
  `key not in meta` 而跳过。
- [x] Phase C beta 选择规则在查看 test 前预注册。当前脚本优先选择
  Phase A，只在 Phase A 不存在时查看 Phase B，这不等于比较 A/B。
  应明确：
  - Phase C 只继承 Phase B 的 beta；或
  - 在 A/B 全部候选上执行同一 validation 规则。
- [x] beta 不能仅按最小 `val_D0` 选择，否则大概率退化为 beta=0。
  推荐预注册约束规则：在 `val_D0` 相对 beta=0 的退化不超过
  \(\rho_D\) 的候选中，最小化 held-out 弹性 span/MAD；交付时报告
  \(\rho_D\) 和完整候选表。
- [x] `elastic_tau` 由 beta=0 validation pilot 的弹性尺度确定，保存选择
  公式和候选值，不能一直使用无依据的默认值 1.0。
- [x] Phase A/B/C 报告总 optimizer step、U-step 和 C-step；相同 epoch
  不能表述为相同参数更新预算。
- [x] 自动选择只使用 validation。test 只在方案和超参数完全冻结后执行。

### 14.6 正式结果交付内容

交付目录至少包含：

- [ ] 冻结版源文件清单、环境信息、完整 CLI 配置和随机种子；
- [ ] train/validation/test feature ID manifest；
- [ ] OPQ artifact 元数据与 Cayley 初始化诊断；
- [ ] 每个配置的完整日志、checkpoint、结构化 result JSON；
- [ ] 单元测试、smoke、显存/吞吐验收报告；
- [ ] Phase A/B 全部 validation 候选表和 Phase C 选择记录；
- [ ] OPQ、原始 ORFC、U-only、Joint、Alternating 的
  \(D_0\)、accuracy、rate、\(\varepsilon_g\)、\(q_g\)、\(\kappa_g\)、
  interaction、梯度范数/夹角及置信区间；
- [ ] 至少 3 个随机种子的最终选定配置，报告均值、标准差和失败运行；
- [ ] 对固定 K8 结论使用“有限响应均衡/ORFC 改进”的表述，不把结果直接
  宣称为 \(\chi_U<1\) 或均匀分配全局最优。

只有 14.2--14.5 全部通过，且 14.6 材料齐全，ORFC-v1 才进入正式交付状态。

### 14.7 冻结版验收记录（2026-07-26）

冻结标签：

```text
orfcv1-delivery-20260726
```

已完成的可重复验收入口与结果：

- `test_v1_contracts.py`：CUDA 契约测试全部通过，覆盖 14.2 的
  train/diagnostic 输出、分块等价、梯度、beta 单步差异、held-out
  batch/顺序不变性、计数、interaction、重建、checkpoint 与 OPQ metadata。
- `run_smoke_v1.sh`：冻结版四配置 smoke 全部通过，正式结果目录为
  `results/dinov2_vitl14/smoke_final_20260726/`。
- `benchmark_probe_chunks_v1.sh`：batch=32 的实测报告位于
  `results/dinov2_vitl14/chunkbench_frozen_20260726/`
  `chunk_benchmark_report.json`。chunk=1/2/4 的峰值显存分别为
  11608/11679/12798 MB，吞吐分别为 37.52/34.26/33.42 images/s。
  因此正式训练探针选 chunk=1；held-out no-grad 全组探针独立使用
  chunk=4。
- `guard_v1.sh RUN_ID`：每 1800 秒记录 tmux、GPU、进程、日志错误模式和
  结构化结果契约；发现重建、逐图计数、interaction 或 JSON 问题时生成
  `GUARD_ALERT` 并返回非零状态。

尚未完成、只能在正式 sweep 结束后勾选的项目：

- 14.4 的正式运行 median GPU utilization 与完整 images/s 汇总；
- 14.6 的全部配置结果、三随机种子统计和最终实验报告。

---

## 15. ORFC-v1.1：响应目标修正与 K256 高码率验证

本节规定下一轮代码和实验。它不修改第 14 节已经完成的
`formal_k8_20260726T175547Z` 历史结果；对 ORFC-v1.1 新运行而言，本节的
随机种子、实验矩阵和交付要求覆盖第 12、14 节中“三随机种子”和
Alternating 正式对照的旧要求。

### 15.1 范围和结论边界

- [ ] 正式训练、validation 选择和最终 test **只使用 seed 42**。
- [ ] 删除 ORFC-v1.1 驱动脚本中的 seed 43、44 最终重复训练，不再报告
  三种子均值和标准差。
- [ ] K8 是实际低码率工作点：
  \[
  G=32,\quad K=8,\quad R=32\log_2 8=96\ {\rm bits/token}.
  \]
- [ ] 增加 K256 高码率验证点：
  \[
  G=32,\quad K=256,\quad R=32\log_2 256=256\ {\rm bits/token}.
  \]
- [ ] K256 只用于检验高码率下量化误差、非线性尺度依赖和响应均衡是否
  发生预期变化，不参与 K8 的损失形式、beta 或其他超参数选择。
- [ ] K256 是独立的固定 K 高码率对照，不等于同一模型内的相邻模式菜单，
  不能据此声称已经计算
  \(\delta_g(k_0),\delta_g(k_0+1)\) 或证明 \(\chi_U<1\)。
- [ ] 本轮结论仍限定为“固定 K 下的有限响应建模与均衡”。真正的
  \(\mu_{\rm eq}-\eta\) 训练留到多模式 ORFC-v2。

### 15.2 统一 D0 的定义和数值尺度

代码中的基点失真明确记为

\[
D_{\rm tail}(F,\widehat F)
=
\frac1B\sum_{i=1}^{B}
\left\|
h_\ell(\widehat F_i)-h_\ell(F_i)
\right\|_F^2.
\]

- [ ] 结果文件保留历史兼容字段 `D0_raw`，其值为按图像平均、对全部
  token 和通道求和的 \(D_{\rm tail}\)。
- [ ] 新增
  ```text
  D0_per_token
  D0_per_element
  ```
  并明确优化器实际使用哪一个。
- [ ] ORFC-v1.1 优化器使用
  \[
  \overline D_0
  =
  D_{\rm tail}/(T D)
  \]
  消除 token/通道数带来的量纲放大；报告时同时给出 raw 和 per-element。
- [ ] 重新测量 gradient clipping。不得接受所有 batch 的 clipping
  触发率仍为 100%；保存 clipping 前后范数和触发比例。
- [ ] 保存
  \[
  \|g_{\overline D_0}\|,\qquad
  \|g_{\rm response}\|,\qquad
  \cos(g_{\overline D_0},g_{\rm response}),
  \qquad
  \|g_{\rm response}\|/\|g_{\overline D_0}\|.
  \]
- [ ] 论文和日志中不把 \(D_0\) 描述成 accuracy 或分类损失；当前 frozen
  tail 不含分类 head。

### 15.3 同时定义“传播响应”和“实际修复收益”

对第 \(i\) 张图、第 \(g\) 组，令原特征空间实际量化残差为
\(e_{g,i}\)，定义

\[
q_{g,i}
=
\frac1T\|e_{g,i}\|_F^2,
\qquad
\bar q_i
=
\frac1G\sum_{g=1}^{G}q_{g,i}.
\]

#### 15.3.1 固定能量传播探针

使用 stop-gradient 的能量匹配：

\[
\widetilde e_{g,i}
=
e_{g,i}
\sqrt{
\frac{\operatorname{sg}(\bar q_i)}
{\operatorname{sg}(q_{g,i})+\epsilon}
}.
\]

于是每张图内所有组的探针能量相同，传播响应不再因组间
\(\|e_g\|\) 不同而直接变化。

- [ ] 新增逐图 `q_g_original`，训练探针使用逐图能量匹配，不使用一个
  batch 全局标量替代。
- [ ] \(q_g\) 和 \(\bar q\) 在归一化因子中必须 detach；固定能量归一化
  不得产生“通过改变分母降低 loss”的梯度路径。
- [ ] 数值审计：
  \[
  \max_{i,g}
  \frac{
  |\|\widetilde e_{g,i}\|_F^2/T-\bar q_i|
  }{\bar q_i+\epsilon}
  \le 10^{-5}.
  \]
- [ ] 对
  \[
  \mathcal A=\{0.1,0.5,1.0\}
  \]
  同时计算单侧固定能量响应
  \[
  S_{g}^{(\alpha)}
  =
  \frac{
  D_{\rm tail}(\widehat F)
  -
  D_{\rm tail}(\widehat F-\alpha\widetilde e_g)
  }{
  \alpha\,[D_{\rm tail}(\widehat F)+\epsilon]
  }.
  \]

#### 15.3.2 实际误差修复收益

同时保留实际残差，不做能量归一化：

\[
M_{g}^{(\alpha)}
=
\frac{
D_{\rm tail}(\widehat F)
-
D_{\rm tail}(\widehat F-\alpha e_g)
}{
D_{\rm tail}(\widehat F)+\epsilon
}.
\]

- [ ] \(\alpha=1\) 时，
  \(\widehat F-e_g\) 必须等价于只修复第 \(g\) 组量化误差；增加旋转空间
  与原空间的一致性单元测试。
- [ ] `M_g_alpha1` 作为当前固定 K 下最接近 operational group marginal
  的量；它仍然不是相邻码率模式收益。
- [ ] 原中心差分
  \[
  (D_g^+-D_g^-)/(2\alpha D_0)
  \]
  政名为 `legacy_signed_elasticity`，仅作历史对照，不再默认作为主损失。
- [ ] 保存各组、多 \(\alpha\) 的 \(S_g^{(\alpha)}\)、
  \(M_g^{(\alpha)}\)、\(q_g\)、\(\kappa_g\) 和 interaction，以及逐图
  统计和置信区间。
- [ ] 新增尺度非线性指标，例如
  \[
  N_g
  =
  \max_{\alpha\in\mathcal A}
  \left|
  \frac{M_g^{(\alpha)}}{\alpha}
  -
  \frac{M_g^{(0.1)}}{0.1}
  \right|,
  \]
  同时报告固定能量版本。该指标用于比较 K8 与 K256，不预设其必然为零。

### 15.4 响应均衡损失

当前每 batch 只采样 4/32 组。sampled smooth max-min 不是完整组 range
的无偏估计，因此主训练损失改为 sampled pairwise dispersion。

对采样集合 \(\mathcal S\)，定义

\[
\mathcal L_{\rm pair}(z)
=
\frac{2}{|\mathcal S|(|\mathcal S|-1)}
\sum_{\substack{g<h\\g,h\in\mathcal S}}
(z_g-z_h)^2.
\]

- [ ] 为以下三种目标提供显式命令行选项：
  ```text
  --response_objective legacy
  --response_objective fixed_energy
  --response_objective operational
  ```
- [ ] `legacy` 复现现有 signed central response，只用于受控基线。
- [ ] `fixed_energy` 优化
  \[
  \sum_{\alpha\in\mathcal A}
  w_\alpha\,
  \mathcal L_{\rm pair}(S^{(\alpha)}).
  \]
- [ ] `operational` 以
  \[
  \mathcal L_{\rm pair}(M^{(1)})
  \]
  为主；可将 \(\alpha=0.1,0.5\) 作为带独立权重的尺度一致性正则。
- [ ] 默认权重、alpha 列表和归一化方式全部写入 result JSON，不使用代码
  内隐式常数。
- [ ] 每隔固定 epoch 在 validation 上计算完整 32 组 hard range、MAD、
  pairwise dispersion 和最大/最小组；训练时 sampled loss 与 held-out
  full-group 指标使用不同字段名。
- [ ] pairwise loss 单元测试：
  - 所有组相等时严格为 0；
  - 任意组排列不改变结果；
  - 对均匀随机采样组，其期望等于完整组 pairwise dispersion；
  - chunk=1/2/4 的数值与梯度一致。
- [ ] 优化目标继续包含 \(\overline D_0\)。只让 response range 变小而使
  response mean、\(D_0\)、\(q_g\) 或 \(\kappa_g\) 恶化的方案不得自动选中。

### 15.5 训练策略

- [ ] **Joint \(U,C\) 作为 ORFC-v1.1 主算法。**
- [ ] U-only 只对 validation 选出的最佳新响应目标运行一次，用于验证该
  目标是否能够单独作用于 \(U\)；不把 U-only 当最终候选。
- [ ] 本轮不正式运行 Alternating。已有实验中
  \(\cos(g_{D_0},g_{\rm response})\) 接近 0，没有强负梯度冲突证据，而且
  原 Alternating 在相同 epoch 下减少了每个参数的 optimizer step。
- [ ] 若将来恢复 Alternating，必须按 \(U\)-step 和 \(C\)-step 分别与
  Joint 对齐更新次数和学习率调度，不能只对齐 epoch。
- [ ] beta 不直接沿用旧损失的数值尺度。先在 seed 42、validation pilot
  上测量未加权响应梯度，然后构造使
  \[
  \|g_{\rm response}\|/\|g_{\overline D_0}\|
  \in\{0.05,0.10,0.20\}
  \]
  的 beta 候选；保存换算过程。
- [ ] K8 beta/目标选择规则预注册为：首先要求相对 beta=0 的 validation
  \(D_{\rm tail}\) 退化不超过 5%，然后最小化
  `M_g_alpha1` 的完整组 pairwise dispersion；以固定能量响应 dispersion、
  \(D_{\rm tail}\) 依次作为 tie-break。
- [ ] test 不参与响应目标、alpha 权重、beta 或 checkpoint 选择。

### 15.6 ORFC-v1.1 受控实验矩阵

所有配置只使用 seed 42。

#### K8 主实验

1. [ ] 标准 OPQ；
2. [ ] 原始 ORFC：Joint，\(\beta=0\)；
3. [ ] Legacy Joint：现有中心差分目标；
4. [ ] Fixed-energy Joint；
5. [ ] Operational-repair Joint；
6. [ ] validation 选出的最佳新目标对应的 U-only 因果对照；
7. [ ] validation 完全冻结后，只对原始 ORFC和最佳 Joint 执行 test；
   其他候选默认只报告 validation，除非在运行前明确预注册为统一 test 对照。

#### K256 高码率验证

1. [ ] 标准 OPQ K256；
2. [ ] 原始 ORFC K256：Joint，\(\beta=0\)；
3. [ ] 将 K8 validation 选定的响应目标、alpha 权重和数值 beta
   **原样迁移**到 K256 Joint，不在 K256 上重新选择；
4. [ ] K256 test 只在 K8 配置完全冻结后执行；
5. [ ] 单独记录 K256 OPQ/K-means 训练时间、峰值显存、码本利用率、
   dead entries 和吞吐。

K256 必须满足以下基础 sanity：

- [ ] `max_rate_bpt == 256`；
- [ ] 每组 label 严格位于 `[0,255]`；
- [ ] OPQ artifact 和 checkpoint 文件名、metadata 显式包含 `K256`；
- [ ] K8 与 K256 不得共享或误加载 OPQ artifact；
- [ ] 同一 split/seed 下，K256 的实际量化难度和 \(D_{\rm tail}\) 应作为
  高率 sanity 与 K8 对照；若没有下降，判为实现或收敛异常并先诊断；
- [ ] 比较 K8/K256 的多尺度 \(N_g\)、\(S_g^{(\alpha)}\)、
  \(M_g^{(\alpha)}\)、\(\kappa_g\) 和 interaction，检验高码率点是否更接近
  局部响应区。该比较是经验诊断，不设置“必须支持理论”的结果门槛。

### 15.7 代码接口和性能要求

- [ ] `elastic.py`：
  - 实现逐图能量匹配；
  - 实现多 alpha 单侧 batched probe；
  - 实现 fixed-energy、operational 和 legacy 指标；
  - 实现 sampled pairwise dispersion。
- [ ] `train_v1.py`：
  - 使用新 `response_objective`；
  - 保存各损失梯度和 clipping 诊断；
  - Joint 为默认；
  - sampled 组仍使用统一覆盖的 sampler。
- [ ] `codec_v1.py`：
  - 复用现有 `e_g_diff` 与 `r_g`；
  - 不得重新构造训练热路径的完整 `[G,BT,D]` 残差；
  - 只有在 α=1 修复审计需要时使用紧凑 sampled-group 张量。
- [ ] `run_v1.py`：
  - 结构化保存所有 response 配置；
  - 分别保存 validation/test 的 fixed-energy 与 operational 指标；
  - 同时保存 raw/per-token/per-element D0。
- [ ] 新建独立 ORFC-v1.1 驱动脚本，禁止覆盖
  `formal_k8_20260726T175547Z` 的日志、checkpoint 和结果。
- [ ] 多 alpha 必须沿 group×sign/repair×batch 维合批；不得退化成
  group×image×alpha 的逐图 tail 循环。
- [ ] K8 与 K256 分别实测 probe chunk 的峰值显存和 images/s；24GB GPU
  的峰值 allocated memory 继续要求低于 22 GiB。

### 15.8 必须新增的契约测试

- [ ] \(\alpha=0\) 时所有 probe 严格回到基点 \(D_{\rm tail}\)。
- [ ] \(\alpha=1\) 的 actual repair 在旋转空间和原特征空间一致。
- [ ] 固定能量探针逐图、逐组满足 15.3.1 的能量误差阈值。
- [ ] `q_g.detach()` 契约：归一化分母没有梯度，响应仍可向 \(U,C\) 反传。
- [ ] 单侧 batched probe 与逐组参考实现数值一致，且 U/C 梯度相对差
  不超过 \(10^{-5}\)。
- [ ] 多 alpha 的输出不因 chunk 或 batch 划分变化。
- [ ] sampled pairwise loss 的 15.4 四项性质全部通过。
- [ ] seed 审计：正式新结果只能出现 seed 42；发现 43/44 即失败。
- [ ] K256 的 label、rate、artifact metadata、checkpoint reload 和
  reconstruction audit 全部通过。
- [ ] 自动选择继续满足 test isolation。

### 15.9 ORFC-v1.1 交付内容

- [ ] 只交付 seed 42，不再要求三种子统计；报告明确注明这是单种子受控
  机制实验，不给出跨种子方差结论。
- [ ] K8 与 K256 各自的完整 CLI、manifest、OPQ metadata、日志、
  checkpoint、result JSON 和性能报告。
- [ ] K8 的 OPQ、原始 ORFC、Legacy、Fixed-energy、Operational 和最佳
  U-only validation 表。
- [ ] K8/K256 原始 ORFC与最佳 Joint 的统一 test 对比，至少包含：
  \[
  D_{\rm tail},\ {\rm accuracy},\ {\rm rate},\
  q_g,\ S_g^{(\alpha)},\ M_g^{(\alpha)},\
  N_g,\ \kappa_g,\ {\rm interaction}.
  \]
- [ ] 报告响应梯度相对 D0 梯度的比例、夹角、clipping 触发率、峰值显存
  和吞吐。
- [ ] 分别回答：
  1. 固定能量后，组间传播响应是否仍异质；
  2. 实际修复收益的均衡是否改善；
  3. 改善来自 \(q_g\)、传播响应还是二者共同变化；
  4. K256 是否比 K8 更接近局部响应区；
  5. 响应均衡是否在不显著损失 \(D_{\rm tail}\) 的情况下实现。
- [ ] 结论不得写成“已经证明均匀分配全局最优”。在没有同一模型的相邻
  码率模式之前，仍不能直接计算理论中的 \(\mu_{\rm eq}\)。

### 15.10 后续 ORFC-v2（本轮不实现）

为了直接训练

\[
\mu_{\rm eq}
=
\min_g\delta_g(k_0)
-
\max_h\delta_h(k_0+1)
\]

和

\[
\mathcal L_{\rm cert}
=
[\eta+\rho-\mu_{\rm eq}]_+,
\]

后续需要增加同一模型内的相邻模式菜单，例如
\(K\in\{4,8,16\}\)，并支持逐组模式 \(k_g\)、真实模式成本和单组
升级/降级。不得用独立 K8/K256 模型之间的差值替代同一模型中的
operational marginal。

### 15.11 代码审查修复记录（2026-07-26）

以下代码修改已经完成；正式 K8/K256 训练仍未启动。

- [x] 固定能量目标的 \(\bar q_i\) 从完整 \(G=32\) 组一次性计算，
  sampled-group 与 held-out chunk 只复用该 stop-gradient 目标。
- [x] held-out 保存逐图 raw \(D_0,D_{\rm probe}\)，遍历结束后使用全数据集
  mean \(D_0\) 计算 \(S,M,\kappa\) 和 interaction，消除 batch/chunk/order
  依赖。
- [x] 新增 operational 与 fixed-energy 的逐图尺度非线性、CI、MAD 和
  最大/最小组。
- [x] operational 默认权重固定为 `0,0,1`；fixed-energy 默认三 alpha
  等权，配置写入 JSON 和文件名。
- [x] 增加首 minibatch 未加权响应梯度测量以及
  `--beta_target_ratio`，正式 K8 候选覆盖 `0.05/0.10/0.20`。
- [x] 增加 validation-only 选择器 `select_v1_1.py`；选择器拒绝任何包含
  test 字段的结果，并执行 5% D0 gate 与预注册 tie-break。
- [x] 增加 `--eval_checkpoint` 冻结评估，final test 不再重新训练。
- [x] 增加 `--opq_artifact`、`--opq_reference_json` 和
  `--codec_reference_json`；旧 Acc 只有在 split/config/checkpoint 契约
  全部匹配时才复用。
- [x] K8 冻结 OPQ Acc `0.234` 与冻结原始 ORFC Acc `0.896` 直接复用；
  新 Joint 和全部 K256 checkpoint 的 Acc 在冻结后各计算一次。
- [x] 修正远端 Conda 启动方式为绝对路径加
  `conda run --no-capture-output`。
- [x] 新增 V1.1 GPU 契约测试：全组能量匹配、stop-gradient、alpha=0、
  alpha=1 repair、probe chunk/gradient、held-out batch/chunk/order
  不变性和 K256 label/checkpoint；`test_v1_contracts.py` 全部通过。
- [ ] 执行 `run_exp_v1_1.sh ... validation`，冻结 K8 选择。
- [ ] 人工复核 `selection_v1_1.json` 后执行
  `run_exp_v1_1.sh ... postselect`。

---

## 16. 固定总码率余项测量（Smoke 已通过，正式实验未运行）

### 16.1 最小改动原则

- [x] 保留现有 `SoftPQ`、`FeatureCodecV1`、teacher cache、frozen tail 和
  下游评估主路径。
- [x] 在 `coding/orfcv1/` 内新增 `MultiModeSoftPQ`，不修改当前存在用户
  改动的 `coding/orfc/soft_pq.py`。
- [x] 多模式只组合现有 `SoftPQ` 实例，共享同一个正交表示和分组；
  不复制 ORFC 编码器、tail 或训练主流程，默认选择最大模式时保持原接口。
- [x] `FeatureCodecV1.forward` 只增加可选逐组 `modes` 参数；单模式调用
  和旧 checkpoint 加载保持兼容。

### 16.2 已实现接口

- [x] `multimode_pq.py`
  - 逐组选择模式；
  - 固定长度模式码率；
  - hard/soft assignment、usage、label 和可选先验码率沿用 `SoftPQ`
    语义；
  - 模式元数据进入 checkpoint。
- [x] `fixed_rate_remainder.py`
  - 名义或外部实际码长表；
  - 固定总码率契约检查；
  - 相邻两组交换、随机多步交换和小菜单完整枚举；
  - 复用现有归一化、codec 和 frozen-tail 批量前向；
  - 计算 \(D,\Phi,E=D-\Phi,\widehat\Omega\)、理想间隔、候选集合及
    单步交换余项变化；
  - 保留逐图数组以支持配对置信区间。

### 16.3 运行前必须完成

- [x] 本阶段保持 `train_v1.py` 不变；P1 先在同一冻结 \(U\) 下按模式复用
  `SoftPQ.init_from_kmeans` 建立码本菜单。P2 再接入多分配联合训练。
- [x] 正式模式菜单预注册为 K8/K16/K32/K64/K128/K256；所有模式在
  同一个冻结 \(U\) 下独立初始化，不能直接拼接具有不同 \(U\) 的
  K8/K256 checkpoint。
- [x] P1 采用 3--8 bit/group 的固定长度模式成本，明确标记为受控代理；
  实际熵码长留作后续独立实验，validation/test 不重新估计成本。
- [x] 用训练集上的实际模式量化误差和有限差分 JVP 估计与当前 \(U\)
  一致的 \(c_g\)，blk20 步长固定为 0.01。
- [x] 增加薄入口 `p1_fixed_rate.py`，直接复用 codec、teacher cache、
  frozen tail 和 `fixed_rate_remainder.py`。
- [ ] 补充交换回返及线性 tail 的 \(E=0\) 单元契约；多模式 checkpoint、
  固定总码率和 allocation chunk 已由 smoke 验证。

### 16.4 Smoke 记录

- [x] `RUN_ID=p1_smoke_20260727T083454Z`；Identity、OPQ、原始 ORFC、
  response-ORFC 四个独立任务分别使用 GPU 4、5、6、7。
- [x] K8/K16 菜单、有限差分系数、112 bit/token 等预算候选、完整
  tail 失真及汇总审计全部通过；4 个 arm-budget 结果的码率误差均为 0。
- [x] 批量 allocation 重构与逐 allocation 参考实现相对误差为 0；
  allocation chunk 1/4 的失真相对差为 \(1.24\times10^{-7}\)。
- [x] Smoke 只使用 2 张图和 9 个候选验证链路，数值不作科学结论；
  正式实验尚未启动。

### 16.5 正式协议

- [ ] 四个阶段依次为：多模式菜单准备、训练集系数与候选校准、
  validation 完整余项测量、CPU 汇总审计。
- [ ] GPU 4--7 只做四个表示任务的并发；单任务内部保持单卡执行。
- [ ] 正式预算为 128/192 bit/token，validation 使用 300 张图；
  每个预算保留解析候选、32 个单次交换和 32 个随机多组交换。
