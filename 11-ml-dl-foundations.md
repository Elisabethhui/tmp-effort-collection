# 11｜机器学习与深度学习基础：算法岗不能丢的底座

> 目标：即使岗位写着“大模型算法”，面试官仍可能用经典 ML/DL 判断你的基础是否扎实。  
> 标准：不是背定义，而是能解释假设、公式、优化目标、失败条件和指标选择。

---

# 1. 监督学习的基本框架 `S`

一个监督学习问题至少包含：

```text
Data (x, y)
 → Model fθ(x)
 → Objective L(fθ(x), y)
 → Optimizer
 → Validation Metric
 → Generalization
```

面试高频挖坑：**loss 和 metric 为什么可以不一样？**

- loss 要可优化、通常可微；
- metric 反映最终业务目标；
- loss 下降不代表线上指标一定提升。

2026 小红书面经里就出现了“MSE 收敛但排序效果下降怎么定位”，本质就是在考这个区别。

---

# 2. Bias / Variance / Overfitting `S`

## 2.1 Bias

模型表达能力或假设带来的系统性误差。

## 2.2 Variance

模型对训练样本扰动过于敏感。

## 2.3 常见调节手段

| 问题 | 可能方案 |
|---|---|
| 高 bias | 更强模型、更多特征、训练更充分 |
| 高 variance | 更多数据、正则化、数据增强、降低容量、early stopping |

### 挖坑

“训练集和验证集都差”通常偏 high bias；“训练好、验证差”通常偏 high variance，但数据分布偏移、标签错误也会产生类似现象，不能机械套公式。

---

# 3. Logistic Regression `S`

## 3.1 为什么是 sigmoid

二分类时令：

```text
p(y=1|x) = sigmoid(w^T x + b)
```

其 log-odds 与特征线性相关：

```text
log(p/(1-p)) = w^T x + b
```

## 3.2 损失

最大似然等价于最小化 BCE / log loss。

## 3.3 LR 为什么仍会被问

因为它能同时考：

- 线性模型；
- 概率解释；
- BCE；
- 正则化；
- 特征工程；
- calibration；
- class imbalance。

## 3.4 L1 vs L2 `S`

- L1：稀疏、可做特征选择；
- L2：平滑收缩、数值更稳定；
- 不要简单回答“L1 一定更好做特征选择”，强相关特征下选择可能不稳定。

---

# 4. SVM `A`

## 4.1 Maximum Margin

SVM 不是“找一条分类线”，而是最大化 margin。

Hard-margin 适合线性可分；soft-margin 引入 slack 和 `C`。

## 4.2 C 的意义

- C 大：更惩罚训练错误，margin 可能更窄；
- C 小：允许更多违反 margin，正则更强。

## 4.3 Kernel Trick

不显式构造高维特征，而用 kernel 计算内积。

### 常见追问

- RBF 的 gamma 什么作用？
- 为什么大数据常不用 kernel SVM？
- SVM 与 LR 区别？

---

# 5. Decision Tree / GBDT / XGBoost `S/A`

## 5.1 Decision Tree

分类常见划分指标：

- entropy / information gain；
- Gini impurity。

### 为什么树不需要 feature scaling？

划分主要依赖阈值和排序，不依赖欧氏距离尺度。

## 5.2 Bagging vs Boosting

- Bagging：并行训练基学习器，主要降 variance；
- Boosting：串行纠错，逐步拟合残差/负梯度。

## 5.3 GBDT

每一轮训练新的弱学习器去拟合当前 loss 关于预测值的负梯度。

## 5.4 XGBoost

面试不要只答“更快”。应掌握：

- 二阶 Taylor expansion；
- 显式 regularization；
- shrinkage；
- column subsampling；
- missing value handling；
- histogram/approximation 相关优化。

### 高频比较：XGBoost vs LR

- LR：线性、可解释、稀疏高维适配好；
- XGBoost：非线性、自动特征交叉能力强；
- 大规模稀疏推荐场景也不能简单说“树一定更好”。

---

# 6. K-Means / Clustering `A`

2026 美团大模型面经也出现 K-Means 是否存在全局最优解的追问。

## 6.1 Objective

最小化簇内平方距离。

## 6.2 Lloyd Algorithm

反复：

```text
assign points → recompute centroid → repeat
```

保证目标不增，但**不保证全局最优**，依赖初始化。

## 6.3 K-Means++

改善初始化，降低坏局部解概率。

## 6.4 常见坑

- 需要标准化吗？距离型算法通常需要考虑尺度；
- 非球形簇效果差；
- 对 outlier 敏感；
- K 如何选：elbow/silhouette 只是辅助，不是绝对规则。

---

# 7. PCA `A`

两种等价视角：

1. 最大化投影方差；
2. 最小化线性重构误差。

通过 covariance matrix eigen decomposition 或 SVD 得到主成分。

### 高频追问

- PCA 前为什么常标准化？
- PCA 和 autoencoder 区别？
- PCA 会不会使用标签？不会，是无监督线性降维。

---

# 8. Loss Functions `S`

## 8.1 MSE

适合连续值回归，隐含高斯噪声假设的经典解释。

## 8.2 MAE

对 outlier 更鲁棒，但 0 点不可导通常用 subgradient/平滑版本处理。

## 8.3 BCE

二分类。工程中优先 `BCEWithLogitsLoss`，避免先 sigmoid 再 log 的数值问题。

## 8.4 Cross Entropy

多分类 / token classification / causal LM 的核心。

## 8.5 Focal Loss `B`

用于严重 class imbalance / easy negative dominating 的场景。

## 8.6 Contrastive / Triplet / InfoNCE

大模型检索和多模态会高频出现，详见 13/14 章节。

---

# 9. Optimizers `S`

2026 字节多模态面经直接问“讲几种优化器”。

## 9.1 SGD

```text
θ ← θ - lr * grad
```

## 9.2 Momentum

累积历史梯度方向，加快一致方向移动并降低震荡。

## 9.3 Adam

同时维护一阶/二阶矩估计。

### Adam 为什么需要 bias correction？

初始 moment 为 0，前几步估计偏向 0，需要校正。

## 9.4 AdamW

把 weight decay 与 gradient update 解耦。大模型训练常见。

### 高频追问

- Adam 为什么训练快但有时 generalization 不如 SGD？
- weight decay 和 L2 regularization 在 Adam 下是否完全等价？不是。
- 哪些参数通常不做 weight decay？bias、norm scale 常被排除，但要看实现。

---

# 10. Learning Rate / Warmup / Scheduler `S`

## 为什么 warmup

Transformer 大模型训练初期参数/梯度统计尚不稳定，直接上大 LR 容易发散。

常见：

- linear warmup + cosine decay；
- constant with warmup；
- inverse square root（经典 Transformer）。

### 挖坑

“warmup 是因为 Adam 不稳定”太窄。它与深层网络、初始化、batch、优化器 moment 和训练早期激活/梯度分布都有关。

---

# 11. Normalization `S`

## BatchNorm

依赖 batch statistics，CV 经典。

## LayerNorm

对单样本 feature 维归一化，适合变长序列和 Transformer。

## RMSNorm

只使用 RMS，不做 mean centering；计算更简单，LLM 常见。

### 高频比较

- BN 为什么不适合 autoregressive Transformer？
- LN 为什么不依赖 batch size？
- RMSNorm 少了什么？会带来什么影响？

详见 01。

---

# 12. Activation Functions `S/A`

## ReLU

简单高效，但可能 dead neuron。

## GELU

Transformer/BERT 经典。

## SiLU/Swish

平滑 gating-like activation。

## SwiGLU

现代 LLM FFN 高频结构；本质不仅是 activation 替换，还改变 FFN gating 结构。

---

# 13. Gradient Problems `S`

## Vanishing / Exploding

与链式乘积、activation、初始化、sequence depth 等相关。

常用处理：

- residual；
- norm；
- careful initialization；
- gradient clipping；
- gating；
- proper optimizer。

### Gradient clipping 为什么常用于大模型/RL？

控制极端 update，尤其序列长、reward 高 variance、mixed precision 下异常 gradient。

---

# 14. Initialization `A`

- Xavier/Glorot：关注前后层方差；
- He/Kaiming：适合 ReLU 类；
- Transformer/LLM 还有 residual scaling、特殊 std 等模型特定设计。

**面试原则：** 不要背“某模型用 0.02”，要解释初始化目标是让信号/梯度尺度可控。

---

# 15. Metrics：Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC `S`

## Confusion Matrix

```text
TP FP
FN TN
```

## Precision

预测为正中有多少是真的。

## Recall

真实正例中找回多少。

## F1

precision 与 recall 的 harmonic mean。

## ROC-AUC

可理解为随机抽正负样本时，正样本得分高于负样本的概率。

### AUC 的陷阱

- 不依赖固定 threshold；
- class imbalance 极重时 PR-AUC 可能更有解释力；
- ranking/search 中 global AUC 不一定对应 per-query top-K quality；
- AUC 高不代表 calibration 好。

---

# 16. 数据泄漏 / 切分 `S`

高频工程题：为什么 validation 很高、上线很差？

先查：

- random split 是否引入未来信息；
- user/item/query 是否跨集合泄漏；
- duplicate/near duplicate；
- feature 是否使用 post-event data；
- label timestamp；
- preprocessing 是否 fit on all data。

对于推荐/搜索，**time split** 往往比随机切分更接近真实上线。

---

# 17. Class Imbalance `A`

方法：

- resampling；
- class weight / pos_weight；
- focal loss；
- threshold moving；
- calibrated probability；
- 分桶评估。

不要只看 accuracy。

---

# 18. Calibration `B/A`

模型输出 0.9 是否真的意味着 90% 概率？

常见：

- Platt scaling；
- isotonic regression；
- temperature scaling。

在风控、广告、ranking、reward model 等场景会有价值。

---

# 19. 2026 真实题目映射

## 小红书 2026-07-28

- AUC 怎么评估？
- MSE 收敛但 ranking 下降为什么？
- BCE + InfoNCE；
- soft label 能不能直接进 BCE？

**考点：** loss != metric、ranking vs classification、数据/标签/评估设计。

## 美团 2026

- K-Means 核心原理；
- K-Means 是否全局最优；
- 聚类算法。

## 字节多模态 2026-04

- 常见 optimizer；
- MHA；
- CLIP；
- ViT/Swin。

---

# 20. 高频等级

## S

- LR / BCE / CE
- Bias-Variance
- Overfit / regularization
- Optimizer / AdamW
- normalization
- AUC / Precision / Recall / F1
- data leakage
- activation

## A

- GBDT/XGBoost
- SVM
- K-Means/PCA
- class imbalance
- learning rate schedule
- initialization

## B/C

- calibration
- kernel details
- classic EM/GMM
- advanced boosting variants

---

# 21. 最小实验

1. 用 sklearn 在同一数据集比较 LR / tree / XGBoost；
2. 人为构造 imbalance，比较 Accuracy 与 PR-AUC；
3. 构造时间泄漏，观察 validation 虚高；
4. MSE-only ranking vs pairwise/InfoNCE，比较 NDCG；
5. SGD/Adam/AdamW 在小网络上比较收敛与 weight decay。

---

# 22. 自测

- [ ] 能从 maximum likelihood 推到 BCE
- [ ] 能解释 AUC，不只会背公式
- [ ] 能解释 XGBoost 为什么用二阶信息
- [ ] 能解释 K-Means 为什么不保证全局最优
- [ ] 能解释 AdamW 与 L2 的差异
- [ ] 能定位 loss 降但 metric 差的原因
- [ ] 能设计无 leakage 的 train/val/test split
