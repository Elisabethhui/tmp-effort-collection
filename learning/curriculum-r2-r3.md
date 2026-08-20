# R2–R3 课程地图与基础知识包

## 0. 学习单元契约

每个单元都要留下：

1. 一页概念卡：What / Why / Mechanism / Trade-off；
2. 一个 CPU/tiny Lab 或源码阅读图；
3. 至少一个失败样例；
4. 30 秒、3 分钟、15 分钟回答；
5. 通过测试后才能标为 `verified`。

## 1. R2：算法、数学、ML/DL 和 PyTorch 基础

### R2.1 Python / 数据结构 / 复杂度

**必须掌握**

- array、hash map/set、stack/queue、heap/top-k、linked list、tree、graph；
- 双指针、滑动窗口、二分、BFS/DFS、排序、递归、动态规划；
- 不变量、时间复杂度、空间复杂度、空输入/重复值/越界测试。

**最小代码**

- top-k with heap；
- LRU cache 的哈希表 + 双向链表思路；
- 无权图最短路 BFS；
- stable merge/sort 或二分边界。

**面试检查**

- 为什么 heap top-k 是 `O(n log k)`？
- BFS 什么时候比 DFS 合适？
- Python list/dict 的常见复杂度和迭代器边界是什么？

### R2.2 线性代数、概率与数值稳定

**核心公式**

```text
X @ W:       [B, T, D] @ [D, D'] -> [B, T, D']
softmax(z)i: exp(zi - max(z)) / Σj exp(zj - max(z))
CE:          -log pθ(y | x)
KL(P||Q):    Σx P(x) [log P(x) - log Q(x)]
```

**必须解释**

- transpose、broadcast、归一化维度和 dtype；
- 为什么 softmax 要减最大值；
- 为什么 logits 直接接 cross entropy，不先手动 softmax；
- expectation、variance、概率分布和采样的关系；
- float16/bfloat16/float32 的范围和稳定性风险。

**最小代码**

- stable softmax；
- logits 版 cross entropy；
- log-prob 版 KL；
- 用有限差分检查一个标量函数的梯度。

### R2.3 梯度、优化器与训练诊断

**必须掌握**

- chain rule、反向传播、梯度归零、参数更新；
- SGD、Momentum、Adam、AdamW、weight decay；
- learning rate、warmup、gradient clipping、vanishing/exploding gradient；
- train/eval、dropout、checkpoint、seed、数据泄漏。

**诊断顺序**

```text
shape/dtype/device → loss 是否可计算 → labels 是否错位
→ 梯度是否为 None/NaN → optimizer 是否更新 → train/eval 是否正确
→ 数据/学习率/初始化 → 才考虑模型结构
```

### R2.4 ML/DL 高频基础与指标

**必须掌握**

- bias/variance、overfit、正则化、early stopping；
- LR、GBDT/XGBoost、SVM、K-Means/PCA 的核心目标和适用边界；
- Accuracy、Precision、Recall、F1、ROC-AUC、PR-AUC、NDCG/MRR；
- class imbalance、calibration、train/validation/test split、数据泄漏。

**面试输出**

- 能解释“为什么离线指标变好，线上不一定变好”；
- 能根据业务代价选择 Precision/Recall/F1/AUC/NDCG；
- 能指出一个错误切分导致的 leakage。

### R2.5 PyTorch 基础与设备抽象

**必须掌握**

- tensor shape、stride、view/reshape/transpose/contiguous；
- `nn.Module`、forward、autograd、optimizer、`train()`/`eval()`；
- CPU、MPS、CUDA 的 device contract；
- checkpoint 中的 model/optimizer/step/seed；
- CPU reference 和 GPU performance 是两类不同证据。

**Mac 验收**

- CPU 上所有 correctness tests 通过；
- 如果 MPS 可用，再做相同输入输出 smoke；
- 不以 MPS/CUDA 速度作为 R2–R3 的通过条件。

## 2. R3：Transformer 主干

### R3.1 Tokenizer、Embedding 与 Decoder-only

**数据流**

```text
text → tokenizer → input_ids [B,T]
     → embedding [B,T,D]
     → N 个 decoder block
     → final norm [B,T,D]
     → lm_head [B,T,V]
     → next-token logits
```

**必须解释**

- tokenization 为什么影响序列长度和训练成本；
- embedding 与 lm_head 为什么可以 weight tying；
- decoder-only 为什么用 causal mask；
- encoder-only、decoder-only、encoder-decoder 的输入输出差异。

### R3.2 Q/K/V 与 scaled dot-product attention

对 `X ∈ R[B,T,D]`：

```text
Q = X WQ, K = X WK, V = X WV
scores = Q Kᵀ / √Dh
A = softmax(scores + mask)
O = A V
```

单头 shape：`Q/K/V = [B,T,Dh]`，多头内部通常是 `[B,H,T,Dh]`。

**必须掌握**

- 为什么 Q/K/V 是三组投影；
- 为什么除以 `√Dh`；
- 为什么 mask 在 softmax 前加；
- 为什么 causal attention 的分数矩阵是三角形；
- attention 的计算和显存复杂度。

### R3.3 MHA、MQA、GQA 与 shape

```text
MHA: Q heads = K heads = V heads = H
MQA: Q heads = H, K/V heads = 1
GQA: Q heads = H, K/V heads = Hkv, Hkv < H
```

**必须掌握**

- `head_dim = D / H`；
- GQA 如何 repeat/interleave K/V；
- MQA/GQA 为什么减少推理 KV Cache；
- GQA 不等于“训练一定更快”；
- key padding mask 和 query padding mask 的区别。

### R3.4 位置编码与 RoPE

**对比**

- learned absolute embedding：简单但受最大长度限制；
- sinusoidal：固定、可外推但表达能力有限；
- RoPE：对 Q/K 做位置相关旋转，使注意力内含相对位置信息。

**必须会写**

- 频率向量、sin/cos、偶奇维度旋转；
- position 0 不改变向量；
- RoPE 作用在 Q/K，不直接加到 V；
- 长上下文 scaling 不是免费扩大上下文。

### R3.5 Residual、Pre-Norm、RMSNorm、FFN、SwiGLU

**Pre-Norm block**

```text
h = x + Attention(Norm(x))
y = h + FFN(Norm(h))
```

**RMSNorm**

```text
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

**SwiGLU**

```text
SwiGLU(x) = SiLU(x Wgate) ⊙ (x Wvalue)
             → Wdown
```

**必须解释**

- residual 为什么帮助梯度和信息流；
- Pre-Norm 与 Post-Norm 的稳定性和表达差异；
- RMSNorm 与 LayerNorm 的计算差异；
- FFN 为什么是 token-wise 的容量扩展；
- gate 为什么比单一激活更灵活。

### R3.6 Causal LM 与训练目标

```text
input:  [B, t0, t1, ..., t(T-2)]
target: [t1,  t2, ..., t(T-1)]
```

**必须掌握**

- logits `[B,T,V]` 与 labels `[B,T]`；
- shifted cross entropy；
- prompt/padding 的 `ignore_index`；
- teacher forcing；
- 训练时为什么通常不用 KV Cache。

### R3.7 KV Cache、Prefill/Decode 与源码阅读

**机制**

- prefill：一次处理 prompt，生成每层历史 K/V；
- decode：每步只计算新 token 的 Q/K/V，复用历史 K/V；
- cache 按 layer 保存，典型形状 `[B,Hkv,T,Dh]`；
- memory 随序列长度增长，系统实现还要处理 block、碎片、prefix reuse 和 eviction。

**源码阅读顺序**

1. Transformers Cache interface；
2. Dynamic/Static layer 的 update/append；
3. attention mask 与 past length；
4. vLLM BlockPool 的 block/ref-count/free queue；
5. KVCacheManager 的 allocate/free/prefix-cache；
6. 回到显存公式和失败场景。

### R3.8 SDPA、FlashAttention、PagedAttention 的边界

**不能混淆**

- SDPA 是算子/API 语义；
- FlashAttention 是 IO-aware/tiled attention 实现；
- PagedAttention 是 KV Cache 分块/管理思路；
- 它们可以组合，但不是同一个东西；
- FlashAttention 不会把数学复杂度自动变成线性。

### R3.9 MoE、长上下文与量化（只做基础认识）

**只要求能解释**

- router、top-k experts、load balancing；
- active parameters 与 total parameters；
- 长上下文的计算、KV memory、位置外推风险；
- weight/activation/KV quantization 的对象和误差边界。

## 3. R2–R3 两周执行顺序

| 天 | 学习单元 | 必做产出 |
|---:|---|---|
| 1 | R2.1 Python/DSA | 2 道题 + 复杂度口述 |
| 2 | R2.2 线代/softmax/CE/KL | stable math Lab |
| 3 | R2.3 梯度/SGD/Adam | 手算一次更新 + 诊断卡 |
| 4 | R2.4 ML/DL/metrics | 指标选择题 + leakage 反例 |
| 5 | R2.5 PyTorch/device | tiny train loop/checkpoint |
| 6 | R3.1 Token/Embedding/LM | `[B,T]→[B,T,V]` shape 图 |
| 7 | R3.2 QKV/attention | 手推一个小矩阵 + MHA test |
| 8 | R3.3 masks/MHA/GQA | causal/padding/GQA test |
| 9 | R3.4 RoPE | position 0/旋转 test |
| 10 | R3.5 Norm/FFN/SwiGLU | Transformer block forward |
| 11 | R3.6 causal loss | label shift/ignore index test |
| 12 | R3.7 KV Cache | CPU cache toy + 显存账本 |
| 13 | R3.8 attention backends | 源码调用链图 |
| 14 | R3.9/R3.10 综合 | 10 道题 + 30/3/15 分钟回答 |

## 4. R2–R3 通过标准

- **基础**：能写代码、说复杂度、解释边界，不靠模板答案；
- **Transformer**：能从 token 追到 logits，并写出每个中间 shape；
- **数值**：能指出 softmax、CE、KL、fp16 的稳定性风险；
- **GPU 主题**：能读源码、画调用链、写 CPU reference，明确未验证的性能边界；
- **面试**：每个 S 级知识点有 30 秒结论、3 分钟机制、15 分钟追问树。
