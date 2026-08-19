# 01｜Transformer 与 Attention：从数学原理到手写实现与 2026 大厂面试

> 目标：不是“背出 Attention 公式”，而是做到四件事：**能解释、能推导、能手写、能分析工程代价**。  
> 适用岗位：大模型算法、LLM/多模态算法、后训练、推理优化、Agent 算法、搜索推荐大模型。  
> 高频标记说明：基于 2026 年公开面经样本整理，不等同于全行业统计。

---

## 0. 这一章为什么必须学透

2026 年大厂对 Transformer 的考法已经不只是“介绍一下 Transformer”。公开面经中可以看到：

- **DeepSeek 大模型算法岗（2026-07-26）**：要求手写完整 Multi-Head Attention，不能只写四个线性层；还会继续追问张量形状、mask、数值稳定、工程实现。
- **美团大模型算法（2026-07）**：问自注意力、LayerNorm/RMSNorm、FFN、SwiGLU、MoE、GQA、LLM 全流程 Tensor shape；还有“为什么 scaled dot-product 要缩放”。
- **腾讯大模型实习（2026-02/03）**：问 Pre-Norm vs Post-Norm、Decoder-only 原因、Transformer 推理优化。
- **字节大模型算法（2026-03）**：问 self-attention、复杂度、为什么 multi-head，并沿着 Agent/推理系统继续问。
- **阿里国际 Agent 实习（2026-03）**：从 Attention 数学本质一直追到长序列复杂度和长上下文遗忘。

因此这一章要建立的是一条完整链路：

```text
Token
  ↓
Embedding
  ↓
Position / RoPE
  ↓
Q K V projection
  ↓
Scaled Dot-Product Attention
  ↓
Causal / Padding Mask
  ↓
Multi-Head / GQA / MQA
  ↓
Output projection + Residual
  ↓
Norm
  ↓
FFN / SwiGLU / MoE
  ↓
下一层
```

你必须能解释每一层：**输入形状是什么、做了什么、为什么需要、时间/显存成本是多少、训练和推理有什么不同。**

---

# Part A｜先把 Transformer 本体讲明白

## 1. Transformer 到底解决了什么问题？ ★★★★★

在 Transformer 之前，序列建模大量依赖 RNN/LSTM。RNN 的核心限制是：第 `t` 个 token 的计算依赖第 `t-1` 个 token 的隐藏状态，因此训练难以在序列维度充分并行；长距离依赖也容易随着递归传播变弱。

Transformer 的核心变化是：

1. 不再靠递归状态逐步传播信息；
2. 使用 Attention 让一个 token 直接读取序列中其他 token 的信息；
3. 训练时所有位置可以并行计算；
4. 位置顺序通过 positional information 单独注入。

**面试不要说**：“Transformer 的优点就是并行、效果好。”

更好的回答是：

> Transformer 把序列依赖建模从“沿时间步递归传播”改成“通过注意力直接构造 token-token 的信息路由”。这缩短了任意两个 token 之间的信息路径，并显著改善训练并行性；代价是标准全注意力的 token-token score matrix 随序列长度呈二次增长。

### 追问：Transformer 的短板是什么？

- 标准 attention 的计算量随上下文长度 `L` 近似 `O(L²·d)`；
- 训练时 attention matrix/中间激活带来较高显存压力；
- 自回归生成阶段仍然是串行 token decoding；
- 超长上下文并不意味着模型能稳定利用所有信息，存在 lost-in-the-middle / attention dilution 等问题；
- 推理阶段 KV Cache 会成为显存与带宽瓶颈。

这就自然引向后续章节：**FlashAttention、KV Cache、GQA、PagedAttention、Speculative Decoding、Context Governance**。

---

## 2. Encoder、Decoder、Decoder-only 为什么不同？ ★★★★☆

原始 Transformer 是 Encoder-Decoder：

- Encoder：双向 self-attention，可读取整个输入序列；
- Decoder：causal self-attention，只能读取当前位置之前的 token；
- Decoder 还通过 cross-attention 读取 Encoder 输出。

现代通用生成式 LLM 大量使用 Decoder-only。原因不是“Encoder 没用”，而是 Decoder-only 与 next-token prediction 的自回归训练目标天然一致，并能把理解、生成、条件控制统一到同一个 token 序列建模框架中。

### 面试追问：为什么不是 Encoder-Decoder？

建议从四个角度回答：

1. **训练目标统一**：next-token prediction 简单、规模化容易；
2. **架构统一**：prompt、上下文、输出都在同一序列；
3. **in-context learning**：通过统一的因果建模学习“前文条件 → 后续输出”；
4. **工程生态成熟**：KV Cache、continuous batching、prefix caching 等 serving 技术围绕 causal decoder 已高度优化。

但 Encoder-Decoder 对翻译、结构化输入输出、某些多模态场景仍有价值，不要回答成“Decoder-only 一定更强”。

---

# Part B｜Self-Attention 的数学本质

## 3. Q、K、V 到底是什么？ ★★★★★

设隐藏状态：

```text
X ∈ R[B, L, D]
```

其中：

- `B`：batch size
- `L`：sequence length
- `D`：hidden size / model dimension

线性投影：

```text
Q = X Wq
K = X Wk
V = X Wv
```

直觉可以这样理解：

- **Query**：当前位置“想找什么信息”；
- **Key**：每个位置“我有什么特征可供匹配”；
- **Value**：真正被汇聚的信息内容。

注意：这是理解类比，不要误以为 Q/K/V 在模型里有人工定义的语义。它们都是训练学习出的向量投影。

### Attention 公式

```text
Attention(Q,K,V) = softmax(QKᵀ / √d_k + mask) V
```

完整流程：

1. `QKᵀ`：计算 query 与所有 key 的相似程度；
2. `/√d_k`：控制点积方差，防止 softmax 过早饱和；
3. `+ mask`：禁止读取不可访问 token；
4. `softmax`：变成归一化权重；
5. `×V`：按权重聚合信息。

---

## 4. 为什么一定要除以 `√d_k`？ ★★★★★｜经典挖坑题

假设 Q/K 每个维度近似零均值、单位方差且相互独立，那么点积：

```text
q · k = Σ q_i k_i
```

其方差会随着维度 `d_k` 增长，量级约为 `d_k`。维度越大，score 的绝对值越容易变大，softmax 更容易进入非常尖锐的饱和区域。

除以 `√d_k` 后，score 的尺度被拉回比较稳定的范围。

### 为什么 softmax 饱和不好？

softmax 输入极大/极小时：

- 概率趋近 0/1；
- 大量位置梯度接近 0；
- 训练更难稳定。

### 面试坑：是“去掉 d_k 维度”吗？

不是。`1/√d_k` 是**缩放 score 的数值尺度**，并没有删掉任何维度。若面试官用模糊表述问“为什么 Attention 要去除/处理 d_k”，应主动澄清他指的是 scaled dot-product 的归一化因子。

---

## 5. Softmax 为什么要减 max？ ★★★★★

直接计算：

```python
exp(x)
```

当 `x` 很大时会 overflow。稳定实现通常使用：

```text
softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)
where m = max(x)
```

减去同一个常数不改变 softmax 结果，因为分子分母都会乘相同缩放因子。

### PyTorch 手写稳定 softmax

```python
import torch

def stable_softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 工程注意

真实框架会使用 fused kernel / mixed precision 策略，不应在生产里手写 Python softmax 替代优化 kernel。面试手写的目的，是证明你理解数值稳定性。

---

## 6. Causal Mask 与 Padding Mask ★★★★★

### Causal mask

自回归模型在训练时不能偷看未来 token。

对长度 4：

```text
位置0：可看 0
位置1：可看 0 1
位置2：可看 0 1 2
位置3：可看 0 1 2 3
```

通常对不可见位置加入一个极大的负数：

```text
score + (-∞)
```

经过 softmax 后概率约为 0。

### Padding mask

batch 内序列长度不同，为了对齐会 padding。Padding token 不应参与 attention，因此同样要 mask。

### 高频陷阱

**Mask 在 softmax 前还是后？**

标准做法是 softmax 前对 logits 加 `-inf`（或数值类型允许的足够小值）。如果 softmax 后再简单乘 0，需要重新归一化，否则权重和不再为 1。

---

# Part C｜Multi-Head Attention：为什么要“多头”

## 7. MHA 的完整 Tensor Shape ★★★★★

输入：

```text
X: [B, L, D]
```

假设 `H` 个头，`Dh = D / H`。

投影后：

```text
Q,K,V: [B, L, D]
```

reshape + transpose：

```text
[B, L, H, Dh]
→ [B, H, L, Dh]
```

score：

```text
Q @ K.transpose(-2,-1)
[B,H,L,Dh] @ [B,H,Dh,L]
→ [B,H,L,L]
```

attention output：

```text
[B,H,L,L] @ [B,H,L,Dh]
→ [B,H,L,Dh]
```

拼回：

```text
[B,H,L,Dh]
→ [B,L,H,Dh]
→ [B,L,D]
```

最后 output projection：

```text
Y = concat(heads) Wo
```

### 为什么面试特别爱问 shape？

因为只会背公式的人，很容易在 `transpose`、mask broadcast、head 维、causal 维度上写错。能完整讲 shape 才说明你真的能实现。

---

## 8. 为什么 Multi-Head 比 Single-Head 有意义？ ★★★★☆

多头允许模型在不同的学习子空间中形成不同的信息路由模式，例如部分头可能更偏局部关系、位置关系、实体指代或某类语法/语义模式。

更严谨的说法是：

> 多头把总 hidden dimension 划分成多个 head 子空间，各头拥有独立投影，使模型能够并行学习多组 query-key 相似性与 value 聚合方式，最后再融合。

### 面试陷阱

“多头一定能学出可解释的固定功能吗？”

不能保证。可视化能观察到一些模式，但不应把“第 3 个头一定负责语法”当作架构定义。

---

## 9. PyTorch 手写完整 MHA ★★★★★

下面这版故意不用 `nn.MultiheadAttention`，因为面试官要看的是底层：

```python
import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        # [B, L, D] -> [B, H, L, Dh]
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, attention_mask=None, causal=True):
        B, L, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            # 假设 attention_mask: [B, L], 1=有效, 0=padding
            key_mask = ~attention_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(key_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        # [B,H,L,Dh] -> [B,L,D]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)
```

### 面试官可能继续挖什么？

1. 为什么 `contiguous()`？
2. mask 如何兼容 left padding？
3. FP16 下 `-inf` 与 softmax 的数值问题？
4. dropout 放在哪里？
5. cross-attention 如何修改？
6. 如何加入 KV Cache？
7. 如何把 MHA 改成 GQA？
8. 为什么生产环境不这么写？——因为需要 fused attention kernel、FlashAttention/SDPA、kernel-level 优化。

---

# Part D｜MHA、MQA、GQA 与 KV Cache

## 10. MHA / MQA / GQA 区别 ★★★★★

### MHA

每个 query head 都有自己的 K/V head。

```text
Q heads: H
K heads: H
V heads: H
```

### MQA

所有 Q heads 共用一组 K/V：

```text
Q heads: H
K heads: 1
V heads: 1
```

优点：大幅降低 KV Cache；缺点：可能损失部分质量/表达能力。

### GQA

折中方案：多个 query head 共享一个 KV group。

```text
Q heads: H
KV heads: G, 1 < G < H
```

GQA 在质量和推理成本之间折中，因此现代模型非常常见。

### 为什么和推理强相关？

生成第 t 个 token 时，以前 token 的 K/V 不需要重新计算，可以缓存。KV Cache 大致随：

```text
layers × sequence_length × kv_heads × head_dim × 2(K,V) × bytes_per_element
```

线性增长。

把 `kv_heads` 从 `H` 降到 `G`，就会直接降低 KV Cache 大小和内存带宽压力。

---

## 11. KV Cache 为什么只缓存 K/V，不缓存 Q？ ★★★★★

每一步生成新 token 时：

- 历史 token 的 K/V 会被当前新 query 再次读取；
- 历史 query 不需要参与未来 token 的 attention 计算。

因此缓存 K/V 有复用价值，而历史 Q 没有。

### 高频追问：KV Cache 是“减少 Attention 的 O(L²)”吗？

需要分训练/推理回答：

- **训练整段序列**：通常仍计算完整 attention；
- **自回归单步 decode**：有 cache 后不再为历史 token 重算 K/V，新 token 只与历史 K/V 做 attention；单步计算随当前上下文长度增长。

KV Cache 的代价是显存：上下文越长、batch 越大，缓存越大。

---

# Part E｜位置编码：为什么 RoPE 高频

## 12. 没有位置编码会怎样？ ★★★★☆

纯 self-attention 对输入 token 的排列本身缺乏顺序感知。若不注入位置，模型无法可靠地区分：

```text
A loves B
B loves A
```

仅靠 token 集合是不够的。

原始 Transformer 使用 sinusoidal positional encoding；后续 LLM 常用 RoPE。

---

## 13. RoPE 的核心思想 ★★★★☆

RoPE 不只是“把 position embedding 加到 token 上”，而是对 Q/K 的二维通道对做与位置相关的旋转。

核心性质：

> 对位置 `m` 的 Q 和位置 `n` 的 K 做旋转后，两者内积可以自然携带 `m-n` 的相对位置信息。

面试回答建议分三层：

1. **直觉**：位置决定旋转角度；
2. **数学**：旋转矩阵作用于 Q/K，点积与相对位置有关；
3. **工程**：RoPE 与 causal decoder、长上下文扩展策略结合广泛，但超出训练长度时通常需要额外 scaling/插值策略，不能简单理解为“无限长度外推”。

### 追问

- 为什么只作用 Q/K，通常不作用 V？
- RoPE 与 absolute position embedding 差别？
- long context 时 RoPE scaling 如何处理？
- 多模态模型如何扩展到 2D/3D position？

---

# Part F｜Norm、Residual、FFN

## 14. LayerNorm vs RMSNorm ★★★★★

### LayerNorm

对单个 token 的 hidden features 做：

```text
x -> (x - mean) / sqrt(var + eps) -> affine
```

核心包含：

- re-centering（减均值）
- re-scaling（按标准差归一）

### RMSNorm

RMSNorm 省略均值中心化，只按 root mean square 做 rescale：

```text
x / sqrt(mean(x²) + eps)
```

再乘可学习 scale。

### 为什么现代 LLM 常见 RMSNorm？

- 结构更简单；
- 计算更省；
- 在大量 LLM 架构中实践稳定。

面试不要说“RMSNorm 一定比 LayerNorm 好”，更准确是：它去掉 re-centering，保留 rescaling，简化计算，并在很多架构中取得良好训练效果。

---

## 15. Pre-Norm vs Post-Norm ★★★★☆

### Post-Norm（原始 Transformer 风格）

```text
x -> sublayer -> residual add -> norm
```

### Pre-Norm

```text
x -> norm -> sublayer -> residual add
```

Pre-Norm 的残差主干更像一条直接的 identity path，通常更利于深层网络梯度传播和训练稳定，因此现代大模型大量采用 pre-norm 风格。

### 追问

- Pre-Norm 为什么更稳定？
- Post-Norm 是否可能有更强表达/最终性能？
- DeepNorm、sandwich norm 是解决什么问题？

---

## 16. FFN 在 Transformer 里干什么？ ★★★★☆

Attention 主要负责**token 间信息混合**；FFN 对每个 token 独立进行**通道维非线性变换**。

原始 FFN：

```text
FFN(x) = W2 σ(W1 x)
```

常见 hidden/intermediate dimension 比 model dimension 更大。

现代 LLM 常见 gated FFN，如 SwiGLU：

```text
SwiGLU(x) ≈ (SiLU(xW_gate) ⊙ xW_up) W_down
```

### 为什么需要 gate？

它让网络通过乘法门控动态调节特征通道，并提供更丰富的非线性组合。

### 面试追问

- GELU vs SiLU？
- 为什么 FFN 参数往往占模型很大比例？
- SwiGLU 为什么有三组 projection？
- Qwen/Llama FFN intermediate size 为什么不是简单 `4D`？

---

# Part G｜MoE 基础（Transformer 结构中的高频延伸）

## 17. MoE 是什么？ ★★★★☆

Dense FFN：每个 token 都执行同一套 FFN 参数。

MoE：准备多个 expert，router 对每个 token 选择少数 top-k expert：

```text
Token
  ↓
Router
  ↓
Top-k Experts
  ↓
Weighted combine
```

核心目标是：**总参数规模可以很大，但每个 token 只激活部分参数，从而控制实际计算量。**

### 工程难点

- expert load balance；
- token dispatch/all-to-all 通信；
- expert parallel；
- capacity / overflow；
- router instability；
- 推理时跨设备通信。

这也是为什么面试问 MoE 时，不能只回答“稀疏激活，参数大计算少”。

---

# Part H｜复杂度与显存：面试一定要能算

## 18. Attention 复杂度怎么分析？ ★★★★★

设序列长度 `L`、hidden dimension `D`。

Q/K/V projection 约：

```text
O(L D²)
```

attention score 和对 V 聚合约：

```text
O(L² D)
```

因此不能简单说“Transformer 永远是 O(L²)”；更准确是 attention token mixing 部分有 `L²` 项，而完整 block 还有线性于 `L`、二次于 `D` 的 projection/FFN 计算。

在短序列、大 hidden 情况下，projection/FFN 也可能非常重；在长上下文时 `L²` attention 更突出。

---

## 19. FlashAttention 是不是把计算复杂度变成 O(L)？ ★★★★★｜大坑

**不是。**

FlashAttention 的关键是 IO-aware：通过 tiling、online softmax、recomputation 等方式减少 HBM ↔ SRAM 的读写和避免物化完整 `L×L` attention matrix，从而显著降低显存占用并提升 wall-clock speed。

它仍然计算 exact attention，并没有把标准 dense attention 的数学 FLOPs 神奇地变成线性。

这道题在推理优化章节会进一步展开。

---

# Part I｜真实 2026 大厂面试拆解

## 20. DeepSeek｜2026-07-26｜手写完整 MHA ★★★★★

### 面试题

> 手写完整 Multi-Head Attention，不能只写框架。

### 为什么会这么考？

DeepSeek 这类大模型算法岗位需要区分：

- 会不会调用 `nn.MultiheadAttention`；
- 是否理解 attention kernel 之前的数学/shape；
- 能不能处理 mask、dtype、数值稳定；
- 后续是否有能力理解 GQA、MLA、KV Cache、FlashAttention、PagedAttention。

### 你应该达到的标准

必须能不看资料写出：

1. Q/K/V projection；
2. reshape/split heads；
3. transpose；
4. scaled dot-product；
5. causal + padding mask；
6. softmax；
7. weighted sum；
8. concat heads；
9. output projection；
10. 解释所有 shape。

### 进阶追问

- 如果改成 GQA，K/V reshape 怎么变？
- decode 阶段 KV Cache 如何接入？
- 为什么 FlashAttention 不需要存完整 attention matrix？
- BF16/FP16 下如何避免 NaN？

---

## 21. 美团｜2026-07｜Transformer 核心链路 ★★★★★

公开面经出现过一整组：

- self-attention 原理；
- LayerNorm / RMSNorm；
- FFN；
- SwiGLU；
- MoE；
- GQA；
- Qwen3 / Qwen3-Next；
- 输入维度到完整推理 Tensor shape；
- scaled dot-product 的缩放原因。

### 为什么这组题很重要？

它说明面试官不是随机抽八股，而是在验证你能否从一个 Transformer block 一路讲到现代 LLM 架构。

### 推荐回答方式

不要逐题孤立背诵。建议画出：

```text
Embedding + RoPE
      ↓
RMSNorm
      ↓
GQA Attention
      ↓
Residual
      ↓
RMSNorm
      ↓
SwiGLU / MoE FFN
      ↓
Residual
```

然后对每个模块回答：

**是什么 → 为什么 → 数学/shape → 现代模型为什么这样选 → 训练/推理成本。**

---

## 22. 腾讯｜2026-02/03｜Pre-Norm、Decoder-only 与推理优化 ★★★★☆

### 高频链路

1. Pre-Norm vs Post-Norm；
2. 为什么生成模型常用 Decoder-only；
3. Transformer 推理有哪些优化；
4. vLLM 为什么更适合 serving。

这是一条“架构 → 训练稳定 → 推理系统”的连续追问。

### 答题策略

- Pre/Post-Norm：讲 residual gradient path；
- Decoder-only：讲 causal LM objective 和统一序列建模；
- 推理优化：KV Cache、GQA/MQA、quantization、FlashAttention、continuous batching、prefix caching、speculative decoding；
- 不要把这些技术混为一类：有的是**模型结构优化**，有的是**kernel 优化**，有的是**cache/memory management**，有的是**scheduler/serving optimization**。

---

# Part J｜最常见的“挖坑题”

## 23. 你应该能立即识别的错误说法

### 坑 1：Attention 除 `√d_k` 是为了降低计算复杂度

错。主要是数值尺度和 softmax 稳定。

### 坑 2：FlashAttention 把 O(L²) 变成 O(L)

错。它主要优化 IO 与中间内存，不是把 dense attention 的数学运算变成线性。

### 坑 3：KV Cache 缓存 Q/K/V

通常缓存 K/V；历史 Q 不需要复用。

### 坑 4：GQA 是为了训练更快

它的重要工程价值主要体现在减少 KV head、降低推理 KV Cache 与带宽压力；训练也会受结构影响，但面试重点常是 serving。

### 坑 5：RMSNorm = LayerNorm 去掉 bias

错。关键差别是是否做 mean centering。

### 坑 6：Decoder-only 没有“理解能力”

错。架构形式不等于只能生成；自回归训练可以学习丰富表示与条件推理能力。

### 坑 7：上下文窗口变大就等于模型能完全记住所有信息

错。容量上限、有效利用、attention 分配与任务结构是不同问题。

---

# Part K｜面试前必须完成的代码练习

## Lab 1：纯 PyTorch MHA

要求：

- 不使用 `nn.MultiheadAttention`；
- 支持 causal mask；
- 支持 padding mask；
- 打印每个 tensor shape；
- 用官方 SDPA 输出做数值对比。

## Lab 2：MHA → GQA

要求：

- `num_q_heads=8`，`num_kv_heads=2`；
- 明确 K/V 如何 repeat/broadcast 到 query groups；
- 计算理论 KV Cache 缩减比例。

## Lab 3：加入 KV Cache

要求：

- prefill 一次；
- decode 每次输入 1 token；
- 保存历史 K/V；
- 验证 cached decode 与 full recompute 的最后一个 token logits 接近。

## Lab 4：数值稳定

分别用 FP16 / BF16 / FP32：

- 构造大 logits；
- 测试 naive softmax；
- 测试 stable softmax；
- 观察 inf / NaN。

## Lab 5：性能实验

比较：

- 手写 attention；
- `torch.nn.functional.scaled_dot_product_attention`；
- 可用环境下 FlashAttention backend；
- 不同 `L` 的耗时与 peak memory。

---

# Part L｜本章题库：按频率学习

## S 级：必须会（★★★★★）

1. Transformer block 完整结构。
2. Self-Attention 公式与 Q/K/V 含义。
3. 为什么 `/√d_k`。
4. causal mask 与 padding mask。
5. softmax 数值稳定。
6. MHA 完整 shape。
7. 手写 MHA。
8. Attention 时间/空间复杂度。
9. KV Cache 原理。
10. MHA/MQA/GQA。
11. LayerNorm vs RMSNorm。
12. Pre-Norm vs Post-Norm。
13. Decoder-only 为什么适合 LLM。

## A 级：高频（★★★★☆）

14. RoPE 原理。
15. SwiGLU / gated FFN。
16. MoE / Router / Top-k。
17. long context 的主要困难。
18. FlashAttention 与普通 attention 的区别。
19. cross-attention vs self-attention。
20. 为什么 FFN 也很耗参数/计算。

## B 级：中频（★★★☆☆）

21. absolute / sinusoidal / relative / RoPE 对比。
22. attention dropout 的位置。
23. head dimension 为什么通常固定在某些尺度。
24. RMSNorm epsilon 的作用。
25. tie embedding / LM head。
26. weight tying 的利弊。
27. logit soft-cap / attention sink 等新结构设计如何理解。

## C 级：低频但容易区分候选人（★★☆☆☆）

28. online softmax 的推导思路。
29. FlashAttention tiling 的 IO 视角。
30. Tensor Core / HBM / SRAM 为什么影响 attention 实现。
31. GQA checkpoint 如何由 MHA uptrain。
32. 长上下文位置编码外推为什么会失败。

---

# Part M｜权威资料（用于验证答案）

1. Vaswani et al., **Attention Is All You Need** — 原始 Transformer。
2. Su et al., **RoFormer: Enhanced Transformer with Rotary Position Embedding** — RoPE。
3. Zhang & Sennrich, **Root Mean Square Layer Normalization** — RMSNorm。
4. Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — GQA。
5. Dao et al., **FlashAttention** / **FlashAttention-2** — IO-aware exact attention。
6. vLLM / PagedAttention 论文与官方文档 — 推理侧 KV Cache 管理。

真实面试来源详见：[`09-2026-real-interviews.md`](./09-2026-real-interviews.md)。

---

# 本章验收标准

如果下面任意一项做不到，本章还不能算学完：

- [ ] 3 分钟白板画出 Decoder-only Transformer block。
- [ ] 不查资料推导并解释 scaled dot-product attention。
- [ ] 不查资料写出完整 MHA，并说清每个 shape。
- [ ] 清楚区分 causal mask / padding mask。
- [ ] 解释 LayerNorm/RMSNorm、Pre/Post-Norm。
- [ ] 解释 MHA/MQA/GQA 与 KV Cache 的关系。
- [ ] 解释为什么 FlashAttention 不是“线性 Attention”。
- [ ] 能回答美团/腾讯/字节/DeepSeek 上述真实追问。
- [ ] 能把理论答案落到 PyTorch 代码和 serving 工程上。
