# 03｜LLM 推理与 Serving：KV Cache、FlashAttention、PagedAttention、Continuous Batching、Prefix Cache、Speculative Decoding

> 目标：能够从“一次请求如何被服务器处理”解释推理系统，而不是只背 vLLM 的三个关键词。

---

## 0. 2026 为什么推理系统越来越高频

真实面经中：

- DeepSeek 2026-07-26：PagedAttention 原理、Speculative Decoding 工程实现。
- 腾讯 2026：Transformer 推理优化、vLLM 的动态 batching 与显存管理。
- 美团：vLLM prefix cache、GRPO rollout 显存、训练/推理硬件。
- 小红书：vLLM FP16 pooling 出现 overflow/NaN 如何排查。

所以要把 serving 拆成四层：

```text
模型算法层：GQA / Quantization / Spec Decode
Kernel 层：FlashAttention / fused kernels
Memory 层：KV Cache / Paged KV / Prefix Cache
Scheduler 层：Continuous Batching / Prefill-Decode scheduling
```

---

# Part A｜先理解 Prefill 与 Decode

## 1. Prefill 和 Decode 的区别 ★★★★★

### Prefill

一次处理整个 prompt，生成每层历史 token 的 K/V，并得到最后位置 logits。

特点：

- 大矩阵运算多；
- GPU compute 利用率通常较高；
- prompt 越长成本越高。

### Decode

每次输入一个新 token，使用历史 KV Cache，只生成下一 token。

特点：

- 自回归串行；
- 每一步要读取大量模型权重与 KV；
- 容易 memory-bandwidth bound；
- 大量并发请求需要 scheduler 动态拼 batch。

这一区分是理解 TTFT 和 TPOT 的基础。

---

## 2. TTFT、TPOT、ITL、Throughput、P99 ★★★★★

- **TTFT**：time to first token，从请求进入到第一个输出 token；受 queue + prefill 很大影响。
- **TPOT**：time per output token，decode 速度指标。
- **ITL**：inter-token latency，连续 token 间延迟。
- **Throughput**：单位时间处理请求或 token 数。
- **P50/P95/P99**：尾延迟。

面试不要只说“吞吐越高越好”。线上系统通常要满足 SLO：吞吐、TTFT、TPOT、P99、成本之间有 trade-off。

---

# Part B｜KV Cache

## 3. KV Cache 为什么必要 ★★★★★

如果每生成一个 token 都重新对完整 prefix 计算每层 K/V，会重复大量工作。

缓存每层历史 K/V 后：

```text
new Q 读取 old K/V + new K/V
```

历史 K/V 不再重算。

### KV Cache 大小如何估

大致：

```text
2 × layers × seq_len × kv_heads × head_dim × bytes × batch
```

其中 2 是 K 和 V。

### 为什么 GQA 会直接影响 serving？

因为 KV head 减少，cache 和每步 memory read 都下降。

---

# Part C｜FlashAttention

## 4. 普通 attention 为什么内存重 ★★★★★

naive 实现会显式生成：

```text
S = QKᵀ  -> [B,H,L,L]
P = softmax(S)
```

长序列时 `L×L` 中间矩阵巨大，而且大量 HBM 读写成为瓶颈。

## 5. FlashAttention 核心：IO-aware exact attention ★★★★★

核心技术：

- tiling；
- 把 Q/K/V block 搬到片上 SRAM；
- online softmax；
- 不把完整 attention matrix 写回 HBM；
- backward 时通过 recomputation 减少保存中间状态。

### 最大陷阱

**FlashAttention 不是近似 Attention，也不是把 dense attention FLOPs 变成 O(L)。**

它主要改变计算顺序和 memory IO，得到同样的 exact attention 结果（在数值误差范围内）。

## 6. FlashAttention 1/2/3 怎么答

不建议背版本 changelog。抓主线：

- FA1：IO-aware tiling + online softmax；
- FA2：更好的 work partition、降低非 matmul FLOPs、提高 occupancy；
- FA3：针对 Hopper 的异步能力/TMA/Tensor Core 与低精度进一步优化。

如果岗位不是 CUDA/kernel，不需要背 warp 级实现；但推理优化岗位要深入。

---

# Part D｜PagedAttention / vLLM

## 7. PagedAttention 解决的不是“Attention 数学问题” ★★★★★

自回归请求长度动态变化。如果给每个请求预留最大长度连续 KV 内存，会造成：

- internal fragmentation；
- external fragmentation；
- 动态扩容困难；
- beam/shared prefix 重复 KV。

PagedAttention 借鉴虚拟内存 paging：

```text
logical token blocks
       ↓ block table
physical KV blocks（可不连续）
```

请求只按需申请物理 block。

### 面试必须说出的四点

1. fixed-size block；
2. logical-to-physical mapping；
3. on-demand allocation / free；
4. prefix/beam 可共享 block，必要时 copy-on-write。

### 大坑

PagedAttention 不是“把 attention complexity 降低”。它核心是 KV cache memory management 与高并发服务。

---

## 8. vLLM 为什么吞吐高？ ★★★★★

不能只回答“因为 PagedAttention”。完整答案包括：

- efficient KV cache management；
- dynamic/continuous batching scheduler；
- optimized model execution/kernels；
- prefix caching；
- speculative decoding 等可选能力；
- quantized KV / model quantization 等工程能力。

当前 vLLM 实现已经不断演化，早期论文的 kernel 细节不应被误认为今天代码完全一致；官方文档也明确把部分旧 PagedAttention 文档标为历史实现。

---

# Part E｜Continuous Batching

## 9. Static batching 为什么浪费 ★★★★☆

假设一个 batch 有三个请求，输出长度不同：

```text
A: 10 tokens
B: 200 tokens
C: 50 tokens
```

static batch 可能要等长请求结束，短请求完成后 slot 空闲。

continuous batching 会在 decode step 间动态：

- 移除结束请求；
- 插入新请求；
- 重组 batch；
- 在 latency/throughput 间调度。

### 工程追问

- prefill 与 decode 是否混批？
- chunked prefill 为什么有用？
- 长 prompt 会不会阻塞 decode 请求？
- scheduler 如何保障 tail latency？

---

# Part F｜Prefix Caching

## 10. Prefix Cache 与 KV Cache 不是一回事 ★★★★★

- KV Cache：单个生成请求内部复用历史 K/V。
- Prefix Cache：跨请求复用**相同 prompt prefix** 已计算的 KV block。

适合：

- 多轮对话，共享长 chat history；
- 多用户共享大 system prompt；
- 同一长文档被反复问不同问题；
- RL rollout 中大量 shared prompt。

### Prefix cache 不会加速什么？

它只节省 shared prefix 的 prefill，不会直接加速之后的新 token decode。

### 2026 新的工程安全点

官方 vLLM 文档特别讨论了多租户 prefix cache 的 hash collision / cache isolation，并支持 request-level `cache_salt`。说明面试“prefix cache 怎么实现”已经可以继续追到安全隔离。

---

# Part G｜Speculative Decoding

## 11. 为什么自回归 decode 慢？ ★★★★★

标准生成：

```text
token1 -> target model
 token2 -> target model
  token3 -> target model
```

每个 token 依赖前一个，目标模型串行前向。

Speculative Decoding 使用便宜 draft proposer 先猜多 token，目标模型一次并行验证。

```text
draft:  a b c d
          ↓
target verifies positions together
          ↓
accept prefix / reject + correction
```

### 关键点

如果算法设计正确，可以保持目标模型原分布，而不是“用小模型近似替代大模型输出”。

### 工程难点

- draft/target tokenizer compatibility；
- acceptance rate；
- draft 成本；
- paged KV 的提交/回滚；
- continuous batching 下每请求接受长度不同；
- quantization / TP 下同步；
- acceptance 太低时自动退化。

### 如何判断值不值得？

不要只看“每轮猜 K 个 token”。要看：

```text
节省的 target serial steps
vs
额外 draft + verify + scheduling + KV management cost
```

最终以端到端 TPOT/ITL/吞吐/P99 测量。

---

# Part H｜Quantization

## 12. 权重量化与 KV Cache 量化 ★★★★☆

### Weight quantization

目标：减少权重显存和 memory bandwidth。

常见：INT8/INT4/FP8 等，不同硬件和 kernel 支持不同。

### KV Cache quantization

目标：减少长上下文/大 batch 时 KV cache 显存。

代价：

- quant/dequant 开销；
- scale metadata；
- accuracy/numerical error；
- kernel 支持。

面试不要把“模型 4bit”与“KV 4bit”混为一谈。

---

# Part I｜推理系统的常见瓶颈诊断

## 13. TTFT 很高怎么办？

排查：

- queue time；
- prompt length；
- prefill batching；
- prefix cache hit；
- tokenizer/CPU preprocess；
- network；
- model load / cold start；
- scheduler starvation。

## 14. TPOT 很高怎么办？

排查：

- batch size；
- memory bandwidth；
- KV cache read；
- quantization；
- tensor parallel communication；
- speculative decode；
- kernel efficiency。

## 15. OOM 但 GPU utilization 不高怎么办？

可能是：

- KV cache 占满；
- fragmentation；
- batch/seq 配置；
- workspace；
- graph capture；
- duplicated model/ref；
- tensor parallel imbalance。

不能只通过 utilization 判断显存问题。

---

# Part J｜真实面试拆解

## 16. DeepSeek｜PagedAttention + Speculative Decoding ★★★★★

### 面试官要求的深度

PagedAttention：至少说清 block table、物理 block、共享/回收。

Spec Decode：至少说清 draft → target verify → accept/reject；工程上说 KV rollback、continuous batching、acceptance rate。

如果只回答“PagedAttention 减少显存碎片，Spec Decode 用小模型加速”，只能算表面。

---

## 17. 腾讯｜vLLM 动态批处理与显存管理 ★★★★★

这类题的答题顺序：

1. 为什么 naive serving 低效；
2. KV cache 为什么是主要动态内存；
3. paged KV 如何管理；
4. continuous batching 如何提高 GPU 利用；
5. prefix caching 在什么 workload 有效；
6. 用 TTFT/TPOT/P99 验证而不是只看 tokens/s。

---

## 18. 小红书｜vLLM FP16 pooling NaN ★★★★☆

这是非常有工程价值的题。

排查路线：

1. 复现 input；
2. 定位第一个出现 inf/NaN 的 tensor；
3. 检查 padding/mask；
4. 检查 FP16 reduce overflow；
5. 尝试 FP32 accumulation / BF16；
6. 检查归一化前的向量范数；
7. 对比 eager 与 fused kernel；
8. 加入 numeric assertions / telemetry。

面试官看的是“你会不会系统 debug 数值问题”，不是一个固定 patch。

---

# Part K｜代码实验

## Lab 1：KV Cache

实现 cached decode，与 full recompute 对比 logits。

## Lab 2：Prefix Cache 命中实验

同一个长 prefix，连续问两个不同问题，记录 TTFT。

## Lab 3：vLLM benchmark

记录：

- input/output tokens；
- concurrency；
- TTFT；
- TPOT；
- throughput；
- P99。

## Lab 4：Spec Decode

选择 draft/target，比较不同 draft length 与 acceptance rate。

## Lab 5：显存账本

输出：

```text
weights
kv cache
activations/workspace
cuda graph
other
```

避免只看 `nvidia-smi` 总量。

---

# 高频题库

## S 级 ★★★★★

- prefill vs decode
- KV Cache
- KV Cache size
- MHA/GQA 与 cache
- FlashAttention 原理与误区
- PagedAttention
- vLLM 为什么快
- continuous batching
- prefix caching
- speculative decoding
- TTFT/TPOT/throughput/P99

## A 级 ★★★★☆

- chunked prefill
- quantization
- KV quantization
- tensor parallel serving
- scheduling fairness
- prefix cache 安全隔离
- numerical stability / NaN debug

## B 级 ★★★☆☆

- disaggregated prefill/decode
- cache eviction policy
- CUDA graph
- expert-parallel serving
- speculative families（n-gram/EAGLE/Medusa 等）

---

# 权威资料

- Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention.
- vLLM 官方设计文档。
- Dao et al., FlashAttention / FlashAttention-2 / FlashAttention-3.
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding.

真实面经索引见 [`09-2026-real-interviews.md`](./09-2026-real-interviews.md)。

---

# 本章验收

- [ ] 画出 prefill/decode 生命周期。
- [ ] 能估 KV Cache。
- [ ] 不把 FlashAttention 和 PagedAttention 混淆。
- [ ] 能解释 continuous batching 和 prefix caching。
- [ ] 能解释 speculative decode 的 acceptance 与工程 rollback。
- [ ] 能用 TTFT/TPOT/P99 分析系统，而不是只说“快”。
