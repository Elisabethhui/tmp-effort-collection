# 09｜2026 大厂真实面试专栏：按公司、岗位、日期复盘

> 版本：2026-08-19  
> 本文职责：记录“今年真实在问什么”，不替代专题知识文件。  
> 学习方式：先看本文件定位题目，再跳转专题文件把原理、代码与工程实现学透。

---

## 0. 如何读这篇专栏

公开面经有天然偏差：候选人记忆可能不完整，题目措辞可能被转述，网上的“参考答案”也不一定正确。因此本仓库把两类证据严格分开：

- **面经证据**：判断某家公司/岗位最近“问了什么”。
- **答案 Authority**：论文、官方技术报告、官方文档、官方代码，用来判断“答案是否正确”。

### 标签

- `🔥 高频`：多个公司/多个面经重复出现，优先掌握。
- `↗ 升温`：2026 年明显增多的新方向。
- `🧨 挖坑`：看似简单，容易被连续追问。
- `💻 手写`：需要现场代码或伪代码。
- `🏗 工程`：要求从真实系统约束回答。

---

# 1. 美团｜2026-08-06｜北斗计划 / 大模型方向

## 1.1 这一场释放了什么信号

这份面经很有代表性：它把 **训练、后训练、推理、RAG、Agent、分布式、算法题** 串在同一轮里。说明大模型岗位已经不是“只背 Transformer”，而是看你能不能理解一个模型从训练到上线再到 Agent 应用的完整生命周期。

## 1.2 题目组 A：SFT 与数据

### Q1：LoRA 的原理是什么？关键超参数有哪些？ `🔥 🧨`

**面试官在考什么**

不是只听“低秩矩阵”。通常会继续问：为什么低秩可行、rank 怎么选、A/B 矩阵怎么初始化、哪些层应该挂 LoRA、merge 后发生什么。

**推荐回答骨架**

1. 冻结原权重 `W`，只学习低秩增量 `ΔW = BA`。
2. `rank r` 控制参数量/表达能力；`alpha` 控制缩放；dropout 是正则项。
3. 常见目标模块是 Attention 的 q/k/v/o projection，是否扩展到 FFN 取决于任务和容量。
4. 推理时可将 `BA` 合并回 `W`，避免额外分支开销。
5. rank 不是越大越好，要用验证集和任务复杂度决定。

**工程注意**

- 检查实际可训练参数数量，不要以为配置生效就真的挂到了正确模块。
- 混合精度下注意 adapter dtype。
- 多 adapter 场景要区分 merge、hot swap 与并行组合。

→ 详见 [`02-post-training.md`](./02-post-training.md)

### Q2：为什么 SFT 通常只计算 answer 部分的 loss？Padding 怎么处理？ `🔥 💻`

**核心答案**

Instruction tuning 的目标是让模型在给定 prompt 后学习目标回答。如果把 system/user prompt 也作为监督目标，模型会被迫学习复述输入模板，且不同模板之间可能互相污染。常见做法是：

```text
input_ids:  [system][user][assistant answer][pad]
labels:     [-100 ][-100][assistant answer][-100]
```

PyTorch `CrossEntropyLoss(ignore_index=-100)` 或 Transformers Trainer 对被 mask 的 token 不计 loss。

**挖坑**

- Causal LM 还要做 **shift-right / next-token prediction**。
- EOS 是否计入 loss 会影响模型停止行为。
- 多轮对话中是否训练所有 assistant turns，要看数据策略。

→ 详见 [`02-post-training.md`](./02-post-training.md)

---

## 1.3 题目组 B：Attention / 分布式 / 推理

### Q3：Attention 如何降低显存？ `🔥 🏗`

不能只回答“用 FlashAttention”。要先拆显存来源：

1. 训练时中间激活和 `N×N` attention score；
2. 推理时 KV Cache；
3. batch / sequence padding 浪费；
4. 参数、optimizer state 和 gradient。

然后对应方案：

- FlashAttention：减少 attention 中间矩阵的 HBM IO / materialization；
- GQA/MQA：减少 KV heads，降低 KV Cache；
- varlen / packing：降低 padding 浪费；
- gradient checkpoint：用计算换激活显存；
- ZeRO/FSDP：分片模型状态。

→ [`01-transformer-attention.md`](./01-transformer-attention.md) + [`03-inference-serving.md`](./03-inference-serving.md)

### Q4：ZeRO-2 和 ZeRO-3 差别是什么？ `🔥`

- ZeRO-1：分 optimizer state；
- ZeRO-2：再分 gradients；
- ZeRO-3：再分 parameters。

真正工程回答还要补：ZeRO-3 节省显存最多，但通信与参数 gather 更复杂，小模型/低通信带宽场景不一定最快。

→ [`07-project-deep-dive.md`](./07-project-deep-dive.md)

### Q5：估算 LLaMA-7B 推理显存。 `🔥 🧨`

先明确假设：仅参数？是否含 KV Cache？什么 dtype？并发多少？

最基础估算：

```text
7B parameters × 2 bytes (BF16/FP16) ≈ 14 GB
```

但真实 serving 还要加：

- KV Cache；
- runtime workspace；
- CUDA graph / allocator reserve；
- batching；
- tokenizer/host memory 等。

面试官通常是在看你会不会**主动澄清条件**，而不是背“14GB”。

→ [`03-inference-serving.md`](./03-inference-serving.md)

### Q6：vLLM 如何优化 latency / throughput？ `🔥 🏗`

建议从四层回答：

1. **KV memory management**：PagedAttention / block-based KV；
2. **scheduler**：continuous batching；
3. **reuse**：prefix caching；
4. **kernel/runtime**：optimized attention/kernel、quantization、CUDA graph 等。

然后强调 latency 与 throughput 常冲突，必须同时看 TTFT、TPOT/ITL、P95/P99、tokens/s。

→ [`03-inference-serving.md`](./03-inference-serving.md)

---

## 1.4 题目组 C：GRPO / RAG / Agent

### Q7：为什么用 GRPO，不用 PPO 或 DPO？ `🔥 🧨`

错误回答：`“GRPO 比 PPO 新，所以更好。”`

正确结构：

- PPO：online RL，通常需要 actor + critic/value；稳定但训练系统复杂、显存和算力成本高。
- DPO：离线 preference optimization，训练简单，但依赖 preference pairs，本身不是在线 rollout RL。
- GRPO：对同一 prompt 采样 group responses，用组内相对 reward 构造 advantage，省去显式 critic，适合可验证 reasoning reward 等场景。

选择依据应是：**数据形式 + reward 是否在线可得 + 系统预算 + 是否需要 exploration**。

→ [`02-post-training.md`](./02-post-training.md)

### Q8：为什么需要 RAG？什么时候比纯 SFT 更合适？ `🔥`

RAG 更适合：知识经常变化、需要引用来源、企业私有知识、多租户权限、希望不重新训练模型即可更新知识。

SFT 更适合改变：输出行为、格式、风格、任务策略和稳定技能。

**挖坑：** Memory、RAG、SFT、Skill 的职责边界一定要分清。

→ [`04-rag-retrieval.md`](./04-rag-retrieval.md) / [`05-agent-memory-context.md`](./05-agent-memory-context.md)

### Q9：为什么用 LangGraph，而不是自己手写 prompt loop？ `↗ 🏗`

不要回答“LangGraph 更方便”。真正要比较：

- State 是否显式；
- checkpoint / resume；
- branching / conditional edges；
- human-in-the-loop；
- failure recovery；
- observability；
- durable execution。

如果只是 2～3 步固定流程，手写 workflow 可能更简单；长任务/多状态/需恢复时 graph runtime 才有明显价值。

→ [`05-agent-memory-context.md`](./05-agent-memory-context.md)

---

# 2. DeepSeek｜2026-07-26｜大模型算法岗笔试

## 2.1 为什么值得重点学

这组题直接把“会解释”推进到“能实现”：手写完整 Multi-Head Attention、DPO 推导、PagedAttention 底层和 speculative decoding 工程实现。

## 2.2 手写完整 Multi-Head Attention `🔥 💻`

### 面试官最低期待

你需要正确处理：

- `Wq/Wk/Wv/Wo`；
- `[B,T,D] → [B,H,T,Dh]` reshape/transpose；
- `QK^T / sqrt(Dh)`；
- causal / padding mask；
- stable softmax；
- dropout；
- concat heads；
- shape assertions。

### 最容易挂的地方

1. 把 `d_model` 当成 scaling denominator，而不是 `d_head`；
2. mask 方向错；
3. transpose 后直接 `view` 导致非 contiguous 错误；
4. 没处理 padding mask broadcast；
5. softmax 前 dtype / `-inf` 处理错误。

→ 完整代码与逐行解释见 [`01-transformer-attention.md`](./01-transformer-attention.md)

## 2.3 PagedAttention 为什么有效？ `🔥 🏗`

面试不能停在“像操作系统分页”。要解释：

- 每个 sequence 的 KV 不必占一整块连续显存；
- logical blocks → physical KV blocks；
- scheduler 可以按需分配/释放；
- 降低 external fragmentation 和预留浪费；
- prefix sharing / copy-on-write 等机制可在 block 粒度工作。

还要知道：**当前 vLLM 实现一直在演进，不能把早期论文伪代码等同于今天源码。**

→ [`03-inference-serving.md`](./03-inference-serving.md)

## 2.4 Speculative Decoding 怎么保证结果分布正确？ `🔥 🧨`

基本思路：draft model 一次提出多个 token，target model 并行验证；接受部分候选，不接受时按校正分布采样，使最终输出仍遵循 target model 的目标分布。

工程追问：

- acceptance rate 太低还会不会快？
- draft 多大合适？
- KV Cache 如何处理接受/拒绝？
- bandwidth / verification kernel 是否成为瓶颈？

→ [`03-inference-serving.md`](./03-inference-serving.md)

---

# 3. 百度文心｜2026-03-20｜大模型后训练

这是 2026 后训练岗位最值得练的一套题之一，因为它不是名词解释，而是让候选人把 **GRPO 数据流 + 数学 + 框架 + SFT 代码** 全串起来。

## 3.1 GRPO 完整数据流 `🔥🔥`

建议面试时按照这个顺序画：

```text
prompt
  ↓
rollout policy 生成 G 个 responses
  ↓
reward / verifier
  ↓
group-relative advantage
  ↓
importance ratio
  ↓
clipped policy objective + KL regularization
  ↓
update policy
```

然后解释 `πθ`、`πold`、rollout policy/reference policy 各自职责，而不是把它们都说成“旧模型”。

## 3.2 On-policy / Off-policy 怎么判断？大 batch 为什么带来 policy lag？ `🔥 🧨`

核心不是背定义，而是看**样本由哪个策略生成，以及更新时策略离生成策略有多远**。大 batch/异步 rollout 时，样本生成到消费之间模型可能已经更新，导致 policy lag；需要 importance sampling、clip、版本控制、staleness 管理等措施。

## 3.3 现场写 Qwen2 SFT `🔥 💻`

应该能从 tokenizer、dataset、chat template、labels mask、DataLoader、model forward、CE loss、optimizer、backward、gradient accumulation、mixed precision 一路写通。

→ 详见 [`02-post-training.md`](./02-post-training.md) 和 [`08-algorithm-coding.md`](./08-algorithm-coding.md)

---

# 4. 阿里｜2026-08-02｜Agent 开发岗

这一场非常重要，因为它已经在直接问 **Agent Runtime / Context Engineering / Memory Governance**。

## 4.1 Agent 完整执行流程 `🔥`

建议回答：

```text
User Request
 → input normalization / policy
 → context assembly
 → model reasoning / planning
 → tool selection
 → schema validation
 → tool execution
 → observation
 → state update
 → retry / branch / continue
 → final answer
 → trace/eval/memory commit
```

## 4.2 多轮、多 Session Memory 怎么做？ `↗ 🔥 🧨`

不要回答“把聊天记录存数据库”。需要区分：

- working/short-term state；
- episodic memory；
- semantic memory；
- user/profile memory；
- procedural/skill knowledge；
- retrieval gate；
- write policy；
- consolidation / forgetting；
- provenance；
- privacy / tenant boundary。

## 4.3 Context 越来越长怎么优化？ `↗ 🔥`

需要谈：

1. selective retrieval；
2. summary / compression；
3. state vs transcript 分离；
4. stable prefix cache；
5. tool output truncation/structured extraction；
6. context budget；
7. stale evidence 淘汰；
8. checkpoint + pointer 恢复，而不是每次重放全历史。

## 4.4 Agent 出错怎么办？ `↗ 🏗`

需要有 failure taxonomy：

- model/tool selection error；
- invalid args；
- tool timeout；
- partial side effect；
- repeated loop；
- stale state；
- token/context overflow；
- permission error。

再分别设计 retry、backoff、idempotency key、checkpoint、compensation、human approval。

→ [`05-agent-memory-context.md`](./05-agent-memory-context.md)

---

# 5. 小红书｜2026-07-28｜大模型算法

这一场特别适合学习**检索模型真实工程问题**。

## 5.1 BCE + InfoNCE 为什么可以一起用？ `🔥`

- BCE 可以做 pair-wise/point-wise relevance supervision；
- InfoNCE 强调 batch 内相对对比；
- 联合目标可以同时约束绝对相关性与表征空间的相对分离。

真正难点在 negative sampling 和 loss scale，而不是把两项简单相加。

## 5.2 InfoNCE false negatives 怎么办？ `🔥 🧨`

可能策略：

- supervised positives 去碰撞；
- semantic near-duplicate filtering；
- debiased contrastive objective；
- hard negative mining 时设置安全边界；
- multi-positive loss。

## 5.3 vLLM FP16 mean pooling 为什么可能 overflow / NaN？ `🏗 🧨`

需要从数值范围与 reduction accumulation 解释，而不是归因“vLLM bug”。可以把 accumulator 临时提升到 FP32，再归一化；同时检查 padding、zero norm、异常 hidden states。

→ [`04-rag-retrieval.md`](./04-rag-retrieval.md) + [`03-inference-serving.md`](./03-inference-serving.md)

---

# 6. 腾讯｜2026｜大模型 / Agent / RAG

公开面经反复出现：

- Pre-Norm vs Post-Norm；
- 为什么当前 LLM 多为 Decoder-only；
- FlashAttention；
- vLLM；
- RLHF/PPO/DPO；
- RAG recall；
- hybrid retriever；
- 知识库热更新；
- tool failure / retry；
- 金融等高风险 Agent 的安全设计。

### 其中最值得练的系统题

> “知识库更新时，怎么做到不停机并且不让 retrieval index 与 metadata 不一致？”

建议回答 versioned index：

```text
source version
  → parse/chunk
  → embedding
  → build new index version
  → validation
  → atomic alias switch
  → drain old version
  → rollback available
```

不要直接在生产索引上边写边删。

→ [`04-rag-retrieval.md`](./04-rag-retrieval.md)

---

# 7. 字节跳动｜2026｜大模型 / Agent

2026 公开面经持续出现两条线：

### 线 A：后训练

- PPO clip；
- online vs offline RL；
- reference model；
- DPO/GRPO/DAPO；
- SFT loss shift；
- reward / data。

### 线 B：Agent 系统

- multi-agent cooperation/conflict；
- Agent memory；
- long-term memory retrieval；
- accuracy-latency trade-off；
- framework 与底层机制。

**趋势判断：** Agent 面试越来越不像“LangChain API 面试”，而更像 distributed stateful application / AI runtime 面试。

→ [`02-post-training.md`](./02-post-training.md) + [`05-agent-memory-context.md`](./05-agent-memory-context.md)

---

# 8. 阿里其他 2026 Agent 面经中的高价值追问

2026-07 的公开题还包括：

- hallucination / tool mis-call 怎么诊断；
- Agent eval 怎么建；
- planner prompt 怎么设计；
- sub-agent context 怎么传；
- ReAct；
- function-calling JSON；
- 模型捏造 tool args 怎么拦截；
- multi-agent 模式；
- 并发 API 打爆怎么办；
- 10k 长文档 RAG 怎么做。

这些题目共同指向：**可靠性、上下文、状态、并发、评测。**

---


# 9. 2026 多模态岗位：字节抖音电商 / 淘天

## 字节抖音电商多模态｜2026-04

公开题目把以下内容放在同一场里：

- SFT 模型、数据、框架；
- PPO reward model / loss / GPU 数；
- DPO 与 PPO 对比；
- 微调效果如何评估；
- 消融实验；
- CLIP；
- optimizer；
- MHA；
- GRPO；
- LoRA rank；
- ViT；
- Swin Transformer；
- Qwen3 fast/slow thinking。

**面试信号：** 多模态算法岗也要求大模型后训练、工程实验和基础 Transformer，而不是只准备视觉模型。

→ [`14-multimodal-vlm.md`](./14-multimodal-vlm.md)

## 淘天多模态｜2026

公开题目包括：CLIP、BLIP 三类 loss、BLIP2/BLIP3、Qwen-VL 训练流程、Q-Former vs LLaVA MLP projector、MHA 手写。

**面试官最关心的是设计动机：** 为什么需要 adaptor？为什么 query bottleneck？什么时候简单 MLP 反而更合适？

→ [`14-multimodal-vlm.md`](./14-multimodal-vlm.md)

---

# 10. 2026 分布式训练专项信号

字节大模型算法二面已经出现：

- PyTorch distributed functions；
- FSDP vs DeepSpeed ZeRO；
- NCCL gather/scatter 等 collective；
- 单卡放不下模型怎么办；
- ZeRO-3 既然参数分片，forward/update 时如何完成计算。

这意味着 ZeRO 不能只背 Stage 1/2/3 表格，必须理解 **parameter materialization、all-gather、reduce-scatter、通信/显存 trade-off**。

→ [`12-distributed-training.md`](./12-distributed-training.md)

---

# 11. 2026 面试频率地图

## S 级：必须熟练到能手写/推导

- Self-Attention / MHA
- mask / scaled dot-product
- Transformer block
- SFT causal loss
- LoRA
- PPO / DPO / GRPO 差异
- KV Cache
- RAG pipeline
- embedding / InfoNCE
- Agent tool calling
- 常用算法题

## A 级：必须能讲工程 trade-off

- GQA / MQA
- RoPE
- RMSNorm / Pre-Norm
- FlashAttention
- PagedAttention / vLLM
- Continuous Batching
- ZeRO / FSDP
- RAG hybrid retrieval / rerank
- Agent Memory / Context
- checkpoint / retry / idempotency
- evaluation

## B 级：2026 明显升温

- speculative decoding
- GRPO variants / DAPO / GSPO
- Agent trajectory data
- synthetic user/environment
- Agentic RL
- long-context engineering
- multi-agent concurrency
- verifier / PRM
- production Agent harness

## C 级：岗位相关，按 JD 深挖

- multimodal position encoding
- MoE expert parallel
- PRM / process supervision
- Text2SQL
- GraphRAG
- AI Search / ranking / NDCG
- retrieval model training

---

# 12. 如何把真实题转成自己的训练题

每看到一道面经题，不要只记答案。至少生成五层问题：

```text
L1 What：它是什么？
L2 Why：为什么需要它？
L3 How：公式/代码/流程怎么实现？
L4 Trade-off：为什么不用另一种方案？
L5 Production：上线以后怎么测、怎么观测、坏了怎么恢复？
```

例如“什么是 KV Cache？”只完成 L1。

真正能通过大模型工程面试，需要继续回答：

- KV Cache shape 怎么算？
- 每个 token 占多少 bytes？
- GQA 为什么能减少 KV？
- 32k context × 100 并发需要多少显存？
- prefix cache 与普通 KV cache 的区别？
- cache eviction 怎么做？
- multi-tenant 是否可能数据泄漏？

这才是一道题完整的学习闭环。

---

# 13. 本专栏维护规则

以后新增面经时：

1. 先查是否已有同题；
2. 如果重复，只增加 `公司 / 日期 / 岗位 / 新追问`；
3. 如果出现新的追问深度，更新对应专题；
4. 只有新知识域才新增专题；
5. 网友答案只作为线索，最终答案回到 primary source 验证；
6. 每月更新一次“频率地图”。
