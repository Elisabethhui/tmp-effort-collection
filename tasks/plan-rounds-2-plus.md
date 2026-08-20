# Job Interview｜第 2 轮及后续学习、资料与代码总计划

> 状态：DRAFT FOR REVIEW
> 本文件只定义后续范围、依赖、验收和审核点；用户确认前不启动后续代码实现。

## 1. 目标与当前基线

目标是把当前面试资料库继续推进成一条可执行的学习流水线：

```text
官方来源 → 概念卡 → 可运行 Lab → 失败样例 → 30 秒/3 分钟/15 分钟回答 → 模拟面试
```

当前第一轮基线：

- GitHub `main` 已推送到 `f8b963c`。
- 已有 Attention、SFT loss、Toy GRPO、RAG 指标、KV Cache 和 Durable Agent 核心 Lab。
- 核心 Lab 全量测试为 20/20 通过。
- [第一轮路线图](../00-learning-roadmap.md) 和 [官方资源研究](../sources/learning-resources-research.md) 已存在。

当前第二轮候选材料已经在本地工作区，但尚未推送：

- `reference/GenAI_Agents/`：只读参考仓库，固定 revision `187b99c015386f9f91d86a0e71721d440bfaa84`；
- `modernized/`：LangChain/LangGraph 与 GraphRAG 的现代化迁移副本；
- `tasks/plan.md`、`tasks/todo.md`：已有的 Agent Notebook 迁移专项计划；
- `.venv-langchain/`、`.venv-graphrag/`：本地环境候选，不应提交。

这些内容只能标为 `candidate / planned / queued`，不能标为已学习、已验证或已推送。

## 2. 总体原则

1. **来源分层**：论文解释为什么成立，官方文档解释当前 API，面经只证明问题出现过。
2. **一项能力一个闭环**：每个专题必须同时有概念、代码、测试、失败样例和面试回答。
3. **先小后大**：先 CPU/离线/tiny 数据正确性，再进入真实模型、网络服务或多卡性能。
4. **版本显式化**：执行实验时记录 Python、包版本、模型/data revision、硬件、命令和 Git commit。
5. **不伪造结果**：没有 Linux/CUDA/API key 的实验只报告理论或 smoke test，不写 benchmark 已复现。
6. **知识晋升有门禁**：未通过官方来源交叉验证和最小实验的内容留在 `job_interview`，不直接进入 `knowledge`。
7. **每轮可暂停**：每个 checkpoint 都可以停下来学习、复盘或调整岗位侧重点。

## 3. Mac 适配的双轨学习与验证范式

Mac 不是把 GPU 内容删掉，而是把“正确性学习”和“硬件性能验证”分开。每个 GPU 相关主题都必须标注以下等级：

| 等级 | 在当前 Mac 上怎么做 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `RUNNABLE_CPU` | CPU 上跑 tiny tensor、纯 PyTorch reference、单元测试 | 公式、shape、mask、状态和数值正确性 | GPU kernel 性能、真实吞吐 |
| `MPS_OPTIONAL` | `torch.backends.mps.is_available()` 为真时再跑；否则跳过 | Apple 后端的接口兼容性 | CUDA/NVIDIA 专属特性 |
| `SOURCE_READ` | 阅读官方文档和源码，画调用链、状态机、内存账本，写 CPU toy 对照 | API 契约、数据流、核心算法、失败路径 | 实际 GPU 带宽和 kernel speedup |
| `REMOTE_GPU` | 以后在 Linux/CUDA/云 GPU 上做最小 smoke 或 benchmark | CUDA kernel、FSDP、vLLM 真实性能 | 不应反推本机结果 |

GPU 专属内容的固定学习步骤：

1. 先读接口和数据结构，不从 kernel 细节开始；
2. 记录输入/输出 shape、生命周期、内存单位和失败条件；
3. 用 CPU/纯 Python 写一个 reference implementation；
4. 用小张量、显存公式或 fake metadata 验证不变量；
5. 最后才读 CUDA/服务调度实现，并把性能结论标为 `REMOTE_GPU`。

当前优先源码阅读入口：

- Transformers KV Cache：`cache_explanation.md`、`cache_utils.py`；
- vLLM V1：`kv_cache_manager.py`、`block_pool.py`、`kv_cache_coordinator.py`、prefix caching design；
- PyTorch Attention：`scaled_dot_product_attention`、attention backend、Transformer building blocks；
- Hugging Face Llama：RoPE、RMSNorm、SwiGLU、GQA 和 decoder layer。

官方入口（执行时重新核对版本）：

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) 与 [MPS fallback 环境变量](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)；
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 与 [Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)；
- [Transformers Cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation) 与 [`cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)；
- [Transformers RoPE utilities](https://huggingface.co/docs/transformers/internal/rope_utils) 与 [Llama model notes](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama.md)；
- vLLM [KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)、[BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)、[KV coordinator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py) 和 [prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)。

## 4. 面试基础与 Transformer 必修覆盖

现有 `01`、`08`、`11` 文档有很多内容，但后续执行必须按下面的“必修清单”验收，不能只以读过文档为完成：

| 优先级 | 必修簇 | 必须会回答/会写 |
|---|---|---|
| S | Python/数据结构 | dict/set、heap/top-k、LRU、二分、滑窗、BFS/DFS、排序、DP、复杂度和边界测试 |
| S | 数学/优化 | 矩阵乘法、softmax/logsumexp、CE、KL、梯度、链式法则、SGD/Adam、数值稳定性 |
| S | PyTorch | tensor shape/device/dtype、autograd、Module、optimizer、train/eval、checkpoint、gradient accumulation |
| S | Transformer block | embedding、Q/K/V、scale、causal/padding mask、MHA/MQA/GQA、残差、Pre/Post-Norm、RMSNorm、FFN/SwiGLU |
| S | Position/LM | learned/sinusoidal/RoPE、decoder-only、logits、label shift、weight tying、tokenizer/chat template |
| S | Inference | prefill/decode、KV cache、prefix cache、continuous batching、量化、TTFT/TPOT/P95 |
| A | Attention systems | SDPA、FlashAttention、PagedAttention、block allocation、memory fragmentation、kernel 与 API 边界 |
| A | ML/DL 基础 | bias/variance、过拟合、正则化、Norm、初始化、学习率、AUC/F1/NDCG、数据泄漏 |
| A | 系统基础 | 吞吐/延迟、并发、队列、重试、幂等、可观测性、显存和通信账本 |
| B | Advanced | MoE/router/load balance、speculative decoding、长上下文、量化误差、多模态 |

Transformer 必须按以下顺序学习：

```text
Embedding/Token
 → QKV 与 shape
 → scaled dot-product + mask
 → MHA/MQA/GQA
 → Position/RoPE
 → FFN/SwiGLU + Residual/Norm
 → Decoder-only LM + label shift
 → KV Cache + Prefill/Decode
 → SDPA/Flash/Paged attention
 → MoE/长上下文/量化
```

## 5. 第 2 轮到第 10 轮路线

这里的“轮”按 7 天计算；每天默认 60–120 分钟，30% 阅读、50% 代码、20% 口述与复盘。

| 轮次 | 天数 | 主线 | 主要资料 | 代码/实验产出 | 面试产出 |
|---|---:|---|---|---|---|
| R2 | 8–14 | Python/算法、数学、ML/DL 面试底座 | `08`、`11`、MIT 6.006、PyTorch Basics | top-k/LRU/BFS/CE/softmax/Adam/metrics CPU Labs | 复杂度、数值稳定、过拟合、指标与边界 |
| R3 | 15–21 | Transformer 核心结构（前置主线） | `01`、Attention 论文、PyTorch building blocks、HF Llama | MHA、mask、RoPE、RMSNorm、SwiGLU、GQA、tiny block | 逐层 shape、复杂度、Pre/Post-Norm、RoPE、MHA/MQA/GQA |
| R4 | 22–28 | PyTorch 训练闭环与 decoder LM | `11`、CS336 A1、`02` | train loop、tokenizer/label shift、checkpoint、tiny LM | 不收敛排查、loss shift、训练/推理边界 |
| R5 | 29–35 | SFT、LoRA、DPO、GRPO | `02`、TRL、PEFT、DPO/DeepSeekMath 论文 | toy SFT/LoRA、DPO loss、GRPO reward-hacking | answer-only loss、LoRA trade-off、KL/clip/reward |
| R6 | 36–42 | 推理、显存与 Serving（源码优先） | `03`、HF Cache、vLLM V1、PagedAttention | KV 账本、CPU cache toy、源码调用链、可选 benchmark | TTFT/TPOT/P95、block pool、prefix cache、量化 |
| R7 | 43–49 | RAG、搜索、热更新与权限 | `04`、`13`、RAG 论文、检索评测 | BM25+dense+RRF、rerank、versioned index、失败分类 | Recall/MRR/NDCG、chunk、权限、RAG vs SFT |
| R8 | 50–56 | Agent Runtime、MCP、Memory、Durability | `05`、`06`、`reference/GenAI_Agents` Phase 1 | 迁移 5 个核心 Notebook；tool schema、trace、checkpoint | Workflow vs Agent、MCP、retry、幂等、human-in-loop |
| R9 | 57–63 | Agent 评测、分布式与系统设计 | `07`、`12`、Evaluate/lm-eval/Inspect、FSDP2/ZeRO | eval harness、显存/通信账本、tiny DDP/source reading | 指标设计、故障恢复、DDP/FSDP/ZeRO |
| R10 | 64–70 | 面试题治理、项目深挖、综合模拟 | `09`、`07`、错题与项目材料 | Question Registry、回归集、项目证据包、模拟面试脚本 | 20 题三档回答、项目 15 分钟深挖 |
| R11（可选） | 71–77 | 多模态与岗位定向补强 | `14`、岗位 JD、指定模型卡 | 只选一个 VLM/多模态小实验 | 视觉编码器、对齐、幻觉与评测 |

R2–R4 是通用基础与 Transformer/训练底座，R5–R7 是 LLM 工程主干，R8–R9 是 Agent/系统能力，R10 是面试转化层。R11 不应在 R2–R10 未完成前抢跑。

## 6. 可执行任务分解

### Phase 0：第二轮资料登记与边界确认

#### Task 0.1：登记第二轮来源与状态

**Description:** 将 `reference/GenAI_Agents`、官方文档、论文和已有现代化 Notebook 统一登记，区分 `candidate / planned / migrated / verified / rejected`。

**Acceptance criteria:**

- [ ] 每个来源有 URL、source revision/date、主题、先修、目标专题和状态；
- [ ] 54 个 Notebook 不全部默认迁移，只保留与面试目标相关的候选；
- [ ] 任何未执行材料不会被标记为 `verified`。

**Verification:** 运行 manifest 生成/校验脚本；人工抽查 5 条来源和 5 个 Notebook。

**Dependencies:** None。
**Likely files:** `meta/resource-registry.md`、`modernized/manifests/notebook-migration.tsv`、`sources/learning-resources-research.md`。
**Scope:** M。

#### Task 0.2：建立更新与知识晋升门禁

**Description:** 建立 Question ID、来源、版本、验证状态、错题和 knowledge promotion 的最小记录格式。

**Acceptance criteria:**

- [ ] 新面经可记录来源、日期、公司/岗位、重复证据和主教材映射；
- [ ] 稳定知识至少有一个 primary source 和一个实验/测试证据；
- [ ] 版本敏感命令保留在 `job_interview`，不直接复制进稳定知识。

**Verification:** 用 3 道现有面经题做一次完整登记；检查能从题目跳到专题、Lab 和来源。

**Dependencies:** Task 0.1。
**Likely files:** `meta/question-registry.md`、`meta/update-ledger.md`、`meta/knowledge-promotion.md`。
**Scope:** M。

### Phase 1A：R2–R3 面试基础与 Transformer 主干

#### Task 1A.1：Python/算法/数学/ML-DL 基础垂直切片

**Description:** 从 `08`、`11` 中选面试最高频的可编码基础，形成 CPU-first 小 Lab，而不是把整本经典 ML/DL 重新读一遍。

**Acceptance criteria:**

- [ ] 完成 top-k/heap、LRU、BFS/DFS、二分或滑窗中的至少 4 个带边界测试的题；
- [ ] 手写 stable softmax、cross-entropy、KL、SGD/Adam 的最小 reference；
- [ ] 能解释过拟合、正则化、Norm、初始化和 Accuracy/F1/AUC/NDCG 的适用边界。

**Verification:** CPU 单元测试 + 5 道算法题复杂度口述 + 3 个数值稳定性反例。

**Dependencies:** Task 0.2。
**Likely files:** `labs/foundations/algorithms.py`、`labs/foundations/losses.py`、`labs/foundations/test_foundations.py`、`labs/foundations/README.md`。
**Scope:** M。

#### Task 1A.2：Transformer 结构地图与逐层 shape

**Description:** 把 Embedding、QKV、Attention、FFN、Residual、Norm、LM Head 和 logits 串成一个 decoder-only block；每一步都写输入/输出 shape、复杂度和 mask 语义。

**Acceptance criteria:**

- [ ] 能从 `[B,T]` 追到 `[B,T,V]`，并标出每个 reshape/transpose；
- [ ] 覆盖 MHA/MQA/GQA、causal/padding mask、Pre/Post-Norm、RMSNorm、SwiGLU；
- [ ] 解释 encoder-only、decoder-only、encoder-decoder 的面试差异。

**Verification:** 用 tiny tensor 画一张数据流图；对每个 shape 写断言；完成 10 道 Transformer 高频题的三档回答。

**Dependencies:** Task 1A.1。
**Likely files:** `01-transformer-attention.md`、`labs/attention/README.md`、`labs/attention/test_transformer_shapes.py`。
**Scope:** M。

#### Task 1A.3：MHA 到 Transformer Block 的 CPU reference Lab

**Description:** 扩展现有 MHA Lab，加入 RoPE、RMSNorm、SwiGLU 和 decoder block；PyTorch SDPA 只作为数值对照，不把 `nn.Transformer` 当作理解替代品。

**Acceptance criteria:**

- [ ] MHA、RoPE、Norm、FFN、残差的 shape/数值/异常路径均有测试；
- [ ] tiny block 能在 CPU 上完成 forward 和一个 loss/backward；
- [ ] 至少有一个错误 mask 或错误 transpose 的失败样例。

**Verification:** `.venv/bin/python -m unittest discover -s labs -p 'test_*.py' -v`；固定 seed 数值对齐。

**Dependencies:** Task 1A.2。
**Likely files:** `labs/attention/transformer_block.py`、`labs/attention/test_transformer_block.py`、`labs/attention/README.md`。
**Scope:** M。

#### Task 1A.4：GPU/Serving 源码阅读包

**Description:** 为 KV Cache、SDPA/Flash、PagedAttention 和 vLLM V1 建立源码导航，不要求本机运行 CUDA；每个主题补一份 CPU toy 或内存账本。

**Acceptance criteria:**

- [ ] 每个主题有入口文件、关键数据结构、状态变化、shape 不变量和失败路径；
- [ ] KV Cache 能用 Transformers Cache 说明 `K/V append`、layer 维度和 mask；
- [ ] vLLM 能解释 `KVCacheManager → BlockPool → scheduler/request` 的关系；
- [ ] 所有性能结论标记为 `SOURCE_READ` 或 `REMOTE_GPU`，不伪造本机 benchmark。

**Verification:** 产出 4 张源码调用链/状态图和 4 个 CPU toy/账本测试；人工逐项复述。

**Dependencies:** Task 1A.2。
**Likely files:** `labs/inference/source-reading.md`、`labs/inference/kv_cache_toy.py`、`labs/inference/test_kv_cache_toy.py`、`sources/learning-resources-research.md`。
**Scope:** M。

### Checkpoint T：基础与 Transformer

- [ ] Python/算法、数学/优化、ML/DL、PyTorch 基础各有最小代码证据；
- [ ] Transformer 逐层 shape、mask、RoPE、Norm、FFN、MHA/MQA/GQA 可以口述和手写；
- [ ] GPU 专属部分完成源码阅读和 CPU reference，不把 GPU 性能当作通过条件；
- [ ] 用户审核：继续训练闭环，还是先做一轮 Transformer 面试题。

### Phase 1B：R4 训练闭环

#### Task 1.1：训练循环与 checkpoint Lab

**Description:** 在现有 PyTorch 核心环境中补一个 tiny classifier/LM train loop，覆盖 train/eval、zero-grad、optimizer、gradient accumulation、保存与恢复。

**Acceptance criteria:**

- [ ] toy 数据可过拟合或 loss 稳定下降；
- [ ] 恢复 checkpoint 后 loss/step/optimizer state 连续；
- [ ] 至少覆盖空 batch、shape 错误和 NaN/梯度异常边界。

**Verification:** 单元测试 + 固定 seed 的短训练 smoke；保存一页结果记录。

**Dependencies:** Checkpoint T。
**Likely files:** `labs/foundations/train_loop.py`、`labs/foundations/test_train_loop.py`、`labs/foundations/README.md`。
**Scope:** M。

#### Task 1.2：Tokenizer/label shift 解释与实验

**Description:** 不依赖大模型下载，使用小词表验证 causal LM 的 input/target shift、padding 和 loss 对齐。

**Acceptance criteria:**

- [ ] 能打印每个 token 的 input、target、mask 和 loss 位置；
- [ ] prompt/padding 不进入 answer-only loss；
- [ ] 能解释 tokenizer、special token 和 chat template 的边界。

**Verification:** 与现有 SFT loss Lab 数值对齐；加入至少一个错位标签的反例。

**Dependencies:** Task 1.1、Task 1A.2。
**Likely files:** `labs/posttraining/label_shift.py`、`labs/posttraining/test_label_shift.py`、`labs/posttraining/README.md`。
**Scope:** S。

### Checkpoint A：R4 训练闭环结束

- [ ] Task 0.1–1.2、Task 1A.1–1A.4 和 Task 1.1–1.2 验收通过；
- [ ] 所有 Lab 测试通过；
- [ ] 能完成 5 道训练基础题的 30 秒和 3 分钟回答；
- [ ] 能区分训练/推理、cache on/off、label shift 和 device/backend 边界；
- [ ] 用户审核：继续 R5 后训练，还是暂停学习并复盘。

### Phase 2：R5 后训练

#### Task 2.1：LoRA/PEFT toy 实验

**Description:** 在小模型或线性 toy network 上比较 full fine-tuning 与 adapter 的可训练参数、梯度路径、保存和合并边界。

**Acceptance criteria:**

- [ ] 打印 trainable params 比例；
- [ ] 保存/加载 adapter 后输出可复现；
- [ ] 能说明 rank、alpha、target modules 和 merge 的 trade-off。

**Verification:** 固定 seed 的参数/输出对比测试；无模型下载时使用 toy module。

**Dependencies:** Checkpoint A。
**Likely files:** `labs/posttraining/lora_toy.py`、`labs/posttraining/test_lora_toy.py`、`labs/posttraining/README.md`。
**Scope:** M。

#### Task 2.2：DPO/GRPO 完整数据流

**Description:** 在现有 Toy GRPO 上补 preference pair、policy/reference log-prob、DPO loss、reward hacking 和 KL 诊断。

**Acceptance criteria:**

- [ ] 能从 prompt 到 reward/advantage/ratio/clip/KL 打印完整中间量；
- [ ] 构造一个 reward hacking 反例，并写出修复约束；
- [ ] DPO/GRPO 的形状、极端 reward 和 policy lag 有测试。

**Verification:** 纯 PyTorch 单元测试；再用 TRL 小样例做一次输出对齐（环境可用时）。

**Dependencies:** Task 1.2、Checkpoint A、现有 `labs/posttraining/grpo_toy.py`。
**Likely files:** `labs/posttraining/dpo_toy.py`、`labs/posttraining/test_dpo_toy.py`、`labs/posttraining/README.md`。
**Scope:** M。

### Checkpoint B：R5 后训练结束

- [ ] 能独立解释 SFT、LoRA、DPO、GRPO 的目标函数和适用边界；
- [ ] 有一个 reward hacking 失败记录；
- [ ] 没有把 TRL 黑盒输出冒充为原理理解。

### Phase 3：R6 推理与 Serving

#### Task 3.1：推理显存与延迟账本

**Description:** 扩展现有 KV Cache Lab，加入权重、KV、激活、workspace、batch/concurrency 和 dtype 的明确假设。

**Acceptance criteria:**

- [ ] 给定模型配置能输出分项显存估算；
- [ ] 能解释 MHA/GQA/MQA 对 KV cache 的影响；
- [ ] 明确哪些是理论值，哪些必须用 profiler/serving 实测。

**Verification:** 参数化测试 + 两个手算样例；不报告未运行的 GPU benchmark。

**Dependencies:** None（可与 Phase 2 并行，但建议顺序执行）。
**Likely files:** `labs/inference/memory.py`、`labs/inference/test_memory.py`、`labs/inference/README.md`。
**Scope:** S。

#### Task 3.2：Serving benchmark（可选硬件门）

**Description:** 在 Linux/CUDA 或明确可运行的本地后端中，用 vLLM/同类服务比较 concurrency、prompt/output length 和 prefix reuse。

**Acceptance criteria:**

- [ ] 保存启动命令、模型 revision、硬件和 workload；
- [ ] 报告 TTFT、TPOT/ITL、P50/P95/P99、tokens/s 和显存；
- [ ] 明确“吞吐更高”不等价于“单请求延迟更低”。

**Verification:** 运行官方 benchmark CLI；没有合适硬件时只完成命令/指标设计，不标为完成。

**Dependencies:** Task 3.1。
**Likely files:** `labs/inference/benchmark.md`、`labs/inference/collect_results.py`、`labs/inference/results/README.md`。
**Scope:** M。

### Phase 4：R7 RAG 与搜索

#### Task 4.1：Hybrid Retrieval 与 rerank

**Description:** 在当前 RAG metrics 基础上补 sparse/dense 两路候选、RRF 融合、rerank 接口和失败 query 分类。

**Acceptance criteria:**

- [ ] dense-only、sparse-only、hybrid 有可比较的 Recall/MRR/NDCG；
- [ ] 每个结果保留 source/chunk/version metadata；
- [ ] 至少有 5 个失败 query，按召回、切分、排序、权限或生成错误分类。

**Verification:** 纯本地小语料集回归测试；结果可重复。

**Dependencies:** Task 0.2、现有 `labs/rag/metrics.py`。
**Likely files:** `labs/rag/hybrid.py`、`labs/rag/test_hybrid.py`、`labs/rag/README.md`。
**Scope:** M。

#### Task 4.2：Versioned index 与权限边界

**Description:** 实现 v1/v2 索引构建、validation gate、atomic alias switch、rollback 和 document ACL 过滤。

**Acceptance criteria:**

- [ ] 新索引验证失败时旧索引继续服务；
- [ ] 删除文档不会返回 ghost chunk；
- [ ] embedding/index/document version 可追踪。

**Verification:** 注入构建失败、删除和回滚场景；检查检索结果和审计日志。

**Dependencies:** Task 4.1。
**Likely files:** `labs/rag/versioned_index.py`、`labs/rag/test_versioned_index.py`、`labs/rag/README.md`。
**Scope:** M。

### Checkpoint C：R6–R7 推理与 RAG 结束

- [ ] 有一份 serving 结果或明确的硬件阻塞记录；
- [ ] Hybrid RAG 有 ablation 和失败分类；
- [ ] 能回答 RAG vs SFT、Recall vs NDCG、热更新和权限问题。

### Phase 5：R8 Agent Runtime 与现代化 Notebook

#### Task 5.1：Phase 1 Notebook 现代化

**Description:** 按现有 `tasks/plan.md`，优先迁移 5 个高价值 Notebook：StateGraph、create_agent、MCP、求职助手、HR workflow。

**Acceptance criteria:**

- [ ] 原 Notebook 保持只读，现代副本有 source mapping；
- [ ] 每个 Notebook 无内置安装命令，有无 API key 的 offline smoke path；
- [ ] 版本、官方来源、live provider 风险和运行命令明确；
- [ ] 5 个 Notebook 通过静态检查，至少 3 个离线 smoke 通过。

**Verification:** `.venv-langchain/bin/python modernized/validation/validate_notebooks.py`；检查 manifest 状态。

**Dependencies:** Task 0.1、Task 0.2；必须先确认 Python/包版本。
**Likely files:** `modernized/all_agents_tutorials/*.ipynb`、`modernized/manifests/notebook-migration.tsv`、`modernized/validation/`。
**Scope:** L（拆为每个 Notebook 一个子任务后执行）。

#### Task 5.2：Memory、durability、MCP 与现有 Lab 对齐

**Description:** 把 LangGraph checkpointer/store、tool schema、retry、idempotency、trace 与现有 `labs/agent` 状态机形成对照实验。

**Acceptance criteria:**

- [ ] 有短期 thread checkpoint 与跨 thread long-term store 的边界说明；
- [ ] tool timeout/429/重复 resume 有故障注入测试；
- [ ] trace 至少包含 state、tool、retry、latency 和失败原因。

**Verification:** 离线状态机测试；若使用真实框架，固定版本并跑最小 smoke。

**Dependencies:** Task 5.1、现有 `labs/agent`。
**Likely files:** `labs/agent/trace.py`、`labs/agent/test_trace.py`、`modernized/all_agents_tutorials/`。
**Scope:** M。

### Phase 6：R9 评测、分布式与综合能力

#### Task 6.1：Interview/Agent evaluation harness

**Description:** 将面试问答和 Agent trajectory 统一成可回归的 task/scorer/report 格式。

**Acceptance criteria:**

- [ ] 支持 correctness、mechanism、trade-off、code、production risk 五维评分；
- [ ] Agent 额外支持 tool selection、argument correctness、loop/retry、latency、cost、side-effect safety；
- [ ] 每次失败保留输入、答案、评分、trace 和修复建议。

**Verification:** 用 10 道去重面经题和 3 条 Agent trace 生成一份回归报告。

**Dependencies:** Task 0.2、Task 5.2。
**Likely files:** `labs/eval/schema.py`、`labs/eval/scorers.py`、`labs/eval/test_scorers.py`、`labs/eval/README.md`。
**Scope:** M。

#### Task 6.2：分布式训练与显存/通信账本

**Description:** 先做 CPU/单卡可解释账本，再在有多 GPU 时做 DDP/FSDP2/ZeRO tiny smoke。

**Acceptance criteria:**

- [ ] 能估算参数、梯度、optimizer state 和 activation 的分片关系；
- [ ] 能解释 all-reduce、all-gather、reduce-scatter 的位置；
- [ ] 没有多 GPU 时标记 `theory-only`，不伪造性能结论。

**Verification:** 账本参数化测试；可选 `torchrun` smoke 和 checkpoint state-dict 检查。

**Dependencies:** Task 3.1。
**Likely files:** `labs/distributed/memory_accounting.py`、`labs/distributed/test_memory_accounting.py`、`labs/distributed/README.md`。
**Scope:** M。

### Phase 7：R10 面试转化与知识晋升

#### Task 7.1：真实面经结构化

**Description:** 将 `09-2026-real-interviews.md` 的题目拆成 Question ID、公司/岗位/日期、证据、主题、难度、回答状态和复现来源。

**Acceptance criteria:**

- [ ] 重复题合并，新增追问链接到主问题；
- [ ] 每题映射至少一个专题、一个 Lab 或理论验证；
- [ ] 二手面经不被写成技术事实，事实字段有 primary source 或 `unverified` 标记。

**Verification:** 先迁移 20 道高频题；生成去重统计和缺口清单。

**Dependencies:** Task 0.2、Task 6.1。
**Likely files:** `09-real-interviews/`、`meta/question-registry.md`、`sources/2026-08-19-sources.md`。
**Scope:** L（先做 20 题垂直切片）。

#### Task 7.2：项目深挖与三档回答包

**Description:** 为每个目标项目/实验生成 30 秒、3 分钟、15 分钟回答，覆盖 What/Why/How/Trade-off/Production。

**Acceptance criteria:**

- [ ] 至少 3 个项目有完整证据包：目标、baseline、指标、bad case、成本、恢复和个人贡献；
- [ ] 每题都有追问树和不会时的诚实边界；
- [ ] 两场 45 分钟模拟面试有评分和复盘。

**Verification:** 使用 evaluation harness 评分；人工复核项目事实。

**Dependencies:** Task 7.1、Task 6.2。
**Likely files:** `07-project-deep-dive.md`、`interview-packs/`、`labs/eval/`。
**Scope:** M。

#### Task 7.3：稳定知识晋升

**Description:** 只将通过来源、实验和稳定性审查的概念抽取到独立 knowledge 层；原始题目、版本命令和错题继续留在本仓库。

**Acceptance criteria:**

- [ ] 每条晋升知识有 primary source、验证 commit、适用版本和反例；
- [ ] 没有把公司/日期/面经样本复制成长期知识；
- [ ] 能从 knowledge 条目反向链接到 job_interview 专题和 Lab。

**Verification:** 先晋升 10 个稳定概念，做链接完整性和重复性检查。

**Dependencies:** Task 7.1、Task 7.2。
**Likely files:** `knowledge/`（若用户批准建立）、`meta/knowledge-promotion.md`、专题 Markdown。
**Scope:** M。

## 7. Checkpoint 与审核顺序

| Checkpoint | 审核内容 | 用户可选动作 |
|---|---|---|
| T | 基础 + Transformer | 进入训练闭环 / 先做 Transformer 面试题 |
| A | R4 训练闭环 | 进入后训练 / 暂停学习复盘 |
| B | R5 后训练 | 进入推理源码阅读 / 先巩固目标函数 |
| C | R6 Serving + R7 RAG | 进入 Agent / 先补硬件 benchmark |
| D | R8 Agent + R9 Eval/Distributed | 进入模拟面试 / 增加岗位专项 |
| E | R10 20 题 + 项目包 | 晋升 knowledge / 保持面经库 |

每个 checkpoint 必须满足：测试通过、文档有来源、失败样例已记录、未验证项已标记、用户可以暂停。

## 8. 推荐执行策略

不建议“先把所有规划和代码全部做完，再开始学习”，也不建议只读资料不写代码。推荐采用 **一轮规划 + 一轮学习 + 一轮实现 + checkpoint 审核**：

```text
先审核本计划
  → 只执行 R2–R3 的基础/Transformer 最小代码切片
  → 用户学习/口述
  → Checkpoint T
  → 再决定进入 R4 训练，还是补 Transformer 薄弱点
```

如果当前暂时不学习，可以先完成 Task 0.1–0.2 的治理登记和 GPU 源码阅读索引；不建议在没有审核的情况下迁移全部 54 个 Notebook，也不建议先做真实 GPU benchmark。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 54 个 Notebook 重复或过时 | 维护成本高、学习主线发散 | 只迁移 5 个 Phase 1 + 4 个 Phase 2 + 3 个 RAG 候选，其余保留索引 |
| 框架版本变化 | Notebook 失效 | 每轮 pin 版本，执行前重验官方迁移文档 |
| Apple 本机缺少 CUDA/部分后端 | benchmark 无法复现 | `RUNNABLE_CPU` + `SOURCE_READ`；真实性能放到 Linux/CUDA |
| API key、费用和隐私 | 产生外部副作用 | offline smoke 默认路径；live provider 显式开关 |
| 面经事实不可靠 | 错误知识进入教材 | 面经只做出现证据；主张必须有 primary source 或 `unverified` |
| knowledge 与 job_interview 重复 | 双份维护、漂移 | 只在通过 promotion gate 后晋升，并保留双向链接 |

## 10. 需要用户审核的四个决定

1. **目标岗位权重**：默认算法/LLM/Agent 综合；是否需要提高某一方向权重？
2. **时间预算**：默认每天 60–120 分钟；是否按 30 分钟或 2 小时版本执行？
3. **第二轮范围**：是否批准先做“基础 + Transformer”双轨，还是先做 Agent/MCP/Memory Notebook？默认建议先基础/Transformer。
4. **knowledge 层**：是否同意在验证 10 个概念后创建独立 `knowledge/` 目录？默认建议暂缓创建。

## 11. 总体完成定义

- [ ] R2–R10 每轮都有可运行、源码阅读或可验证产出；
- [ ] 20+ 道高频题完成三档回答和评分；
- [ ] 至少 3 个项目有可核验的项目证据包；
- [ ] Agent、RAG、Serving、Distributed 的理论/代码/工程边界清楚；
- [ ] knowledge 只收录稳定、来源清楚、实验验证过的概念；
- [ ] 所有未完成硬件/版本/API 依赖都有明确状态，不伪装成完成。
