# 学习资源研究｜2026-08-20

> 目标：为 `tmp-effort-collection-v2` 选择可复现、可验证、能直接转化为代码练习的学习资源。
> 本文件只收录 primary sources：官方课程/文档、官方代码仓库、论文原文或会议论文页面。
> 检索日期：2026-08-20（Asia/Shanghai）。网页会继续更新；执行实验时必须重新确认版本，并在实验记录中锁定 commit、包版本、模型 revision 和数据版本。

## 0. 快速决策

不要把下面所有资源同时读完。对面试基础和代码能力，建议按这个最短主线执行：

```text
PyTorch Basics
  → HF Course + Transformers
  → CS336 Assignment 1（手写 Transformer）
  → PEFT/TRL（SFT → DPO/GRPO）
  → vLLM + PagedAttention（服务与 benchmark）
  → RAG paper + Agent course + LangGraph persistence
  → Evaluate / lm-evaluation-harness / Inspect（评测）
  → FSDP2 / DeepSpeed ZeRO（分布式）
```

当前仓库的专题映射：

| 当前文件/实验 | 优先资源 | 第一项可交付代码 |
|---|---|---|
| `01-transformer-attention.md` | LR-001、LR-003、LR-004 | 不调用 `nn.MultiheadAttention` 的 MHA，含 causal/padding mask |
| `02-post-training.md` | LR-006～LR-012 | toy SFT、DPO、GRPO；输出 loss/reward/advantage |
| `03-inference-serving.md` | LR-016、LR-017、LR-026 | vLLM 服务 + TTFT/TPOT/P95/吞吐 benchmark |
| `04-rag-retrieval.md` | LR-018、LR-019、LR-020 | BM25+dense+RRF 的 ablation |
| `05-agent-memory-context.md` | LR-019～LR-021 | checkpoint、resume、retry、tool schema、context budget |
| `06-agent-data-eval.md` | LR-022～LR-025 | 题目集、scorer、轨迹日志和回归报告 |
| `08-algorithm-coding.md` | LR-005 | 每周完成 MIT problem set 中的 Python 题并口述复杂度 |
| `10-engineering-labs.md` | LR-001、LR-004、LR-013～LR-017 | 逐个完成 Lab 01～09 的最小验收 |
| `11-ml-dl-foundations.md` | LR-002、LR-005 | 线代/概率/优化复习 + 手写训练循环 |
| `12-distributed-training.md` | LR-013～LR-015、LR-027 | DDP → FSDP2 → ZeRO 配置对比 |
| `14-multimodal-vlm.md` | LR-006、LR-007、LR-022 | 先完成 Transformers 基础，再选一个 VLM 官方模型卡/代码做小实验 |

## 1. 资源目录

### A. 基础、PyTorch 与算法题

| ID | 适用主题 | 官方来源（原文/代码） | 版本或日期（检索时） | 先修 | 代码练习与验收 |
|---|---|---|---|---|---|
| LR-001 | PyTorch tensor、Dataset、模型、autograd、优化、checkpoint | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | 页面头部为 Tutorials 2.12.0+cu130；最后更新 2026-01-20；教程总索引已显示 2.13.0+cu130 | Python、基础 DL | 用 FashionMNIST 完成数据→模型→反向→保存/加载；另写一个 toy loop，比较 `train/eval`、梯度归零和 checkpoint 恢复 |
| LR-002 | 深度学习数学与可运行 notebook；优化、NLP、推荐、RL | [Dive into Deep Learning 官方仓库](https://github.com/d2l-ai/d2l-en)；[在线书](https://d2l.ai/) | 仓库 release 1.0.3，2026-08-18 | Python、微积分/线代基础 | 只选与当前项目对应的章节：attention、优化、NLP pretraining、推荐；每章至少运行 2 个 notebook，并把一个 NumPy/PyTorch 版本改成自己的测试 |
| LR-003 | Transformer/attention 的原始机制、复杂度、位置编码 | [Attention Is All You Need（NeurIPS 2017）](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | 2017 会议论文，版本稳定 | 线代、softmax、反向传播 | 手写 scaled dot-product attention、multi-head reshape、mask；用小张量逐项验证与 PyTorch SDPA 的数值误差 |
| LR-004 | 从 tokenizer 到训练、系统优化、数据、SFT/RL 的完整 LLM implementation track | [Stanford CS336: Language Modeling from Scratch（Spring 2026）](https://cs336.stanford.edu/)；[课程 lecture repo](https://github.com/stanford-cs336/lectures) | Spring 2026；课程页当前含 5 个 assignment | 基本 ML/DL、PyTorch；课程明确是 implementation-heavy | A1：tokenizer、Transformer、optimizer、最小 LM；A2：profiling、Triton FlashAttention2、memory-efficient distributed training；A3～A5 按时间选 data、scaling、SFT/RL。先做 A1，再按面试岗位选 A2 或 A5 |
| LR-005 | 数据结构、算法证明、Python 编码和复杂度表达 | [MIT OCW 6.006 Problem Sets（Spring 2020）](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/problem-sets/)；[课程主页](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) | Spring 2020；公开课材料稳定 | Python、离散数学入门 | 每周 1 个 problem set；先独立完成 coding 部分，再看 solution；每题写 invariant、复杂度和边界测试。不要把 solution 当背题库 |

### B. Transformers、数据、PEFT 与后训练

| ID | 适用主题 | 官方来源（原文/代码） | 版本或日期（检索时） | 先修 | 代码练习与验收 |
|---|---|---|---|---|---|
| LR-006 | Transformer 使用、tokenizer、Trainer、datasets、tokenizers、semantic search、从头训练 | [Hugging Face LLM Course](https://huggingface.co/learn/nlp-course/en/chapter1/1) | Live course；页面检索时未固定 release | Python、PyTorch、基础 NLP | 顺序完成 chapter 1～3 的 quiz/code；chapter 5 做数据切分与 FAISS semantic search；chapter 7 选 causal LM fine-tuning。保存 tokenizer、chat template、数据处理和评估脚本 |
| LR-007 | `Trainer` 的 batching、padding、forward、loss、backward、分布式参数 | [Transformers Trainer 官方文档](https://huggingface.co/docs/transformers/trainer) | Latest docs；未在 URL 中固定版本 | LR-001、LR-006 | 先写手动 SFT loop，再用 `Trainer` 复现同一 toy 数据；逐项对齐 shifted labels、padding mask、gradient accumulation 和 checkpoint |
| LR-008 | LoRA/QLoRA、adapter 注入、训练/推理、合并与多 adapter | [PEFT Quicktour](https://huggingface.co/docs/peft/quicktour)；[LoRA conceptual guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora) | Latest docs；页面未固定版本 | LR-006、Transformer module naming | 对一个小 Causal LM：比较 full fine-tuning 与 LoRA 的 trainable params、显存、吞吐；记录 `target_modules`、rank、alpha、merge 前后输出一致性 |
| LR-009 | SFT、DPO、GRPO、Reward Modeling、CLI 与分布式训练入口 | [TRL Quickstart](https://huggingface.co/docs/trl/en/quickstart)；[TRL CLI](https://huggingface.co/docs/trl/clis)；[GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) | 当前文档索引可见 v1.7.0；quickstart/CLI 为 latest docs | LR-006～LR-008；GRPO 需要概率、RL 基础 | 用官方小模型/数据做最小 SFT；再执行 DPO/GRPO 示例。验收：打印 token-level log-prob、reference/policy、group reward/advantage、KL/clip 相关量；配置文件和运行命令可重放 |
| LR-010 | 统一单卡/多卡启动、Accelerator、FSDP/DeepSpeed、Big Model Inference | [Accelerate Quicktour](https://huggingface.co/docs/accelerate/main/en/quicktour)；[installation](https://huggingface.co/docs/accelerate/basic_tutorials/install) | Main/latest docs；installation 页说明 Python 3.8+ tested | PyTorch training loop | 用 `accelerate config`、`accelerate test`；把 LR-007 手写 loop 改造成 `Accelerator` 版本，单卡与多卡输出都记录；保留 config YAML |
| LR-011 | DPO 数学、RLHF 对比、preference pair objective | [DPO 原始论文（arXiv 2305.18290）](https://arxiv.org/abs/2305.18290)；[OpenReview paper](https://openreview.net/pdf?id=HPuSIXJaa9) | 2023 原始论文 | 交叉熵、KL、policy/reference log-prob | 在 2 个 response pair 上手算 DPO loss；实现一个纯 PyTorch loss，与 TRL 输出对齐；构造 preference label 翻转，观察梯度方向 |
| LR-012 | GRPO 的由来、group-relative advantage、reasoning RL 数据流 | [DeepSeekMath 原始论文](https://arxiv.org/abs/2402.03300)；[DeepSeek 官方仓库](https://github.com/deepseek-ai/DeepSeek-Math) | 2024 原始论文/官方实现 | PPO ratio、advantage、KL；先完成 LR-011 | 先做 `10-engineering-labs.md` Toy GRPO：prompt→G samples→reward→normalize→ratio/clip→KL→update；再对照 TRL GRPOTrainer，不直接把黑盒结果当理解 |

### C. 推理、Kernel 与 serving

| ID | 适用主题 | 官方来源（原文/代码） | 版本或日期（检索时） | 先修 | 代码练习与验收 |
|---|---|---|---|---|---|
| LR-013 | DDP、FSDP2、TP、DeviceMesh、torchrun、通信 | [PyTorch Distributed tutorials](https://docs.pytorch.org/tutorials/distributed.html)；[FSDP2 tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)；[TP tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) | Tutorials 2.13.0+cu130；FSDP1 页面已标注 deprecated，应优先 FSDP2 | LR-001、Linux/CUDA、多 GPU；单卡只能读/做 shape 级 smoke test | DDP MNIST→FSDP2 Transformer toy→TP toy；用 `torchrun` 运行；记录 world size、通信量、峰值显存、checkpoint state dict，并解释 all-reduce/all-gather/reduce-scatter |
| LR-014 | ZeRO stage 1/2/3、offload、checkpoint 与配置 | [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)；[ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)；[config JSON](https://www.deepspeed.ai/docs/config-json/) | Getting Started/ZeRO 页面上次更新约 2026-08-06（检索时） | LR-013、Linux/CUDA、多 GPU/云 GPU | 同一小模型分别跑 DDP、ZeRO-2、ZeRO-3；只改 JSON 不改模型逻辑；记录 per-rank memory、吞吐、checkpoint 合并成本。先做 tiny config，不要从 1.5B 示例起步 |
| LR-015 | TP/PP/DP/EP/CP、Megatron Core、数据准备与大规模训练 | [NVIDIA Megatron-LM docs](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/index.md)；[README](https://github.com/NVIDIA/Megatron-LM) | README release 0.15.0（检索时） | LR-013/014；NVIDIA GPU、Linux；源码量大 | 只做 quickstart 和 parallelism guide；画出一层 Transformer 在 TP/PP/DP 下的 tensor/通信流；不要在没有多 GPU 环境时声称 benchmark 已复现 |
| LR-016 | offline inference、OpenAI-compatible serving、attention backends、benchmark | [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)；[vLLM CLI guide](https://github.com/vllm-project/vllm/blob/main/docs/cli/README.md) | Latest page dated 2026-07-21；prerequisite Linux/Python 3.10–3.13，页面也提供 Apple Silicon path | Python、HTTP/curl、模型下载；CUDA 或 Apple Silicon/Metal 更适合运行 | `vllm serve` 启动小模型；curl `/v1/chat/completions`；用 `vllm bench latency/serve/throughput` 记录 TTFT、TPOT/ITL、P50/P95/P99、tokens/s、GPU memory；保存命令和环境 |
| LR-017 | KV cache、PagedAttention、serving memory fragmentation 与吞吐/延迟 trade-off | [PagedAttention 原始论文（ACM SOSP 2023）](https://doi.org/10.1145/3600006.3613165)；[vLLM 官方 quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) | 论文 2023；实现持续更新 | LR-003、LR-016 | 先用普通 HF generate 做 baseline，再用 vLLM；改变 concurrency、prompt/output length，解释为什么“高吞吐”不等于“单请求低延迟” |
| LR-026 | FlashAttention/IO-aware exact attention、kernel 与 profiling | [FlashAttention 原始论文（NeurIPS 2022）](https://papers.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)；[官方实现](https://github.com/Dao-AILab/flash-attention) | 论文 2022；仓库 main 持续更新 | LR-003、GPU memory hierarchy、Triton 基础可后补 | 用短序列做 naive attention vs SDPA/FlashAttention 数值对齐；用 PyTorch Profiler 记录 HBM/运行时间。CS336 A2 的 Triton 实现是进阶项，不是第一周任务 |

### D. RAG、Agent、Memory 与评测

| ID | 适用主题 | 官方来源（原文/代码） | 版本或日期（检索时） | 先修 | 代码练习与验收 |
|---|---|---|---|---|---|
| LR-018 | RAG 的 parametric/non-parametric memory、dense retrieval、provenance | [RAG 原始论文（NeurIPS 2020）](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | NeurIPS 2020，概念稳定 | embedding、检索指标、生成模型基础 | 对仓库中的小文档集合实现 BM25、dense、RRF；报告 Recall@K/MRR/NDCG、answer faithfulness、失败 query，不把“用了向量库”当作 RAG 完成 |
| LR-019 | Agent fundamentals、tools、thought/action/observation、smolagents/LlamaIndex/LangGraph、agentic RAG、evaluation | [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)；[Agentic RAG unit](https://huggingface.co/learn/agents-course/unit3/agentic-rag/agentic-rag) | Live course；课程建议每章每周约 3–4 小时 | 基础 Python、基础 LLM；需要 HF account 做 Spaces/Hub hands-on | 完成 Unit 1 fundamentals；Unit 3 做 agentic RAG；Bonus Unit 2 做 observability/evaluation；把 tool call、参数、失败和轨迹保存为 JSONL |
| LR-020 | Durable execution、checkpoint、human-in-loop、memory、time travel、fault recovery | [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)；[persistence docs](https://docs.langchain.com/oss/python/langgraph/persistence) | Current OSS Python docs；未在 URL 固定版本 | Python、状态机/异步基础 | 用 `StateGraph` + checkpointer 实现 `PLANNING/TOOL_PENDING/RETRY/WAITING_HUMAN/COMPLETED/FAILED`；故障注入后 resume；测试重复 resume 的幂等性 |
| LR-021 | Agent tools、handoff、guardrail、session、trace、human-in-loop | [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)；[Quickstart](https://openai.github.io/openai-agents-python/quickstart/)；[Tracing](https://openai.github.io/openai-agents-python/tracing/) | Current docs；安装命令为 `pip install openai-agents`，运行需 API key | Python、函数调用、JSON schema；API key 与费用控制 | 做一个最小 tool agent；给工具增加 Pydantic/参数验证、最大步数、timeout/retry；比较 trace 中 LLM/tool/handoff/guardrail span。不要把含敏感数据的 trace 默认上传 |
| LR-022 | metric、comparison、measurement、evaluator、数据集评估 | [Hugging Face Evaluate quick tour](https://huggingface.co/docs/evaluate/a_quick_tour)；[installation](https://huggingface.co/docs/evaluate/installation) | Current docs；installation 页注明 Python 3.7+ tested | Python、分类/生成指标 | 用 `evaluate.load` 跑 exact match、accuracy；为面试问答集写 custom metric：定义、机制、trade-off、代码正确性、线上风险；把 metric version 和数据快照写入报告 |
| LR-023 | LLM benchmark、zero/few-shot、custom task、可重现 YAML、vLLM backend | [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)；[new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md) | Repo main；release 页面检索时显示 v0.4.12（2026-05-11）；必须 pin commit | Python 3.9+、模型 backend；本地多 GPU 需另看 Accelerate/vLLM | 先跑 10 条小任务；再把 10～20 道去重面试题写成 YAML task；保存 task config、model args、commit、sample output；防止数据泄漏和把训练题当测试题 |
| LR-024 | coding/agent/reasoning/knowledge/multimodal eval、安全 sandbox、日志分析 | [UK AI Security Institute Inspect](https://inspect.aisi.org.uk/?lang=en-US)；[agent evaluations](https://inspect.aisi.org.uk/agents.html) | Current docs；页面列出 200+ pre-built evaluations | Python、模型 provider 或本地 HF/vLLM；不可信代码要 sandbox | 写一个 `Task` + solver + scorer；先跑小样本，查看 eval log/transcript；为 Agent 增加 tool-use correctness、loop rate、retry rate、latency scorer |
| LR-025 | 多维、透明、可复现的 foundation-model benchmark | [Stanford HELM repo](https://github.com/stanford-crfm/helm)；[HELM site](https://crfm.stanford.edu/helm/index.html) | Repo 明确写明 2026-06-01 起进入 maintenance mode | 基础 benchmark、模型 API/本地服务 | 只把 HELM 当作 benchmark 设计参考或复现历史结果；若运行，固定 suite/schema/model version。不要把 maintenance-mode leaderboard 当作当前市场排名 |

### E. 分布式理论补充

| ID | 适用主题 | 官方来源 | 版本或日期（检索时） | 先修 | 代码练习与验收 |
|---|---|---|---|---|---|
| LR-027 | ZeRO 的参数/梯度/optimizer state 分片与内存推导 | [ZeRO 原始论文](https://arxiv.org/abs/1910.02054)；[DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) | 论文 2019；DeepSpeed tutorial 2026-08 更新 | DDP、optimizer state、collective communication | 为 1 个 toy Transformer 估算 DP/FSDP/ZeRO-1/2/3 的参数、梯度、Adam state；再用 tiny run 验证每 rank memory 变化 |

## 2. 6 周快速学习/代码路线

每周 6 天、每天 60–120 分钟：`30% 阅读 + 50% 写代码 + 20% 口述/复盘`。每个周末必须留下一个可运行 commit 和一页结果记录。

| 周 | 知识主线 | 必做代码 | 面试输出 |
|---|---|---|---|
| Week 1 | PyTorch、tensor shape、autograd、训练循环、Attention | LR-001 + LR-003；完成 MHA/CE/optimizer 小实现 | 30 秒解释 Q/K/V、`sqrt(d_head)`、causal mask；3 分钟手推 shape/复杂度 |
| Week 2 | Transformer、tokenizer、Trainer、SFT、LoRA | LR-004 A1（缩小版）+ LR-006～LR-008 | 能解释 answer-only loss、padding、LoRA target module、full FT trade-off |
| Week 3 | DPO、PPO/GRPO、reward、KL、policy lag | LR-009～LR-012；完成 Toy GRPO | 给出 prompt→sample→reward→advantage→ratio→clip/KL 数据流；构造并解释 reward hacking |
| Week 4 | KV cache、PagedAttention、FlashAttention、serving | LR-016、LR-017、LR-026；完成 serving benchmark | 区分 TTFT/TPOT/P95/吞吐；解释 KV cache memory、continuous batching、IO-awareness |
| Week 5 | RAG、Agent、Memory、durability | LR-018～LR-021；完成 Hybrid RAG + durable agent | 能画 retrieval/agent state graph；说明 retry、idempotency、checkpoint、RAG vs SFT |
| Week 6 | Eval、DDP/FSDP2/ZeRO、综合模拟 | LR-013～LR-015、LR-022～LR-025；跑小型 eval/regression | 2 场 45 分钟模拟面试；回答“指标—实现—失败—线上恢复—trade-off”闭环 |

### 每日固定模板

1. 10 分钟：回忆前一天的 3 个公式/数据流，不看答案。
2. 25–40 分钟：只读一个资源小节，并写 5 行自己的解释。
3. 30–60 分钟：实现或修改一个最小代码单元；先写 shape/输入输出测试。
4. 10–15 分钟：录音回答一题 30 秒版和一题 3 分钟版。
5. 5 分钟：记录 `unknowns`、失败日志、下一步；把“看懂但没跑通”的内容标为 `unverified`。

## 3. 实验与环境的可复现规则

- 每个实验建独立环境；优先使用 `uv`/`venv`，不要把系统 Python 当项目环境。
- 每个实验记录：Python、PyTorch、Transformers、TRL、Accelerate、vLLM/DeepSpeed 版本；GPU/驱动/CUDA；OS；命令；随机种子；模型和数据的 Hub revision；git commit。
- 先用 tiny model、少量数据、CPU/MPS 或单 GPU 做 correctness smoke test，再上多 GPU/云 GPU 做性能实验。没有对应硬件时只能报告“未运行/理论推导”，不能伪造 benchmark。
- 对 CUDA/多 GPU 实验，优先按 PyTorch FSDP2、DeepSpeed、vLLM 当前文档安装；FSDP1 已被官方教程标记 deprecated，不再把 FSDP1 作为新实验入口。
- 任何会下载模型、调用 API、上传 trace 或执行不可信工具代码的实验，都要先检查许可、费用、隐私和 sandbox；API key 只能通过环境变量或 secret manager 注入。
- 代码验收至少包括：小样本数值检查、shape/边界测试、异常路径测试、一个可解释指标；性能结果必须同时报告 workload（prompt/output length、batch/concurrency、precision）。

## 4. 资源治理与更新规则

1. 资源只负责指向官方事实，不把第三方博客或聚合榜单写成“权威答案”。
2. URL 能访问不代表版本可复现；每季度检查一次链接，并把版本化 URL/commit 写入 `meta/update-ledger.md`。
3. 论文用于解释算法为什么成立；官方代码用于解释 API/实现如何运行；面经只证明问题出现过，不能替代论文或官方文档。
4. 新资源进入本文件前，必须填写：主题、primary URL、发布日期/版本、先修、可执行练习、预期验收、与当前专题的映射。
5. 同一个工具只保留一个推荐入口和一个备用入口，避免学习计划被框架切换拖慢；例如 serving 先 vLLM，分布式先 PyTorch FSDP2，再补 DeepSpeed/Megatron 对比。
6. 将来若把稳定知识晋升到 `knowledge`，只晋升经过实验或官方来源交叉验证的概念/模式；版本、命令、面经样本和个人错题留在 `job_interview`。

## 5. 事实边界与待确认项

- 本研究检索的是官方课程、文档、论文和代码；没有验证课程视频是否每个地区可访问，也没有替用户购买云 GPU 或课程。
- `latest/main` 页面可能在下一次检索后改变；本文件的日期不是未来实验的版本锁定，实验必须重新记录 commit/版本。
- vLLM、FSDP2、TRL、Agents SDK、LangGraph 的 API 变化较快，面试中应同时会讲原理和当前实现，但回答具体参数时必须说明版本。
- HELM 已进入 maintenance mode；它适合作为透明 benchmark 的参考和历史复现，不适合作为唯一的最新评测入口。
