# R9：评测与分布式训练

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；多卡性能与通信实验标为 `REMOTE_GPU`。
> 目标：先建立可信的评测和显存/通信账本，再读 DDP、FSDP、ZeRO、NCCL 的实现。

## 1. Mission

模型项目常见的失败不是“不会跑”，而是无法证明变好了：评测集泄漏、只报平均分、Agent 轨迹未评分、训练吞吐没有硬件基线。R9 将两条线合并为工程能力：

```text
eval set/version → task + rubric → model/trajectory run
                 → quality/safety/grounding/latency/cost
                 → regression gate + error analysis

model state → data parallel / sharding → communication + memory ledger
```

## 2. Prerequisites

- R2：统计、抽样、数据切分、指标和置信区间基本概念；
- R4：训练/验证 loop、checkpoint、梯度和优化器；
- R6/R8：推理指标、服务 trace、Agent 轨迹；
- 不要求本机有多 GPU；先用伪分布式、源码和账本回答问题。

## 3. Learning outcomes

完成本包后能够：

1. 为生成模型、RAG 和 Agent 分别设计可复现的 eval case、rubric、版本和失败分类；
2. 区分 correctness、helpfulness、faithfulness、safety、robustness、latency、cost；
3. 用 bootstrap/分桶/置信区间解释分数变化，避免把噪声当提升；
4. 解释 DDP、FSDP、ZeRO 的状态分片对象，以及通信/显存 trade-off；
5. 从 `world_size`、参数、梯度、optimizer state 和 activation 粗算单卡账本；
6. 说清本地 CPU/source read 与远端多卡 benchmark 的证据边界。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 Eval contract | case、input、reference、rubric、version | 没有固定契约就无法回归 | dataset hash、model/prompt/config lineage | 写一个 10 case eval manifest |
| U2 Rubric/trajectory | exact match、LLM judge、tool trace、grounding | 文本质量和路径质量不同 | scoring schema、pairwise/pointwise、judge calibration | 给三条答案分档并解释理由 |
| U3 Statistics/regression | bootstrap、CI、slice、threshold | 小提升可能是噪声 | paired comparison、分桶、显著性与最小提升 | 计算一次 paired delta |
| U4 Production eval | latency、cost、safety、drift、human review | 离线分数不等于线上体验 | trace sampling、red team、canary、rollback | 设计发布门禁 |
| U5 DDP | replicated model、all-reduce、global batch | 最基本的数据并行机制 | 每 rank 处理 shard，梯度同步 | 画两 rank 一步通信 |
| U6 FSDP/ZeRO | parameter/gradient/optimizer sharding | 单卡放不下完整状态时如何训练 | shard、all-gather、reduce-scatter、offload | 做内存/通信账本 |
| U7 NCCL/system | collective、拓扑、带宽、straggler | 多卡瓶颈常在通信与系统 | ring/tree、overlap、bucket、failure | 读日志判断通信异常 |

### 显存账本

先分开计算参数、梯度、optimizer state、activation、通信 buffer 和 KV Cache；再说明 dtype、checkpointing、sharding 和 offload 对每一项的影响。估算是面试模型，不是 benchmark 结果，所有常数和临时 workspace 都要标注假设。

## 5. Mac validation lane

- `RUNNABLE_CPU`：运行本地评测集、分桶统计、paired comparison、trace scorer 和 memory ledger。
- `SOURCE_READ`：阅读 PyTorch distributed/FSDP2、DeepSpeed ZeRO 和 NCCL 官方概念/源码入口，画 collective 调用链。
- `MPS_OPTIONAL`：只做单设备模型/评测 smoke test；MPS 不代表 NCCL 多卡环境。
- `REMOTE_GPU`：多卡 DDP/FSDP/ZeRO、通信 overlap、吞吐和故障注入只能在匹配硬件环境中验证。

## 6. Planned labs（本包后续实现）

1. `labs/eval/eval_manifest.py`：生成带 hash、版本、rubric 和 expected evidence 的评测清单；
2. `labs/eval/trajectory_scorer.py`：按最终答案、引用、工具路径和安全分项评分；
3. `labs/eval/regression_stats.py`：paired delta、bootstrap CI、slice report 和 threshold gate；
4. `labs/distributed/memory_ledger.py`：参数/梯度/optimizer/activation/通信账本；
5. `labs/distributed/collective_sim.py`：用 CPU 模拟 all-reduce、all-gather、reduce-scatter 的数据流。

## 7. Failure modes

1. **评测集污染/泄漏**：训练、调参、公开题和最终测试集合交叉；
2. **只报平均分**：长尾、困难 slice、安全失败被平均数掩盖；
3. **LLM judge 无校准**：rubric 含糊、位置偏置、judge 与被评模型同源；
4. **回归门禁过度敏感**：小样本噪声触发回滚，或阈值太松漏掉安全问题；
5. **显存估算漏项**：只算参数，不算梯度、optimizer、activation、通信 buffer；
6. **把 DDP 当 FSDP**：DDP 复制完整模型，FSDP/ZeRO 对状态分片且通信模式不同；
7. **本地伪分布式冒充多卡实测**：必须明确 `SOURCE_READ` 或 `REMOTE_GPU`。

## 8. Interview rehearsal

- **30 秒**：说明一个可信评测需要哪些元数据和指标。
- **3 分钟**：从 eval set 到 regression gate，解释为什么要分桶、配对和置信区间。
- **15 分钟白板/代码**：估算 7B 训练状态显存，比较 DDP/FSDP/ZeRO 的通信和失败模式。

推荐 retrieval questions：

1. 为什么两个版本平均分差 0.3 不能直接宣布提升？
2. DDP、FSDP、ZeRO 分别复制/切分什么状态？
3. 怎样为 Agent 的“工具路径正确但最终文本错误”单独评分？

## 9. Acceptance gate

- [ ] 写出一份带版本、rubric、evidence 和 slice 的评测 manifest；
- [ ] 对 toy runs 计算 paired delta/CI，并解释一次误报或漏报；
- [ ] CPU 模拟至少一个 collective，并完成显存/通信账本；
- [ ] 读过 PyTorch FSDP/DeepSpeed ZeRO/NCCL 的关键入口并标注证据级别；
- [ ] 能完成三档面试回答，且不把估算写成实测；
- [ ] 通过 gate 后，再把稳定评测规范晋级到 `knowledge/`。

## 10. Primary sources

- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)
- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Inspect AI](https://inspect.aisi.org.uk/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
