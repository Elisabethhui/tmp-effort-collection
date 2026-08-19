# 12｜大模型分布式训练：DDP / FSDP / ZeRO / TP / PP / CP

> 2026 面试信号：字节直接问 PyTorch 分布式函数、FSDP vs ZeRO、NCCL 原语，以及 ZeRO-3 单卡放不下参数时如何完成 forward/update；美团也持续问 ZeRO-2/3 和大模型显存估算。

---

# 1. 先从显存账本开始 `S`

训练显存不是“参数量 × 2 bytes”这么简单。

典型组成：

```text
Parameters
Gradients
Optimizer States
Master Weights (视精度/优化器)
Activations
Temporary Buffers
Communication Buckets
Allocator Fragmentation
```

以 Adam + mixed precision 为例，optimizer states 往往比参数本身更占显存。

**面试第一原则：** 估算前先声明 dtype、optimizer、是否 mixed precision、是否 checkpointing、是否 sharding。

---

# 2. Data Parallel `S`

## 2.1 基本思想

每张 GPU 一份模型，不同数据 shard：

```text
GPU0 model + batch0
GPU1 model + batch1
GPU2 model + batch2
       ↓
  gradient all-reduce
       ↓
same updated model
```

## 2.2 DDP

PyTorch DistributedDataParallel 常用 NCCL 做 GPU 通信。

### 高频追问

- 为什么 DDP 比 DataParallel 好？
- gradient 什么时候 all-reduce？
- bucket 的作用？
- `no_sync()` 什么时候用？gradient accumulation 时。

---

# 3. NCCL / Collective Communication `S/A`

2026 字节面经直接点名 communication primitives。

需要理解：

## All-Reduce

每卡都有 input，最终每卡得到聚合结果。DDP gradient 同步核心。

## All-Gather

每卡持有 shard，最终每卡拿到所有 shard。

## Reduce-Scatter

先 reduce，再把结果分 shard 发给各 rank。

## Broadcast

一个 rank 发给所有 rank。

## Gather / Scatter

汇总到 root / 从 root 分发。

### 为什么 ring all-reduce 常见？

把数据切块，通过 ring 上的 reduce-scatter + all-gather 充分利用带宽，避免单中心瓶颈。

---

# 4. ZeRO `S`

核心思想：**不要让每张 GPU 都复制所有 model states。**

## ZeRO-1

shard optimizer states。

## ZeRO-2

再 shard gradients。

## ZeRO-3

再 shard parameters。

### 面试表述

```text
Stage 1: O
Stage 2: O + G
Stage 3: O + G + P
```

其中 O=optimizer states, G=gradients, P=parameters。

---

# 5. ZeRO-3 最关键挖坑：参数都分片了，forward 怎么算？ `🔥`

2026 字节直接问类似问题。

答案不能说“每卡只算自己的参数”。一个 layer 执行前，参与该层计算的参数需要通过通信临时 materialize / all-gather 到需要的 rank；计算后再释放或保持，根据实现策略决定。

因此 ZeRO-3 的本质是：

```text
storage = sharded
compute moment = materialize needed parameters
then free/reshard
```

**trade-off：** 显存省，但通信更多。

---

# 6. FSDP vs DeepSpeed ZeRO `S/A`

两者目标相近：full sharding。

## FSDP

PyTorch 原生生态，按 wrapped module 管理 parameter sharding / all-gather / reduce-scatter。

## ZeRO

DeepSpeed 更完整地覆盖 training engine、offload 等生态。

### 面试不应回答

“FSDP 是 PyTorch 的 ZeRO-3，所以完全一样。”

接口、execution scheduling、state dict、prefetch/offload、mixed precision、生态集成都有差别。面试应说**思想相近但实现/运行时不同**。

---

# 7. Tensor Parallelism `S/A`

当单个 layer 的参数/计算也放不下一张卡，就需要 model parallel。

## Column Parallel Linear

沿输出维切 `W`。

## Row Parallel Linear

沿输入维切 `W`，通常需要 reduce。

Transformer 中 Attention/MLP 都可做 TP。

### 通信代价

TP 通信频繁，对 NVLink/NVSwitch/高速互联依赖高。

---

# 8. Pipeline Parallelism `A`

不同 layers 放不同 GPU/stage。

问题：pipeline bubble。

解决思路：micro-batch，1F1B schedule 等。

### 高频追问

- PP 为什么会有 bubble？
- microbatch 太少会怎样？
- stage 不平衡怎么办？

---

# 9. Sequence / Context Parallelism `A/B`

长上下文让 activation/attention 沿 sequence 维成为瓶颈。

Context Parallel 将 sequence token 分到多个 device，并设计 attention 所需的 KV 通信。

2026 长上下文岗位值得掌握：

- sequence parallel；
- context parallel；
- ring attention 类思想；
- Ulysses/all-to-all 类型方案（按框架实现理解）。

---

# 10. Expert Parallelism `A/B`

MoE 模型把不同 experts 放在不同 GPU。

核心问题：router 之后 token 要通过 all-to-all 发送到目标 expert。

工程难点：

- load imbalance；
- all-to-all communication；
- token dropping / capacity；
- expert placement。

---

# 11. 3D Parallelism `A`

实际超大模型通常组合：

```text
DP × TP × PP
```

MoE 再加 EP；长 context 再考虑 CP。

**面试重点不是背名字，而是根据瓶颈选切分维度。**

---

# 12. Gradient Accumulation `S`

如果 micro-batch 太小：

```text
global_batch = micro_batch × grad_accum_steps × data_parallel_size
```

### 常见坑

- loss 是否要除 accumulation steps；
- scheduler step 是 optimizer step 还是 micro-step；
- DDP accumulation 时避免每个 micro-step 都 all-reduce (`no_sync`)。

---

# 13. Activation Checkpointing `S`

不保存部分 forward activation，backward 时重算。

本质：**compute 换 memory**。

与 ZeRO 不同：

- ZeRO 主要省 model states；
- checkpointing 主要省 activations。

---

# 14. CPU/NVMe Offload `A/B`

把 optimizer/parameter 等移到 CPU/NVMe，进一步省 GPU memory。

代价：PCIe/IO latency，容易成为吞吐瓶颈。

---

# 15. Mixed Precision：FP16 / BF16 / FP8 `S/A`

## FP16

mantissa 较多但 exponent range 小，容易 overflow，需要 loss scaling。

## BF16

exponent 与 FP32 接近，range 大，训练通常更稳。

## FP8

更激进的吞吐/显存优化，需要 scale/format 管理和硬件支持。

### 2026 美团问 BF16 vs FP16

建议从：range、precision、loss scaling、hardware support、training stability 回答。

---

# 16. Distributed Checkpoint `A`

不能把 checkpoint 理解成 `torch.save(model.state_dict())`。

大模型场景需要：

- sharded state；
- optimizer state；
- RNG state；
- data loader progress；
- scheduler；
- topology portability；
- atomic commit；
- corruption detection。

---

# 17. 故障恢复 `A`

真实多机训练常见：

- worker crash；
- NCCL timeout；
- network partition；
- OOM；
- corrupted checkpoint；
- straggler。

要设计：

- frequent checkpoint；
- elastic launch（如果适用）；
- retry；
- metric/log persistence；
- data cursor resume。

---

# 18. 性能分析 `A`

先算 MFU/吞吐只是开始，还要定位：

- compute-bound；
- memory bandwidth-bound；
- communication-bound；
- input pipeline-bound；
- imbalance。

工具：PyTorch Profiler / Nsight Systems / framework profiler。

---

# 19. 2026 真实面试拆解

## 字节大模型算法二面｜2026-02

问题：

- PyTorch distributed functions；
- FSDP vs ZeRO；
- NCCL gather/scatter 等；
- 一张卡放不下模型怎么办；
- ZeRO-3 参数更新/计算如何完成。

**这说明只背 ZeRO 1/2/3 表格不够，必须知道通信和 runtime。**

## 美团北斗｜2026-08-06

- ZeRO-2 vs ZeRO-3；
- LLaMA-7B 显存估算；
- BF16 vs FP16；
- serving latency/throughput。

---

# 20. 高频等级

## S

- DDP
- all-reduce/all-gather/reduce-scatter
- ZeRO 1/2/3
- FSDP vs ZeRO
- gradient accumulation
- activation checkpointing
- FP16/BF16
- memory estimation

## A

- TP/PP
- NCCL
- checkpoint/resume
- offload
- profiler

## B

- CP
- EP
- FP8
- advanced topology/scheduling

---

# 21. 最小实验

1. 两 GPU DDP 训练并验证参数一致；
2. gradient accumulation + `no_sync`；
3. FSDP 小模型观察每卡显存；
4. DeepSpeed ZeRO-2/3 对比 memory/throughput；
5. 人为 kill worker，验证 checkpoint resume；
6. 用 profiler 看 communication overlap。

---

# 22. 自测

- [ ] 能算 global batch size
- [ ] 能解释 all-reduce 和 reduce-scatter
- [ ] 能解释 ZeRO-3 forward 为什么仍能拿到完整 layer parameter
- [ ] 能比较 FSDP/ZeRO，而非说完全一样
- [ ] 能解释 TP/PP/DP 适合什么瓶颈
- [ ] 能给 7B/70B 做显存估算并声明假设
- [ ] 能解释 BF16 比 FP16 稳在哪里
