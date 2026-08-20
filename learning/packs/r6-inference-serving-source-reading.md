# R6：推理与 Serving 源码阅读

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；真正吞吐与显存实验标为 `REMOTE_GPU`。
> 目标：先把生成、KV Cache 和调度的机制读懂，再讨论 vLLM、量化和服务优化。

## 1. Mission

“KV Cache 怎么省算力”“Paged Attention 为什么像虚拟内存”“TTFT 和 TPOT 有什么区别”是高频面试题，也最容易把 GPU 专有实现背成口号。R6 采用源码优先的路径：

```text
tokenizer → prefill → cache K/V
         → decode one token → append/reuse cache
         → scheduler/batching → stream response → metrics
```

本包要建立的是可迁移的内存、shape、复杂度和调度模型，不要求 Mac 复现 CUDA kernel。

## 2. Prerequisites

- R3：MHA/MQA/GQA、RoPE、causal mask、`[B,H,T,D]` shape；
- R4：logits、generation loop、checkpoint 和基本 profiling；
- R5 可并行，但不要求先会部署大模型；
- 愿意阅读官方 Python/C++/CUDA 入口，并把关键路径改写成 CPU reference。

## 3. Learning outcomes

完成本包后能够：

1. 逐步解释 prefill 与 decode 的计算差异，写出 greedy/sampling generation loop；
2. 计算 MHA/GQA 的 KV Cache 形状、每 token 字节数和上下文长度影响；
3. 解释 contiguous cache、block/page cache、prefix caching 和 block reuse 的取舍；
4. 区分 batch size、concurrency、TTFT、TPOT、throughput、P95 latency；
5. 读懂 Hugging Face cache 与 vLLM V1 KV cache 入口，明确哪些结论只是 `SOURCE_READ`。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 Generation loop | greedy、temperature、top-k/p、stop criteria | 生成是推理入口，不是黑盒 `generate()` | 每轮 logits → 采样 → append token → stop | 用 toy logits 手算两轮输出 |
| U2 Prefill/decode | 一次处理 prompt 与逐 token 处理 | 解释首 token 和后续 token 的延迟差 | prefill 建 cache；decode 复用历史 K/V | 画两阶段 shape/复杂度图 |
| U3 KV Cache | 每层 K/V、layout、dtype、batch/sequence | cache 常是长上下文瓶颈 | `[B,H_kv,T,D]`，新 token 追加，Q 只取当前步 | 计算 MHA 与 GQA 的内存比 |
| U4 Attention backend | SDPA、Flash/融合 kernel、mask 约束 | API 相同不等于实现相同 | backend selection、dtype、causal flag、fallback | 对同一输入比较 reference 与 SDPA |
| U5 Paged/Block cache | block table、free list、prefix reuse | 动态请求长度会造成碎片与浪费 | 固定 block 管理物理 KV，逻辑 token 映射到 block | CPU 模拟分配/释放和命中 |
| U6 Batching/scheduling | continuous batching、并发、抢占 | 服务优化是调度与内存的共同问题 | request state、等待队列、预算、preemption | 为 3 个请求写调度时间线 |
| U7 量化与指标 | weight/KV quantization、TTFT/TPOT/P95 | 速度、质量、内存必须一起报告 | 精度/带宽/解码步数/队列等待拆账 | 设计一张 benchmark 记录表 |

### Cache 账本

对每层每个 token，近似缓存字节数为：

\[
\text{bytes/token/layer} \approx 2 \times H_{kv} \times D_{head} \times \text{bytes(dtype)},
\]

其中 2 表示 K 和 V。总量还要乘以层数、batch/并发和已缓存 token 数。这个公式只用于容量账本；实际 layout、对齐、压缩和临时 workspace 需要以实现和测量为准。

## 5. Mac validation lane

- `RUNNABLE_CPU`：写一个小模型的 reference generation、KV cache append/read、block allocator 和 prefix hit/miss 模拟。
- `SOURCE_READ`：按调用链阅读 Hugging Face `cache_utils.py`、cache explanation、vLLM V1 `KVCacheManager`/`BlockPool`/prefix caching design。
- `MPS_OPTIONAL`：在极小模型上验证 attention 数值和设备迁移；不把 MPS 结果外推成 CUDA kernel 性能。
- `REMOTE_GPU`：只有远端 GPU 才做 vLLM benchmark、paged kernel、量化吞吐和多并发 P95；必须另存硬件与版本。

## 6. Planned labs（本包后续实现）

1. `labs/inference/generation_reference.py`：CPU greedy/top-k 生成，支持显式 stop criteria；
2. `labs/inference/kv_cache_reference.py`：实现无 cache 与 cache 两条路径，断言 token-by-token logits 一致；
3. `labs/inference/cache_memory_ledger.py`：按层数、head 数、GQA、dtype 和上下文长度计算缓存账本；
4. `labs/inference/paged_cache_sim.py`：模拟 block allocation、释放、prefix reuse 和碎片；
5. `labs/inference/serving_metrics.py`：从事件时间戳计算 TTFT、TPOT、吞吐和 P50/P95。

## 7. Failure modes

1. **cache 结果与无 cache 不一致**：位置编码 offset、mask、K/V 维度或 batch 复用错误；
2. **把 KV Cache 当成“减少所有计算”**：它主要避免重复计算历史 K/V，当前 query 和输出投影仍需计算；
3. **MHA/GQA 内存估算错误**：把 query heads 当成 KV heads，或忘记 K/V 两份与 dtype 字节数；
4. **Paged cache 命中率低**：prefix 不一致、block 对齐、租约/释放和 cache key 设计有问题；
5. **平均延迟很好但用户体验差**：队列等待或长尾被均值掩盖，应看 TTFT、TPOT、P95/P99；
6. **盲目宣称 GPU 优化完成**：没有硬件、版本、数据规模和 benchmark 证据时只能标 `SOURCE_READ`。

## 8. Interview rehearsal

- **30 秒**：说清 prefill/decode、KV Cache 的收益和一个代价。
- **3 分钟**：从 `[B,H_kv,T,D]` 推导 cache 内存，并解释 GQA 为什么降低 K/V 存储。
- **15 分钟白板/代码**：手写 cache generation，设计 paged allocator 和 TTFT/TPOT 统计，最后指出 GPU 验证边界。

推荐 retrieval questions：

1. 为什么第一次 token 慢、后续 token 的计算模式不同？
2. Prefix caching 与普通 KV Cache 的复用边界分别是什么？
3. 一个服务的 TTFT 变差但 TPOT 不变，优先查哪里？

## 9. Acceptance gate

- [ ] 不看资料画出 prefill/decode 和 cache shape；
- [ ] CPU reference 与无 cache 路径的 logits 一致；
- [ ] 手算 MHA/GQA/dtype 下的内存账本；
- [ ] 完成 vLLM/HF 源码调用链笔记，标注 `SOURCE_READ` 而非 GPU 实测；
- [ ] 能用事件时间线解释 TTFT、TPOT 和 P95；
- [ ] 通过三档面试回答后，才将稳定结论晋级到 `knowledge/`。

## 10. Primary sources

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- [Hugging Face cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Transformers `cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [vLLM V1 KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [vLLM V1 BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [vLLM prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
