# R2–R3 基础与 Transformer 教学包

这是当前阶段的完整学习入口。后续 SFT、Serving、RAG、Agent 先保留在总路线中，不在本阶段展开。

## 入口

1. 先读 [使命](./MISSION.md) 和 [课程地图](./curriculum-r2-r3.md)；
2. 再看 [资源清单](./RESOURCES.md) 的对应官方小节；
3. 运行 CPU Labs；
4. 最后做每个单元的 retrieval questions 和三档面试回答。

## 教学页与速查

- [Lesson 0001：R2 基础](./lessons/0001-r2-foundations.html)
- [Lesson 0002：Attention Shapes](./lessons/0002-r3-attention-shapes.html)
- [Lesson 0003：Transformer Block](./lessons/0003-r3-transformer-block.html)
- [R2 基础速查](./reference/foundations.html)
- [R3 Transformer 速查](./reference/transformer.html)

## R4 训练闭环

- [Lesson 0004：Causal loss 与 label shift](./lessons/0004-r4-causal-loss.html)
- [Lesson 0005：Tiny Decoder LM](./lessons/0005-r4-tiny-decoder-lm.html)
- [Lesson 0006：训练诊断与 checkpoint](./lessons/0006-r4-training-diagnostics.html)
- [R4 训练闭环速查](./reference/training-loop.html)
- [R4 训练 Labs](../labs/training/README.md)

## R2–R3 地图

```text
R2 基础底座
├── Python / DSA / complexity
├── Linear algebra / probability / numerical stability
├── Loss / gradient / SGD / Adam
├── Generalization / metrics / data leakage
└── PyTorch tensor / autograd / device / checkpoint

R3 Transformer
├── Tokenizer / embedding / decoder-only
├── QKV / shape / scaled attention / masks
├── MHA / MQA / GQA
├── Sinusoidal / learned position / RoPE
├── Residual / Pre-Norm / RMSNorm / FFN / SwiGLU
├── Causal LM / logits / label shift / weight tying
├── KV Cache / prefill-decode / complexity
└── SDPA / Flash / Paged attention 源码阅读
```

## 验收入口

```bash
.venv/bin/python -m unittest discover -s labs -p 'test_*.py' -v
```

GPU 内容的完成状态必须写成 `RUNNABLE_CPU`、`MPS_OPTIONAL`、`SOURCE_READ` 或 `REMOTE_GPU`，不使用模糊的“已完成”。
