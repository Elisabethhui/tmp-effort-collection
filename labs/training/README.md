# R4 训练闭环 Lab

这些实验把 R3 的 Transformer block 接成一个可训练、可评估、可恢复的 tiny decoder LM。默认是 `RUNNABLE_CPU`；MPS 只做可选 smoke test，不把本机结果当成 CUDA 性能结论。

## 运行入口

单独验证三个层次：

```bash
.venv/bin/python -m unittest labs/training/test_label_shift.py -v
.venv/bin/python -m unittest labs/training/test_tiny_lm.py -v
.venv/bin/python -m unittest labs/training/test_train_loop.py -v
```

运行一个完整的 toy 训练：

```bash
.venv/bin/python -m labs.training.train_tiny_lm --epochs 5 --device cpu
```

可选 MPS 烟测（先确认 PyTorch 能看到 MPS）：

```bash
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
.venv/bin/python -m labs.training.train_tiny_lm --epochs 2 --device mps
```

## 文件地图

| 文件 | 学习对象 | 关键断言 |
| --- | --- | --- |
| `label_shift.py` | `logits[:, :-1]` 对 `labels[:, 1:]`、有效 token 平均 | shift shape、mask、空目标、反向传播 |
| `tiny_lm.py` | embedding + position + R3 TransformerBlock + LM head | `[B,T] → [B,T,V]`、causal isolation、weight tying |
| `train_loop.py` | train/eval、token-normalized accumulation、checkpoint | 累积等价、loss 下降、resume 等价 |
| `train_tiny_lm.py` | 可直接运行的周期 toy corpus | CPU 训练曲线和设备标签 |

## 学习闭环

1. 先读 `learning/lessons/0004-r4-causal-loss.html`；
2. 不看实现，写出 shift 和一次 optimizer update；
3. 跑对应测试，阅读一个失败断言；
4. 再看 `0005` 和 `0006`，运行完整 toy 训练；
5. 记录训练前后 loss、一个失败样例，以及 30 秒/3 分钟/15 分钟回答。

## 证据边界

- `RUNNABLE_CPU`：shape、loss、backward、训练曲线、checkpoint/resume；
- `MPS_OPTIONAL`：同一 tiny 实验能否在 Apple Silicon 上运行；
- `SOURCE_READ`：PyTorch autograd、optimizer、serialization 和 Transformer building blocks 的 API 语义；
- `REMOTE_GPU`：mixed precision、CUDA kernel、吞吐、显存和多卡扩展，本 Lab 不声称已完成。
