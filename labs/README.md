# 可运行 Labs

这里是把教材里的机制变成“能运行、能测试、能解释”的最小实验。先跑已实现的实验，再回到对应专题和面经做口头回答。

## 当前可运行

| 能力 | 入口 | 运行命令 |
|---|---|---|
| 算法 / 数学基础 | [`foundations/README.md`](./foundations/README.md) | `.venv/bin/python -m unittest labs/foundations/test_algorithms.py labs/foundations/test_math_ops.py -v` |
| Multi-Head Attention | [`attention/README.md`](./attention/README.md) | `.venv/bin/python -m unittest labs/attention/test_mha.py -v` |
| Transformer Block / RoPE / GQA | [`attention/README.md`](./attention/README.md) | `.venv/bin/python -m unittest labs/attention/test_transformer_block.py -v` |
| Answer-only SFT loss | [`posttraining/README.md`](./posttraining/README.md) | `.venv/bin/python -m unittest labs/posttraining/test_sft_loss.py -v` |
| Toy GRPO objective | [`posttraining/README.md`](./posttraining/README.md) | `.venv/bin/python -m unittest labs/posttraining/test_grpo_toy.py -v` |
| RAG retrieval metrics / RRF | [`rag/README.md`](./rag/README.md) | `.venv/bin/python -m unittest labs/rag/test_metrics.py -v` |
| KV-cache 显存账本 | [`inference/README.md`](./inference/README.md) | `.venv/bin/python -m unittest labs/inference/test_memory.py -v` |
| Durable Agent 状态机 | [`agent/README.md`](./agent/README.md) | `.venv/bin/python -m unittest labs/agent/test_run.py -v` |

一次跑完所有实验：

```bash
.venv/bin/python -m unittest discover -s labs -p 'test_*.py' -v
```

## 环境

核心实验只依赖 Python、PyTorch 和 NumPy：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r labs/requirements-core.txt
```

vLLM、FSDP、真实模型 SFT 和 LangGraph 实验属于后续环境切片，不在本地核心依赖中强行安装；它们需要按目标机器的 CUDA、驱动和模型条件单独建立环境。

## 学习闭环

每跑完一个 Lab，必须留下：

1. 一张公式或状态图；
2. 一次测试结果和一个失败样例；
3. 30 秒、3 分钟、15 分钟三档面试回答。
