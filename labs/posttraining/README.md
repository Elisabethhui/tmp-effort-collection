# Lab｜手写最小 SFT Loss

## 目标

先不用 `Trainer` 或 `SFTTrainer`，独立完成：

1. prompt/answer 拼接后的 label mask；
2. padding mask；
3. causal language modeling 的 shift-right / next-token loss；
4. 与框架实现的输入输出契约对齐。

参考：

- [Transformers Trainer](https://huggingface.co/docs/transformers/trainer)
- [TRL](https://huggingface.co/docs/trl/)

## 运行

```bash
.venv/bin/python -m unittest labs/posttraining/test_sft_loss.py -v
```

## 验收标准

- prompt token 不产生监督 loss；
- padding token 不产生监督 loss；
- logits 与 labels 的时间维正确错位一位；
- loss 能反向传播；
- 能解释 EOS 是否纳入 loss、为什么不能把 prompt 当作目标答案。

## 下一步：Toy GRPO

`grpo_toy.py` 把下面的数据流拆成可测试函数：

```text
rewards
 → group-relative advantage
 → policy ratio
 → clipped surrogate
 → reference KL
 → loss
```

它不替代 TRL 的 `GRPOTrainer`，而是先帮助你验证公式、shape、clip 和
policy lag，再对照 [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)。
