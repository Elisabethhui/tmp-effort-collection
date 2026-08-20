# R4：训练闭环与 Decoder LM

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU`；`MPS_OPTIONAL` 只做烟测。
> 目标：把 R3 的 Transformer block 接成一个能训练、能诊断、能恢复的最小语言模型。

## 1. Mission

面试中常问的“训练为什么不收敛”“label 为什么要右移”“`model.train()` 和 `eval()` 有什么区别”，本质上都在考训练闭环，而不是某个 Trainer API。R4 的目标是让你从一批 token 出发，手动追踪：

```text
dataset → batch → logits → shifted labels → loss
       → backward → optimizer step → checkpoint → eval
```

完成后，应该能用一个几十万参数以内的 tiny decoder LM 解释每一步，并能定位至少三类训练故障。

## 2. Prerequisites

- R2：张量、矩阵乘法、交叉熵、softmax、梯度、SGD/Adam、过拟合与数据泄漏；
- R3：`[B, T] → [B, T, D] → [B, T, V]`、causal mask、残差、RMSNorm、SwiGLU、label shift；
- 不要求 CUDA；不要求先会 Hugging Face Trainer。

## 3. Learning outcomes

完成本包后能够：

1. 从 logits 和 labels 手工写出 causal LM loss，并解释为什么最后一个位置没有下一个 token；
2. 设计可复现的 train/eval loop，控制随机种子、梯度清零、梯度裁剪、累积步数和 checkpoint；
3. 判断 loss 不降、NaN、过拟合、验证集恶化分别更可能来自数据、数值、优化器还是模型；
4. 解释 gradient accumulation、学习率 warmup、weight decay、early stopping 的取舍；
5. 把“本机 CPU 正确性”和“远端 GPU 性能”分开陈述。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 数据与 batch | token stream、window、padding、mask、train/valid split | 训练信号先由数据定义 | 同一文本的切窗、`input_ids`/`labels`、有效 token mask | 画出一个 `[B,T]` batch 并指出泄漏位置 |
| U2 Causal LM objective | next-token prediction、label shift、ignore index | 训练目标决定 logits 与标签的对齐 | `logits[:, :-1]` 对 `labels[:, 1:]`，再做 masked CE | 用 2×5 toy batch 手算一个位置 |
| U3 Forward/backward | logits、loss、梯度、参数更新 | 闭环必须可逐步观察 | `zero_grad → forward → loss → backward → clip → step` | 打印一层参数更新前后差值 |
| U4 Optimizer 与调度 | AdamW、warmup、decay、梯度累积 | 面试追问通常落在稳定性和吞吐 | micro-batch 累积后再 step；区分 decoupled weight decay | 比较累积 4 步与单 batch 的等价条件 |
| U5 train/eval 与 checkpoint | dropout、norm、保存/恢复、best checkpoint | 训练结果需要可复现和可回滚 | `state_dict`、optimizer state、step、rng、配置一起保存 | 中断后恢复并复现下一次 loss |
| U6 诊断与监控 | loss 曲线、grad norm、参数/激活统计 | 只看最终 loss 无法定位故障 | 先缩小数据，再检查 shape、dtype、范围、梯度和 split | 对三个故障样例写诊断顺序 |

### 最小公式卡

给定 `x = [x_0, …, x_{T-1}]`，模型输出 `z_t = f(x_≤t)`，目标是：

\[
\mathcal{L} = -\frac{1}{|M|}\sum_{t=0}^{T-2} M_t \log p(x_{t+1}\mid x_{\le t}),
\quad p=\operatorname{softmax}(z_t).
\]

这里 `M` 是有效 token mask；padding、被屏蔽的 prompt 位置或越界位置不能悄悄进入平均分母。面试时必须说明“平均的是有效 token，而不是盲目平均所有 `[B,T]` 位置”。

## 5. Mac validation lane

- `RUNNABLE_CPU`：用字符级或小词表 tiny corpus 让模型过拟合 8–32 条样本；验证 loss、梯度、checkpoint 和 label shift。
- `MPS_OPTIONAL`：仅在 `torch.backends.mps.is_available()` 时跑同一 tiny 实验；记录设备，不比较 CUDA 吞吐。
- `SOURCE_READ`：阅读 PyTorch autograd、optimizer、checkpoint 相关 API 的官方文档，逐项映射到手写 loop。
- `REMOTE_GPU`：后续若需要，才比较 mixed precision、吞吐、显存和多卡；本包验收不依赖它。

## 6. Planned labs（本包后续实现）

1. `labs/training/label_shift.py`：构造 toy logits，断言 shift 后的 shape 和 loss；
2. `labs/training/train_tiny_lm.py`：只依赖 PyTorch 的最小 decoder LM，支持 CPU/MPS 设备选择；
3. `labs/training/checkpoint_resume.py`：保存模型、优化器、步数、配置和随机状态，比较连续训练与恢复训练；
4. `labs/training/diagnose_training.py`：注入错误的 mask、过大学习率、泄漏 split，要求输出最短诊断路径。

每个 Lab 必须有 deterministic seed、最小断言、运行时间上限和失败样例；不能以“loss 看起来下降”作为唯一测试。

## 7. Failure modes

1. **loss 不降**：先检查 labels 是否右移、mask 是否反了、`optimizer.step()` 是否执行，再检查学习率和初始化；
2. **loss 变 NaN**：检查 logits/梯度范围、学习率、dtype、空有效 token batch，再考虑数值稳定的 log-sum-exp；
3. **训练集很好、验证集很差**：先检查切分泄漏和重复样本，再讨论模型容量、正则化和数据质量；
4. **恢复后曲线跳变**：通常漏存 optimizer state、scheduler state、global step 或随机状态；
5. **梯度累积结果不一致**：确认 loss 是否按有效 token 数归一化，以及是否在每个 micro-batch 错误地 `step()`。

## 8. Interview rehearsal

- **30 秒**：讲清 next-token loss、label shift 和一次参数更新的顺序。
- **3 分钟**：给出一个从 dataloader 到 checkpoint 的训练 loop，并说出三个监控量。
- **15 分钟白板/代码**：手写 masked causal CE、gradient accumulation 和 resume；解释一个 loss 不降的排查过程。

推荐 retrieval questions：

1. 为什么 `logits[:, :-1]` 和 `labels[:, 1:]` 对齐，而不是反过来？
2. 为什么保存模型权重而不保存 optimizer state 会导致恢复训练不等价？
3. 如何区分过拟合、label 泄漏和学习率过大？

## 9. Acceptance gate

- [ ] 不看资料画出训练闭环，并标注每个张量 shape；
- [ ] 在 CPU tiny corpus 上过拟合小样本，且测试 label shift；
- [ ] 故意制造一个 NaN 或 mask 错误，能按顺序定位；
- [ ] 从 checkpoint 恢复后，能解释曲线为何与连续训练一致或不一致；
- [ ] 完成 30 秒、3 分钟、15 分钟三档回答；
- [ ] 用户确认后，才把稳定结论提炼到 `knowledge/`。

## 10. Primary sources

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [PyTorch Optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main/en/trainer)
