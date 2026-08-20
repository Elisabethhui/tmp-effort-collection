# R5：后训练——SFT、LoRA、DPO 与 GRPO

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；真实大模型训练不作为 Mac 本地门槛。
> 目标：从“会训练语言模型”过渡到“能解释对齐数据、参数高效微调和偏好优化”。

## 1. Mission

后训练面试经常把四件事混在一起问：SFT 在学什么、LoRA 为什么省显存、DPO 是否需要 reward model、GRPO 为什么需要 group/sample。R5 用一条清晰的因果链拆开它们：

```text
预训练模型
  → SFT：从示范数据学会“回答格式与任务行为”
  → 偏好数据：chosen / rejected 或可计算 reward
  → DPO / GRPO：改变偏好，而不是重新预训练世界知识
  → 评测：质量、拒答、安全、长度偏置与回归
```

目标不是背训练器参数，而是能说清数据契约、损失、参考模型/奖励、KL 约束和失败模式。

## 2. Prerequisites

- R4：训练闭环、causal loss、optimizer、checkpoint、训练诊断；
- R3：attention、residual、normalization、logits 和 token-level loss；
- 线性代数：低秩分解、矩阵乘法和参数量估算；
- 不要求 Mac 能加载 7B 模型；toy batch 和源码阅读足够开始。

## 3. Learning outcomes

完成本包后能够：

1. 定义 SFT、LoRA、DPO、GRPO 的输入、输出、优化对象和适用边界；
2. 计算 LoRA 的可训练参数量，并解释为什么 base weights 可以冻结；
3. 写出 DPO 的 log-ratio、reference policy 和 implicit reward 的关系；
4. 解释 GRPO 的 group sampling、relative advantage、KL/长度偏置和 reward hacking；
5. 设计从数据清洗、训练、评测到回滚的最小治理闭环。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 SFT 数据契约 | instruction、conversation、prompt/completion、assistant-only loss | 数据格式决定监督信号 | tokenizer、chat template、response mask、packing | 给一条对话画出参与 loss 的 token |
| U2 SFT 训练与评测 | full fine-tune、packing、eval、遗忘 | SFT 是行为塑形，不等于知识注入 | causal CE + 质量/安全/回归集 | 用 toy 数据过拟合并测 held-out |
| U3 LoRA/PEFT | rank、A/B、alpha、target modules、merge | 用少量增量参数适配任务 | `W' = W + sBA`，base frozen，adapter 可合并 | 手算参数量，比较 rank 与容量 |
| U4 偏好数据 | chosen/rejected、tie、长度、拒答、安全标签 | 偏好噪声会直接进入策略 | pairwise schema、去重、质量分层 | 找出一个“长但不优”的错误 pair |
| U5 DPO | policy/reference、log probability ratio、β | 不训练显式 reward model 也能优化偏好 | 偏好 logistic loss + reference anchor | 用 2 个候选手算 sign 和梯度方向 |
| U6 GRPO | group samples、reward、relative advantage、KL | 同一 prompt 的相对比较降低 critic 需求 | 采样一组回答、归一化 reward、更新 policy | toy reward 上追踪一组 advantage |
| U7 对齐治理 | reward hacking、mode collapse、长度偏置、回滚 | 训练曲线好不代表行为可靠 | 分桶 eval、拒答集、data/model lineage | 设计失败触发条件和回滚点 |

### 必须分清的四个问题

| 方法 | 直接优化什么 | 典型输入 | 常见误解 |
| --- | --- | --- | --- |
| SFT | 示范 token 的 likelihood | prompt + ideal response | 不是自动学会真实世界事实 |
| LoRA | adapter 参数 | 任意监督/偏好目标 | 不是新的优化目标，而是参数化方式 |
| DPO | chosen 相对 rejected 的 policy/reference log-ratio | 偏好 pair | 不等于“完全不需要 reference” |
| GRPO | 基于一组采样回答的相对 reward | prompt + group completions + reward | reward 设计差会被快速利用 |

## 5. Mac validation lane

- `RUNNABLE_CPU`：用小词表、小模型和人工构造的偏好 pair 验证 loss、mask、LoRA 参数量和 toy reward。
- `SOURCE_READ`：先读 PEFT/TRL 的配置、batch collator、trainer loss 入口，再画“数据 → loss → optimizer”的调用链。
- `MPS_OPTIONAL`：若模型和 batch 足够小，可做单步前向/反向烟测；不宣称显存节省比例或训练吞吐。
- `REMOTE_GPU`：大模型、量化、gradient checkpointing、真正的多卡训练另立实验记录，不能混入本地已验证结论。

## 6. Planned labs（本包后续实现）

1. `labs/posttraining/sft_masking.py`：检查 prompt/assistant-only mask、padding 和有效 token 平均；
2. `labs/posttraining/lora_math.py`：从模块 shape 自动计算 LoRA 参数量，验证冻结 base weights；
3. `labs/posttraining/toy_dpo.py`：在二选一 toy preference 上打印 policy/reference ratio、β 和 loss；
4. `labs/posttraining/toy_grpo.py`：为同一 prompt 生成一组候选，计算 reward、relative advantage 和 KL 账本；
5. `labs/posttraining/alignment_eval.py`：按任务、长度、安全和拒答分桶，比较 SFT 前后回归。

## 7. Failure modes

1. **SFT 只学会复述 prompt**：response mask 或 chat template 错误，导致 prompt token 占主导；
2. **LoRA 效果接近随机**：target modules 不匹配、rank 太小、学习率/scale 不对，或 adapter 没有实际挂载；
3. **DPO 偏好颠倒**：chosen/rejected 顺序、reference logprob、padding mask 或 β 符号实现错误；
4. **GRPO reward 很高但答案变差**：reward hacking、长度偏置、验证集泄漏或 reward 与目标不一致；
5. **训练后安全性回退**：只看任务平均分，没有固定安全集、拒答集和回滚门槛；
6. **把训练器配置当原理**：必须回到 batch schema、logprob、advantage 和参数更新路径。

## 8. Interview rehearsal

- **30 秒**：分别说清 SFT、LoRA、DPO、GRPO 的优化对象和一条边界。
- **3 分钟**：给出“从 instruction 数据到 adapter checkpoint”的流程，并解释 assistant-only loss。
- **15 分钟白板/代码**：推导 DPO 的 log-ratio，手写 LoRA forward，设计 GRPO reward 诊断和回滚。

推荐 retrieval questions：

1. LoRA 的参数量如何从 `in_features/out_features/rank` 算出来？
2. DPO 为什么仍然需要 reference policy，它在 loss 中起什么作用？
3. 为什么 GRPO 的 group reward 可能鼓励更长而不是更正确的答案？

## 9. Acceptance gate

- [ ] 能画出 SFT → preference optimization 的数据和梯度边界；
- [ ] 在 CPU toy batch 上验证 response mask、LoRA 参数量和 DPO sign；
- [ ] 能用一个反例解释 reward hacking 或长度偏置；
- [ ] 读过 TRL/PEFT 的关键入口并能指出配置如何落到 loss；
- [ ] 完成三档面试回答，并记录至少一个未验证假设；
- [ ] 用户确认后，才把稳定结论提炼到 `knowledge/`。

## 10. Primary sources

- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index)
- [TRL Quickstart](https://huggingface.co/docs/trl/quickstart)
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [TRL GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
