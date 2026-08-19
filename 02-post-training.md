# 02｜大模型后训练：SFT、RLHF、PPO、DPO、GRPO、DAPO 与工程实践

> 目标：建立“数据 → 目标函数 → rollout → reward/advantage → policy update → 评测”的统一模型，而不是把 PPO/DPO/GRPO 当成互不相关的八股。

---

## 0. 2026 面试为什么把后训练问得这么深？

公开面经已经出现：

- 百度文心后训练（2026-03-20）：GRPO 数据流、KL、softmax 稳定、`πθ / πold / πrollout`、on/off-policy、batch 太大导致 policy lag、TRL/verl，并要求现场用 Transformers + PyTorch 写 Qwen2 SFT。
- 美团（2026-03/07）：为什么 Reasoning data 做了 SFT 还需要 RL；SFT checkpoint 如何选；importance sampling、clip；7B GRPO 显存与优化。
- 字节：PPO/DPO/GRPO/DAPO 区别、reference model、SFT loss shift-right。
- DeepSeek：DPO 推导、GRPO/推理相关系统知识。

这说明：**“知道名词”已经不够，必须会训练数据流和工程实现。**

---

# Part A｜先建立 Post-training 全景图

```text
Base / Pretrained Model
      ↓
SFT（行为模仿 / 指令遵循）
      ↓
Preference / RL Stage
      ├── Reward Model + PPO
      ├── DPO / preference optimization
      └── GRPO / online verifiable reward RL
      ↓
Evaluation
      ↓
Data flywheel / hard-case mining
```

不同方法解决的问题不同：

- SFT：教模型“像什么样的答案”。
- Preference optimization：教模型“两个看似合理答案里更偏好哪个”。
- Online RL：让模型通过 rollout + reward 从自身探索数据中改进策略。

---

## 1. SFT 到底优化什么？ ★★★★★

SFT 本质是 teacher forcing 下的 next-token cross entropy。

给定：

```text
input_ids = [x1, x2, ..., xT]
```

模型第 `t` 个位置输出 logits 用于预测 `x_{t+1}`。

因此 labels 要 shift：

```text
logits[:, :-1] 预测 labels[:, 1:]
```

### 为什么字节会专门问 shift-right？

因为很多人使用 Hugging Face Trainer 时由模型内部自动处理 loss，于是并不真的理解 causal LM loss 的对齐关系。

### 手写核心 loss

```python
import torch.nn.functional as F

def causal_lm_loss(logits, labels, ignore_index=-100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

### Instruction SFT 为什么经常 mask prompt token？

若目标是只学习 assistant answer，可把 system/user 部分 label 设为 `-100`，只对 assistant completion 计算 loss。否则模型也被要求“预测用户输入本身”。

这不是绝对规则：continued pretraining / full sequence LM 可能有不同目标。面试时先说任务目标再说 mask 策略。

---

## 2. SFT 数据质量比数量更重要吗？ ★★★★★

不能绝对化。更好的回答：

> SFT 数据要同时考虑质量、覆盖、难度、多样性、分布和规模。高噪声数据会直接教错行为；但覆盖不足的“极少精品数据”也无法支撑广泛能力。

数据 pipeline 至少包含：

1. source collection；
2. normalization；
3. deduplication；
4. quality filtering；
5. safety / PII filtering；
6. formatting / chat template；
7. difficulty / domain tagging；
8. train/valid/test contamination control；
9. mixture sampling；
10. human/LLM audit。

### 真实面试延伸

美团问过：不同领域、难度、类型数据的混合配比怎么定。

好的答案不是“按经验 4:3:3”，而是：

- 从目标能力拆分 eval slices；
- 基于 data ablation / scaling experiment；
- 避免某域过采样造成 catastrophic forgetting；
- 根据梯度贡献、训练 loss、下游 eval 与业务分布动态调整。

---

## 3. 为什么 SFT 后还需要 RL？ ★★★★★

SFT 是模仿已有 demonstrations。它的能力上限受示范数据分布强约束。

RL/Preference 的价值包括：

- 使用 preference/reward 信号优化“答案质量”而不仅是 token imitation；
- 允许 model-generated rollout 提供新训练数据；
- 在可验证任务上鼓励探索多种 reasoning path；
- 对 correctness、format、safety、tool success 等目标做更直接优化。

### 但为什么不能“只 RL 不 SFT”？

取决于起点和任务。没有良好初始策略时：

- exploration space 太大；
- reward sparse；
- rollout 质量差；
- 训练不稳定、成本高。

DeepSeek-R1-Zero 展示了 pure RL 可以产生 reasoning，但其论文也报告了 readability、language mixing 等问题；DeepSeek-R1 因此加入 cold-start data 和多阶段训练。

面试中这正是“不要绝对化”的典型题。

---

# Part B｜RLHF 与 PPO

## 4. RLHF 经典流程 ★★★★★

经典 pipeline：

1. SFT policy；
2. 收集 preference pairs；
3. 训练 reward model；
4. policy rollout；
5. reward model 打分；
6. PPO 更新 policy；
7. 通过 KL/reference 约束策略不要偏离过远。

实践版本可能变化，不能把这一套当成唯一 RLHF 定义。

---

## 5. PPO 为什么有 clip？ ★★★★★

策略梯度使用 importance ratio：

```text
r_t(θ) = πθ(a_t|s_t) / πold(a_t|s_t)
```

如果新旧策略变化太大，ratio 可能极端，更新不稳定。

PPO clipped objective 大意：

```text
min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
```

通过限制一次 update 对旧策略的偏移，形成近似 trust-region 效果。

### 高频挖坑

“clip 就能保证 KL 一定小吗？”

不能严格保证。PPO clipping 是局部 surrogate 约束；实际系统还常监控 KL，并配合 early stop / KL penalty 等策略。

---

## 6. Reference Model 是干什么的？ ★★★★★

LLM RL 中 reference policy 常用于 KL regularization：

> 防止当前 policy 为了追 reward 过度偏离原有语言模型分布，从而发生语言质量下降、reward hacking 或模式坍缩。

但 reference model、old policy、rollout policy 是不同角色：

- `πθ`：正在更新的 actor；
- `πold`：生成这批训练数据时对应的旧 policy，用于 importance ratio；
- `πref`：冻结参考模型，用于 KL 约束；
- `πrollout`：实际生成数据的 serving policy，分布式异步系统里可能相对训练 actor 有 lag。

百度 2026 后训练面经就明确追这几个 policy 的关系。

---

# Part C｜DPO

## 7. DPO 为什么能绕过显式 Reward Model + Online PPO？ ★★★★★

DPO 从带 KL regularization 的 RLHF optimum 出发，把隐式 reward 与 policy/reference log-ratio 联系起来，然后使用 chosen/rejected preference pair 直接构造 classification-style loss。

训练数据：

```text
(prompt, chosen, rejected)
```

核心比较：当前 policy 相对 reference 对 chosen 与 rejected 的偏好差异。

### 直觉

希望：

```text
log πθ(chosen|x) - log πref(chosen|x)
```

相对 rejected 更大。

### 优点

- 不需要在线 rollout；
- 不需要单独训练 critic；
- 不需要显式 reward model training loop；
- 实现和训练更简单。

### 局限

- 强依赖 preference dataset；
- offline data 覆盖限制探索；
- preference noise / annotator bias；
- 长度偏好、reference choice、distribution shift 都可能影响结果。

---

# Part D｜GRPO

## 8. GRPO 的核心数据流 ★★★★★

对一个 prompt：

```text
prompt x
  ↓
policy rollout G 个 completions
  ↓
reward function / reward model
  ↓
组内相对 advantage
  ↓
policy ratio / clipping / KL（取决于实现）
  ↓
更新 actor
```

原始 GRPO 的重要动机：**不训练独立 critic/value model**，用同一 prompt 下多条 rollout 的相对 reward 构造 baseline/advantage，从而降低 PPO value model 的资源负担。

一个常见组内标准化形式：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

现代实现对 reward scaling、loss aggregation、KL placement 等细节已有不少修正/变体，面试时要明确“原始论文”与“当前工程实现”可能不完全相同。

---

## 9. GRPO 是 on-policy 还是 off-policy？ ★★★★★｜2026 百度真题

算法意图通常是 online/on-policy：使用当前 policy 产生 rollout 再更新。

但分布式系统中，rollout engine 与 trainer 解耦后会出现 **policy lag**：

```text
rollout policy version < current training policy version
```

因此实际数据会带一定 off-policy 程度。

### batch 太大为什么可能更 off-policy？

如果生成一大批 rollout 需要很久，而 actor 在消费数据并更新，那么后面的训练 step 仍在使用较旧策略生成的数据，staleness 增强。

### 如何缓解？

- 缩短 rollout-to-update delay；
- 降低一次积压 rollout 数；
- policy versioning；
- 同步/半同步 actor-rollout；
- importance sampling correction；
- 限制可接受 policy lag；
- 异步系统使用 truncated IS 等校正方案。

这是很典型的“算法 + distributed system”结合题。

---

## 10. Reward Hacking / Reward Collapse / Entropy Collapse ★★★★★

### Reward hacking

模型找到奖励函数漏洞，分数高但真实目标差。

缓解：

- 多维 reward；
- hard negative / adversarial cases；
- held-out verifier；
- reward model ensemble；
- process/result cross-check；
- 人工 audit；
- 定期刷新 reward。

### Entropy collapse

policy 输出分布过早变尖，探索减少，生成模式单一。

需要监控：

- token entropy；
- response diversity；
- KL；
- reward distribution；
- completion length；
- per-prompt variance。

不能只靠一个 reward 曲线判断 RL 健康。

---

# Part E｜DAPO、DrGRPO、GSPO 等怎么准备

2026 面试已经开始出现“PPO/DPO/GRPO/DAPO 等区别”。此类新算法不建议死背公式，要学会用统一坐标系比较：

1. online / offline？
2. 是否需要 critic？
3. advantage 怎么估计？
4. importance ratio 在 token 还是 sequence 粒度？
5. clip 怎么设计？
6. KL 怎么处理？
7. reward normalization 怎么做？
8. 长 CoT 的 length bias 怎么处理？
9. rollout 与训练 policy mismatch 怎么校正？
10. 工程吞吐和显存成本？

这样新论文出来时你也能快速定位创新点。

---

# Part F｜TRL 与 verl：面试问“用过吗”到底想听什么

## 11. TRL ★★★★☆

Hugging Face TRL 提供 SFT/DPO/GRPO/Reward 等 trainer，并与 Transformers、PEFT、DeepSpeed、vLLM 等集成。

如果回答“我用过 GRPOTrainer”，面试官可能继续问：

- dataset format；
- reward function signature；
- generation bottleneck；
- vLLM colocate vs server；
- GPU memory contention；
- distributed training；
- logging 哪些指标；
- 自定义 reward 如何 debug。

## 12. verl ★★★★☆

verl 是面向 LLM RL post-training 的分布式框架，核心价值是将 actor/rollout/reference/critic 等计算角色与 dataflow、device placement 解耦，并集成 FSDP/Megatron、vLLM/SGLang 等基础设施。

面试至少要知道：

- actor 与 rollout 为什么可能分离；
- rollout 是在线 RL 大头成本；
- colocate / separate GPU 的 trade-off；
- FSDP/Megatron 的训练并行；
- vLLM/SGLang 的 rollout serving；
- GRPO 无 critic 但仍有 actor/ref/rollout 等角色。

---

# Part G｜显存怎么估：7B GRPO 为什么会问？

不要背“7B 需要多少 GB”的固定数字，因为取决于：

- dtype；
- optimizer state；
- gradient dtype；
- activation checkpointing；
- sequence length；
- batch/microbatch；
- ZeRO/FSDP shard；
- LoRA/全参；
- actor/ref/critic/reward 是否 colocate；
- rollout KV Cache；
- vLLM memory utilization。

面试正确姿势：先拆账。

```text
Model params
+ Gradients
+ Optimizer states
+ Activations
+ Temporary buffers
+ Reference / Critic / Reward model
+ Rollout KV cache
```

然后说明用了哪种 parallel/sharding 才给数值估算。

---

# Part H｜真实 2026 大厂面试拆解

## 13. 百度文心后训练｜2026-03-20 ★★★★★

题目链：

1. GRPO 数据流；
2. KL 公式及平滑；
3. softmax 数值稳定；
4. `πθ / πold / πrollout`；
5. on/off-policy；
6. batch 很大怎么缓解 off-policy；
7. TRL/verl；
8. Transformers/PyTorch；
9. 写 Qwen2 SFT。

### 面试官真正验证什么？

这是一个非常典型的“理论 + 框架 + 工程”组合：

- 知道 GRPO 公式不够；
- 必须知道 dataflow；
- 知道 dataflow 不够；
- 必须知道 rollout 与 trainer 在真实系统中会发生 policy mismatch；
- 最后再用 SFT coding 验证你是否真的做过训练。

---

## 14. 美团｜SFT → RL 的阶段选择 ★★★★★

真题类型：

- 已有 reasoning SFT 数据为什么还需要 RL？
- SFT 达到什么程度才进入 RL？
- checkpoint 怎么选？

好答案不是一个固定 threshold，而是建立 gate：

- format compliance；
- base reasoning correctness；
- instruction following；
- safety；
- rollout diversity；
- reward learnability；
- eval benchmark；
- RL 初始化稳定性。

只有“基础策略已经能产生足够比例可学习轨迹”，RL 才更有效率。

---

# Part I｜代码实验

## Lab 1：手写 SFT loss

- 支持 `-100` mask；
- 验证 shift；
- 与模型内置 loss 对比。

## Lab 2：最小 SFT Trainer

用小 Qwen 模型跑：

- chat template；
- tokenize；
- labels mask；
- gradient accumulation；
- save checkpoint；
- eval loss。

## Lab 3：DPO 数据

构造 `(prompt, chosen, rejected)`，打印 chosen/rejected log-prob，理解 reference log-prob。

## Lab 4：GRPO toy

用可验证数学任务：

- 每 prompt 生成 G 条；
- 自定义 correctness reward；
- 观察 group mean/std；
- 记录 entropy/KL/reward/length。

## Lab 5：rollout lag 模拟

保存 policy version，故意让 rollout 使用旧 checkpoint，观察 importance ratio 与训练稳定性。

---

# Part J｜题库优先级

## S 级 ★★★★★

- SFT loss / shift / mask
- SFT 数据构造
- RLHF pipeline
- PPO clip / importance sampling
- reference/old/rollout policy 区别
- DPO 原理与优缺点
- GRPO 数据流与 critic-less
- GRPO on/off-policy 与 policy lag
- KL 在 alignment 中作用
- reward hacking
- RL vs SFT

## A 级 ★★★★☆

- TRL / verl 架构
- rollout bottleneck
- vLLM 与 RL generation
- LoRA RL / full parameter trade-off
- DAPO / DrGRPO / GSPO 比较方法
- entropy collapse
- reward design
- checkpoint selection

## B 级 ★★★☆☆

- process reward vs outcome reward
- verifier-based reward
- length bias
- truncated importance sampling
- async RL
- multi-turn agent RL

---

# 权威来源

- Schulman et al., Proximal Policy Optimization Algorithms.
- Rafailov et al., Direct Preference Optimization.
- Shao et al., DeepSeekMath（GRPO）。
- DeepSeek-AI, DeepSeek-R1.
- Hugging Face TRL 官方文档。
- verl 官方文档/仓库。

真实面试来源见 [`09-2026-real-interviews.md`](./09-2026-real-interviews.md)。

---

# 本章验收

- [ ] 能从 token loss 解释 SFT shift-right。
- [ ] 能画出 PPO/DPO/GRPO dataflow。
- [ ] 能清楚区分 πθ、πold、πref、πrollout。
- [ ] 能解释 policy lag 为什么造成 off-policy。
- [ ] 能写一个最小 Qwen SFT 脚本。
- [ ] 能解释 GRPO 显存不只取决于“7B 参数量”。
- [ ] 能回答百度 2026 后训练二面完整题链。
