# 06｜Agent Data、Synthetic Trajectory、Reward、Evaluation 与数据飞轮

> 目标：理解 2026 新增热点——Agent 不只是运行时架构，也需要专门的数据、轨迹、环境、奖励和评测体系。

---

## 0. 为什么这是 2026 必须新增的一章

近期公开面经开始问：

- Agent trajectory data 与普通 LLM SFT 数据有什么区别？
- 如何模拟 User ↔ Agent interaction？
- synthetic data 最难的问题是什么？
- data quality pipeline 怎么做？
- tool call 的训练数据该包含什么？
- 一个 query 有多个工具都能完成时，训练集怎么设计？
- 如何评价 Agent 系统好坏？
- 工具调用频率过高/过低怎么判断？

这些题都指向一个核心：**Agent 的学习对象不再只是“最终回答文本”，而是整个交互轨迹。**

---

# Part A｜普通 SFT 数据 vs Agent Trajectory

## 1. 普通 instruction data

典型：

```text
user prompt
→ assistant answer
```

监督重点是输出 token。

## 2. Agent trajectory data ★★★★★

可能包含：

```text
Goal
State_0
Plan
Tool Call 1
Observation 1
State_1
Tool Call 2
Observation 2
...
Final Answer
Outcome
```

因此质量维度更多：

- plan 是否合理；
- tool 是否选对；
- args 是否正确；
- observation 是否正确利用；
- 是否多走无效步骤；
- 是否在失败后正确恢复；
- 最终任务是否成功。

---

# Part B｜Tool-use Data

## 3. 工具训练数据应该包含什么？ ★★★★★

不能只收“成功调用”。需要覆盖：

- no-tool cases；
- single-tool；
- multi-tool sequence；
- ambiguous tool choice；
- invalid args；
- missing info；
- tool timeout/error；
- permission denied；
- repeated tool result；
- malicious tool output；
- stop/no-op。

否则模型只学会“见问题就调工具”。

---

## 4. 两个工具都能做同一件事，label 怎么定？ ★★★★★

不要机械只标一个“唯一正确工具”。

可以：

- 多正例；
- 按 cost/latency/reliability 定 preference；
- context-dependent tool policy；
- reward-based trajectory selection。

例如：

```text
SQL tool：准确但高延迟
Cache tool：快但可能 stale
```

选择取决于 freshness requirement。

这类数据才会教模型 trade-off。

---

# Part C｜Synthetic Data

## 5. 为什么要 synthetic？ ★★★★☆

Agent 长尾场景巨大，真实高质量 trajectory 稀缺且昂贵。

Synthetic 可以生成：

- user goals；
- environments；
- tool outcomes；
- failure cases；
- adversarial cases；
- alternative trajectories。

### 最大风险

**teacher model 的偏差会被复制。**

因此需要 verify/filter，而不是“GPT 生成后直接训练”。

---

## 6. User-Agent interaction simulation ★★★★☆

可设计 user simulator：

- hidden goal；
- persona；
- constraints；
- information only revealed when asked；
- frustration / correction；
- adversarial behavior。

Agent 与 simulator 多轮交互后，根据最终 task completion 评分。

### 关键：避免 simulator 泄露答案

user simulator 必须有自己的 hidden state，不应把完整 goal 直接放进 agent 可见 context。

---

# Part D｜Environment

## 7. 为什么 Agent RL 需要 Environment？ ★★★★★

语言模型普通 RL 可以对文本 completion 评分。

Agent 需要和外部世界交互：

```text
policy action
→ environment transition
→ observation
→ next action
```

environment 可以是：

- code sandbox；
- browser；
- database；
- game；
- mock business system；
- tool suite。

Hugging Face TRL 当前 GRPO 已支持 tools / agent training 与 multi-environment 类型能力，说明 Agent RL 正快速工程化。

---

# Part E｜Reward Design

## 8. Final-answer reward 不够 ★★★★★

一个 Agent 最终答对，但可能：

- 调了 30 次无关工具；
- 泄漏敏感数据；
- 修改了不该改的文件；
- 成本极高。

因此 reward 可以拆：

```text
Task success
+ Tool correctness
+ Safety
+ Efficiency
+ Format
+ Constraint satisfaction
```

### 但 reward 越多越好吗？

不是。多个 reward 会冲突，权重不当会造成 gaming。

需要：

- reward ablation；
- per-dimension metrics；
- Pareto/trade-off；
- hard constraints 与 soft rewards 分离。

例如“不能转错钱”应该是 hard policy，不只是 -5 reward。

---

# Part F｜Process Reward / Verifier

## 9. Outcome Reward vs Process Reward ★★★★☆

Outcome：只看最终答案。

Process：对中间步骤/推理/工具调用给信号。

### Process reward 的价值

- sparse reward 变 dense；
- 更容易定位坏步骤；
- 可以指导长轨迹。

### 风险

- 中间步骤标注很贵；
- verifier 本身会错；
- 模型可能 optimize verifier quirks。

所以 verifier 也需要独立评测。

---

# Part G｜Agent Evaluation

## 10. 评测集应该怎么做？ ★★★★★

分层：

### Capability slices

- retrieval；
- coding；
- planning；
- tool calling；
- multi-turn；
- long context；
- recovery。

### Difficulty

- easy deterministic；
- ambiguous；
- long-horizon；
- adversarial；
- partial failure。

### Environment variability

不要让 benchmark 只适配一套固定 mock response。

---

## 11. Agent 指标 ★★★★★

### Outcome

- success rate；
- exact correctness；
- business KPI。

### Trajectory

- steps；
- tool precision/recall；
- redundant action rate；
- recovery rate；
- invalid action rate。

### Cost

- input/output tokens；
- tool calls；
- wall time；
- $ cost。

### Robustness

- retry；
- service failure；
- prompt injection；
- noisy tool output。

---

# Part H｜Data Quality Pipeline

## 12. Agent 数据怎么过滤 ★★★★★

```text
raw trajectories
→ schema validation
→ dedup
→ outcome verification
→ tool/action validation
→ safety validation
→ trajectory quality scoring
→ diversity sampling
→ train/eval split
```

### 去重为什么很重要？

大量 synthetic prompts 可能只是模板变体，造成 benchmark leakage 或训练分布假繁荣。

可在：

- text hash；
- normalized template；
- embedding similarity；
- trajectory signature

多层去重。

---

# Part I｜Hard Case Mining 与 Data Flywheel

## 13. 线上数据如何回流？ ★★★★★

```text
production traces
→ privacy/redaction
→ failure clustering
→ root-cause labeling
→ hard-case set
→ fix prompt/model/tool/runtime
→ regression eval
→ deploy
```

重点：不是所有线上 trace 都直接训练。

需要 governance：

- consent/privacy；
- PII；
- sensitive action；
- data retention；
- contamination。

---

# Part J｜真实面试答题模板

## 14. “Synthetic data 最难的是什么？”

建议答：

1. coverage：是否覆盖真实长尾；
2. fidelity：模拟 environment 是否真实；
3. label correctness；
4. diversity；
5. leakage；
6. teacher bias；
7. cost；
8. evaluation correlation。

## 15. “怎么评价工具调用频率？”

不是找固定次数。

定义：

- necessary tool recall；
- unnecessary tool rate；
- task success；
- latency/cost；
- repeated-call rate。

根据任务类型分桶比较。

---

# Part K｜实验

## Lab 1：Trajectory schema

设计 JSONL：

```text
goal/state/actions/observations/final/outcome/cost/error
```

## Lab 2：User simulator

hidden goal + progressive disclosure。

## Lab 3：Failure injection

随机让 tool 10% timeout、5% malformed output，测 recovery。

## Lab 4：Reward ablation

移除 efficiency reward，观察 tool calls 是否增加。

## Lab 5：Regression suite

把生产 bad case 固化成不允许回归的 test set。

---

# 高频题库

## S 级 ★★★★★

- Agent trajectory vs SFT data
- tool use dataset
- synthetic data quality
- user-agent simulation
- reward design
- Agent eval dimensions
- hard-case mining

## A 级 ★★★★☆

- process reward
- verifier
- environment design
- multi-turn RL
- failure injection
- benchmark leakage

## B 级 ★★★☆☆

- curriculum
- self-play
- offline trajectory filtering
- counterfactual trajectories

---

# 本章验收

- [ ] 能定义 Agent trajectory schema。
- [ ] 能说清 tool/no-tool/multi-tool 数据怎么构造。
- [ ] 能设计 user simulator 而不泄露 hidden goal。
- [ ] 能从 outcome/trajectory/cost/safety 四层评 Agent。
- [ ] 能设计线上 bad-case 数据飞轮。
