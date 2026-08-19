# 05｜Agent 系统：Planning、Tool Calling、Memory、Context、Checkpoint、Multi-Agent 与 Failure Recovery

> 目标：从“会用 LangGraph/LangChain”升级到“能设计 Agent Runtime”。2026 面试已经大量从框架 API 转向状态、容错、安全、评测与长任务治理。

---

## 0. 真实面试已经在问什么

- 阿里 Agent 开发（2026-08-02）：Agent 完整流程、调优、语义检索、向量库、为什么 LangChain/LangGraph、输出规则、Memory、多轮多会话、异常容错、Context 优化。
- 阿里 Agent（2026-07）：幻觉/乱调工具、Agent eval、规划 Prompt、子 Agent context、ReAct、Function Calling JSON、假参数、多 Agent、并发、1 万长文档。
- 淘天 Agent（2026-07-22）：Skill / Memory / RAG 三者分别解决什么问题。
- 字节（2026-03）：多 Agent 协作、冲突、Memory、长期记忆检索效率。
- 腾讯/蚂蚁（2026-04）：多 Agent 编排、并发、失败重试、资金安全、RAG 热更新。
- 美团 2026：Memory vs RAG、Workflow vs Agent、Claude Code memory、Agentic RL、反思 Agent/harness。

这已经不是“ReAct 是什么”级别。

---

# Part A｜Agent 到底是什么

## 1. Workflow vs Agent ★★★★★

### Workflow

控制流主要由开发者预定义：

```text
A -> B -> if -> C/D -> E
```

优点：可预测、容易测试、成本稳定。

### Agent

LLM 在运行时根据状态决定：

- 下一步行动；
- 调哪个工具；
- 参数是什么；
- 是否继续/停止；
- 是否重新规划。

优点：适应开放任务；缺点：不可预测、成本/安全/循环风险高。

### 高频答案

> 能用 deterministic workflow 解决的部分优先 workflow；只有真正需要模型动态决策的节点才 agentic。

这比“Agent 更智能”成熟得多。

---

# Part B｜Agent Runtime 的基本循环

## 2. 一个可用 Agent 至少有哪几个组件？ ★★★★★

```text
User Goal
  ↓
Context Builder
  ↓
Planner / Policy
  ↓
Tool Selection
  ↓
Tool Execution
  ↓
Observation
  ↓
State Update / Checkpoint
  ↓
Continue / Replan / Finish
```

外围还需要：

- memory store；
- permission/policy；
- timeout/retry；
- tracing；
- eval；
- human approval；
- budget；
- sandbox。

---

## 3. ReAct 是什么？ ★★★★☆

ReAct 将 reasoning 与 action/observation 交替：

```text
reason -> action -> observation -> reason -> ...
```

价值：让模型能根据工具返回动态修正下一步。

局限：

- 长轨迹 token 成本；
- 容易 loop；
- reasoning/action 混杂；
- tool error 会污染后续；
- 难以保证关键业务流程。

生产系统通常会对 ReAct 外围加 state machine、budget、tool policy、checkpoint。

---

# Part C｜Function Calling / Tool Calling

## 4. Tool schema 为什么关键 ★★★★★

一个工具应有：

- stable name；
- clear purpose；
- typed input schema；
- output contract；
- error semantics；
- idempotency/side-effect metadata；
- permission requirement。

差的 tool description 会让模型选错工具或编错参数。

### MCP 的关键思想

MCP 标准化工具/资源等能力暴露方式，但 MCP 并不自动解决：

- 权限；
- tool quality；
- prompt injection；
- human approval；
- business transaction safety。

协议 ≠ 安全策略。

---

## 5. JSON 不标准怎么办？ ★★★★★｜阿里真实题

不要只说“再 prompt 一次”。

分层：

1. structured output / schema constrained decoding；
2. parser validation；
3. repair retry；
4. field type/range validation；
5. business validation；
6. unsafe side effect 前 human approval。

若模型生成：

```json
{"amount": -1000000}
```

它 JSON 完全合法，但业务非法。所以 schema validation 只是第一层。

---

## 6. 模型捏造工具参数怎么办？ ★★★★★

- required fields 不允许模型猜；
- provenance：参数来自用户/DB/上一步 tool；
- low-confidence clarification；
- entity resolution；
- server-side authorization；
- dry-run；
- destructive action confirm；
- tool output verification。

核心原则：**LLM 建议参数，可信系统验证参数。**

---

# Part D｜Memory：不要把所有东西都叫 Memory

## 7. Working / Short-term Memory ★★★★★

当前 thread 的运行状态，例如：

- messages；
- 当前计划；
- tool results；
- active task；
- temporary variables。

适合 checkpoint/state store。

## 8. Long-term Memory ★★★★★

跨 thread 持久化：

- user preference；
- durable facts；
- historical decisions；
- reusable experience。

必须有写入门控，不能把所有聊天自动写成长期真相。

## 9. Episodic / Semantic / Procedural Memory ★★★★☆

- Episodic：发生过什么任务、结果如何；
- Semantic：抽象事实/知识；
- Procedural：怎么做某件事，可固化成 Skill/Workflow。

这套划分对 Agent 工程很有用，因为不同 memory 的生命周期、检索和更新策略不同。

---

## 10. Memory vs RAG vs Skill ★★★★★｜淘天 2026 真题

| 组件 | 核心问题 | 典型内容 |
|---|---|---|
| Memory | 过去发生过什么、主体状态是什么 | 用户偏好、任务进度、历史决策 |
| RAG | 当前需要哪些外部知识 | 文档、数据库、代码、商品信息 |
| Skill | 某类任务应该怎么稳定完成 | workflow、工具、规则、验收标准 |

它们可以共用检索技术，但**语义和 authority 不一样**。

---

# Part E｜Context Engineering

## 11. Context 不是越多越好 ★★★★★

Context Window 是有限注意力资源。

常见问题：

- token cost；
- lost-in-the-middle；
- stale state；
- conflicting evidence；
- tool dump 太大；
- irrelevant memory；
- prompt injection。

Context builder 应做：

```text
Goal
+ Current State
+ Minimal Relevant History
+ Relevant Memory
+ Retrieved Knowledge
+ Tool Contracts
+ Current Constraints
```

而不是把整个项目历史塞进去。

---

## 12. 长任务如何做 Context 管理？ ★★★★★

核心是把**durable state** 与 **ephemeral context** 分离。

可以保存：

- goal；
- plan；
- completed steps；
- artifacts；
- evidence；
- unresolved blockers；
- next action；
- checkpoint/version。

新 session 只加载 summary + pointers，需要时 progressive disclosure。

### 面试场景：上下文超长怎么办？

不要只说 summarization。

可以组合：

- sliding window；
- structured state extraction；
- hierarchical summary；
- retrieval；
- artifact pointers；
- active-set selection；
- stale-state pruning；
- context budget per component。

---

# Part F｜Persistence / Checkpoint

## 13. 为什么 Agent 需要 Checkpoint？ ★★★★★

长任务如果第 8 步失败，没有 checkpoint 就可能从头执行，甚至重复副作用。

LangGraph 官方 persistence 设计就是在执行步骤保存 state checkpoint，用于：

- memory；
- human-in-the-loop；
- time travel/debug；
- fault tolerance；
- resume。

### 但 checkpoint 不等于 long-term memory

Checkpoint 保存运行状态；长期 memory 保存跨任务有意义的信息。不要混为一谈。

---

# Part G｜Failure Recovery

## 14. Agent 失败应该怎么分类？ ★★★★★

至少分：

1. model failure；
2. tool transport failure；
3. tool business error；
4. invalid parameters；
5. timeout；
6. partial side effect；
7. permission denied；
8. contradictory state；
9. planner loop；
10. external service degradation。

不同错误不能统一“retry 3 次”。

---

## 15. Retry 为什么可能非常危险？ ★★★★★

读取类 API retry 通常较安全。

转账/发券/发邮件/创建订单等 side effect：

```text
第一次请求其实成功
但 response timeout
Agent 以为失败重试
=> 重复执行
```

解决：

- idempotency key；
- transaction ID；
- read-after-write verification；
- exactly-once illusion via dedup；
- compensation；
- human approval。

腾讯/蚂蚁跨境汇款 Agent 面试就会从这里问资金安全。

---

## 16. 断点恢复怎么设计？ ★★★★★

每 step 输出：

```text
step_id
input_hash
status
result/artifact
side_effect_id
checkpoint_version
next_step
```

恢复时：

1. 加载最后 durable checkpoint；
2. 判断 in-flight step 是否已产生 side effect；
3. 若成功则重放 observation，不重做副作用；
4. 若安全失败才 retry；
5. 从下一 step 继续。

这比“保存聊天历史”更接近生产级 Agent。

---

# Part H｜Multi-Agent

## 17. 什么时候需要 Multi-Agent？ ★★★★★

不要为了“高级”而多 Agent。

适合：

- 清晰角色/权限隔离；
- 并行独立子任务；
- 不同模型/工具/上下文专长；
- reviewer/verifier separation。

不适合：

- 简单线性流程；
- agent 间传大量共享上下文；
- 协调成本 > 分工收益。

---

## 18. 多 Agent 通信怎么做？ ★★★★☆

可通过：

- shared structured state；
- message/event bus；
- artifact store + reference；
- orchestrator request/response；
- blackboard。

生产中尽量传**结构化状态和 artifact reference**，不要无限复制完整聊天 transcript。

---

## 19. 多 Agent 冲突怎么办？ ★★★★☆

- 明确 authority；
- versioned state；
- optimistic concurrency control；
- locks for critical resource；
- planner/arbitrator；
- deterministic merge rules；
- human escalation。

“让另一个 LLM 投票”不是所有冲突的通用解。

---

# Part I｜并发与资源治理

## 20. 多用户 API 一多就卡死怎么设计？ ★★★★★｜阿里真实题

需要工程化回答：

- async I/O；
- worker pool；
- queue/backpressure；
- per-user concurrency limit；
- tool rate limit；
- timeout；
- cancellation；
- budget；
- model request batching；
- circuit breaker；
- priority scheduling。

Agent 系统本质上也是 distributed application。

---

# Part J｜Agent Evaluation

## 21. Agent 系统好坏怎么评？ ★★★★★

不能只看最终 answer。

至少：

### Outcome

- task success；
- correctness；
- human satisfaction。

### Trajectory

- tool selection accuracy；
- parameter accuracy；
- unnecessary steps；
- loop rate；
- recovery success；
- plan adherence。

### System

- latency；
- token cost；
- tool cost；
- error rate；
- P95/P99。

### Safety

- unauthorized action；
- prompt injection success；
- sensitive data leak；
- destructive action without approval。

---

# Part K｜真实面试拆解

## 22. 阿里 Agent｜2026-08-02 ★★★★★

题链：完整流程 → 调优 → 向量检索 → 框架选择 → output control → memory → 多会话 → exception → context。

这组题说明面试官在判断你是不是只“拼了 LangChain demo”。

### 回答策略

项目介绍时主动按：

```text
Goal → State → Planner → Tools → Memory → Retrieval
→ Guardrails → Checkpoint → Eval → Metrics
```

这样后面的追问都有结构。

---

## 23. 美团｜Claude Code Memory / Memory vs RAG ★★★★★

这类题会考你对 coding agent/harness 的理解：项目状态、文件环境、命令结果、session context、持久化信息如何分层。

关键不是背某个产品内部未公开细节，而是说明合理架构，并明确哪些是公开已知、哪些是你的工程推断。

---

# Part L｜实验

## Lab 1：Tool validation

构造错误 JSON、类型正确但业务非法参数、缺字段，分别验证。

## Lab 2：Idempotent tool

实现 `create_order(idempotency_key)`，模拟 timeout 后 retry。

## Lab 3：Checkpoint resume

5-step workflow 在 step 3 崩溃，恢复时不重复前两步副作用。

## Lab 4：Context budget

给 messages/memory/RAG/tool result 设置 token budget，观察 task success 与 cost。

## Lab 5：Agent trajectory eval

记录每一步：state/action/observation/error/cost，计算 unnecessary tool rate。

---

# 高频题库

## S 级 ★★★★★

- Workflow vs Agent
- ReAct
- Function Calling
- tool parameter hallucination
- Memory vs RAG vs Skill
- short-term vs long-term memory
- Context management
- checkpoint/resume
- retry/idempotency
- Agent eval
- Multi-Agent selection
- concurrency/backpressure

## A 级 ★★★★☆

- planner design
- loop prevention
- human-in-the-loop
- MCP
- permission model
- multi-agent communication
- long task context compression
- observability

## B 级 ★★★☆☆

- episodic/semantic/procedural memory
- time-travel debugging
- event sourcing
- compensation/Saga
- policy engine

---

# 权威来源

- LangGraph 官方 Persistence / Memory 文档。
- Model Context Protocol 官方 Specification。
- 相关工具/框架官方文档优先于二手教程。

真实题来源见 [`09-2026-real-interviews.md`](./09-2026-real-interviews.md)。

---

# 本章验收

- [ ] 能画 production Agent runtime。
- [ ] 能区分 Memory/RAG/Skill/Checkpoint。
- [ ] 能解释 context 不是越多越好。
- [ ] 能设计安全 retry。
- [ ] 能设计多 Agent 并发和冲突处理。
- [ ] 能定义 outcome/trajectory/system/safety eval。
