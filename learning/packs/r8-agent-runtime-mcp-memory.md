# R8：Agent Runtime、MCP 与 Memory

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；外部模型和线上工具默认关闭。
> 目标：从“会调用一个工具”进阶到能设计可恢复、可观测、可控成本的 Agent runtime。

## 1. Mission

Agent 面试最容易落入框架 API 记忆：一个 prompt、一个 tool call、一个 graph 并不等于可靠 Agent。R8 关注运行时的边界和故障：

```text
goal → plan/state → tool contract → execute
     → validate/observe → retry or human approval
     → checkpoint/memory → final answer + trace
```

要能回答“什么时候不该用 Agent”“工具失败后如何重试且不重复扣款”“长期记忆如何避免污染上下文”。

## 2. Prerequisites

- R4：函数调用、错误处理、checkpoint 和可复现训练思维；
- R6：延迟、token budget、缓存和服务指标；
- R7：检索、引用和权限边界；
- Python 基础、JSON schema、HTTP/进程边界；不要求先掌握某个 Agent 框架。

## 3. Learning outcomes

完成本包后能够：

1. 区分 workflow、router、tool-using agent、multi-agent 的必要条件和成本；
2. 设计带输入/输出 schema、超时、权限、幂等键和副作用声明的工具契约；
3. 解释 ReAct/function calling/MCP 的消息与执行边界；
4. 用状态图实现 checkpoint、retry、backoff、human-in-the-loop 和 resume；
5. 区分短期上下文、episodic memory、semantic memory，并控制召回、过期和删除；
6. 为 Agent 记录 trace、工具耗时、token、失败原因和决策证据。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 Workflow vs Agent | 固定 DAG、router、循环决策、multi-agent | 复杂度和不确定性决定是否需要 Agent | 状态、条件边、预算和终止条件 | 给 4 个业务需求选架构 |
| U2 Tool contract | schema、权限、timeout、idempotency、side effect | 工具是风险边界，不是 prompt 片段 | JSON schema、typed result、error taxonomy | 为付款/搜索工具写契约 |
| U3 ReAct/function calling/MCP | thought/action/observation、结构化 tool call、协议边界 | 解释模型和 runtime 谁负责什么 | message envelope、tool registry、transport | 画一轮调用时序图 |
| U4 State runtime | graph state、checkpoint、resume、retry/backoff | 进程崩溃后仍需可恢复 | durable state、step id、dedupe key | 模拟中断并恢复 |
| U5 Memory | context、episodic、semantic、profile、TTL | “记住一切”会污染和越权 | write policy、retrieval、confidence、deletion | 设计记忆写入/撤回规则 |
| U6 Human/safety | approval、sandbox、budget、allowlist、PII | 有副作用的动作不能完全自动化 | risk tier、interrupt/resume、audit log | 为工具分级并设审批点 |
| U7 Observability/eval | trace、trajectory、tool success、cost、latency | Agent 质量要看路径，不只看最终文本 | span/run id、事件 schema、trajectory eval | 设计一张单次运行 trace |

### Runtime 心智模型

模型只能提出结构化意图；runtime 才拥有工具权限、重试策略、状态持久化、预算和终止权。将两者混为一谈，会导致“模型说成功所以真的成功”“重试造成重复副作用”等事故。

## 5. Mac validation lane

- `RUNNABLE_CPU`：默认使用离线 fake model/fake tools，测试状态图、schema、重试、幂等、checkpoint 和 trace。
- `SOURCE_READ`：优先阅读 MCP specification/SDK、LangGraph persistence/interrupt 入口以及现有 notebook 中的最小模式。
- `MPS_OPTIONAL`：只做本地小模型或 embedding 烟测；不要求在线 provider。
- `REMOTE_GPU`：真实模型、多 agent 并发、长上下文和成本压测另立实验，保留 provider/model/version。

## 6. Planned labs（本包后续实现）

1. `labs/agents/tool_contract.py`：schema 校验、超时、错误分类、权限和幂等 key；
2. `labs/agents/state_machine.py`：纯 Python 状态图，支持 retry/backoff、interrupt/resume；
3. `labs/agents/memory_policy.py`：短期上下文、可撤回长期记忆、TTL 和 ACL 过滤；
4. `labs/agents/mcp_smoke.py`：本地 stdio fake MCP server/client，不调用外部服务；
5. `labs/agents/trace_eval.py`：输出 trace、token/cost/latency、tool success 和轨迹评分。

## 7. Failure modes

1. **把固定流程做成 Agent**：不确定性很低却引入循环和模型调用，成本、延迟和可审计性变差；
2. **工具 schema 松散**：模型生成缺字段或错误类型，runtime 只能靠自然语言猜测；
3. **重试重复副作用**：没有 idempotency key、事务边界或人工审批；
4. **checkpoint 只存对话文本**：漏掉 tool result、版本、step id 或外部状态，恢复后无法继续；
5. **memory 污染**：把一次猜测写成事实、没有来源/置信度/TTL、无法删除或越权召回；
6. **只评最终答案**：忽略工具失败、绕过权限、超预算和不必要循环。

## 8. Interview rehearsal

- **30 秒**：解释 Agent runtime 与 LLM 的职责边界。
- **3 分钟**：设计一个带工具、状态、重试、审批和 trace 的 Agent。
- **15 分钟白板/代码**：实现可恢复状态图，处理工具超时、重复扣款和记忆撤回。

推荐 retrieval questions：

1. 什么条件下 workflow 比 Agent 更合适？
2. 你如何保证一个有副作用的工具重试不会重复执行？
3. 长期 memory 写入前需要哪些证据、版本和删除机制？

## 9. Acceptance gate

- [ ] 用 fake model/fake tools 完成一个可恢复的 CPU state machine；
- [ ] 工具契约包含 schema、权限、timeout、错误类型和幂等设计；
- [ ] 本地 MCP smoke test 能解释 transport、tool result 和失败边界；
- [ ] 一次运行能输出完整 trace，并能按轨迹定位循环或越权；
- [ ] 完成三档面试回答，且明确哪些结论未经过真实 provider 验证；
- [ ] 通过 gate 后，再把 Agent 设计原则晋级到 `knowledge/`。

## 10. Primary sources

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification/latest)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [LangGraph overview](https://langchain-ai.github.io/langgraph/)
- [LangGraph persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit0/introduction)
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
