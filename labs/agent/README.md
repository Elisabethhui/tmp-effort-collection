# Lab｜可恢复 Agent 状态机

## 目标

先不用 LangGraph，手写一个最小 durable-run 抽象，理解：

- 显式状态转移；
- checkpoint / resume；
- 最大步数防死循环；
- tool effect 幂等；
- 失败与人工等待边界。

随后再对照 [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) 的 checkpointer/store 区分。

## 运行

```bash
.venv/bin/python -m unittest labs/agent/test_run.py -v
```

## 验收标准

- 非法状态转移会失败；
- checkpoint 恢复后可继续执行；
- 同一个 effect id 重试不会重复产生副作用；
- 超过最大步数会被拒绝；
- 能画出 `PLANNING → TOOL_PENDING → TOOL_RUNNING → COMPLETED/FAILED`。

