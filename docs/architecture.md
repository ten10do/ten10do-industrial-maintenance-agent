# Architecture

This document describes how a request travels through the Industrial Maintenance
Agent, which component owns each decision, and where the LLM is allowed to act.
It is the reference behind the summary in the README.

## 1. Purpose and boundaries

The service answers maintenance questions about registered industrial equipment
and returns an answer plus the evidence it was built from. Two properties define
the design.

The first is that the answer is assembled deterministically. The synthesis step
renders text from tool results and retrieval hits. It never asks a model to write
prose about equipment, because prose cannot be audited and a hallucinated torque
value is worse than no answer.

The second is that the LLM, when it is enabled at all, is confined to the
planning step. It chooses which tools to call and with which arguments. It does
not execute tools, does not see retrieval results when it plans, and does not
compose the final answer. With `PLANNER_MODE=rule` there is no LLM in the process
at any stage.

## 2. Component map

```mermaid
flowchart TB
    Client["Client<br/>HTTP or TestClient"]

    subgraph Transport["Transport layer: app/api"]
        MetaRoutes["meta routes<br/>GET /, GET /health"]
        AgentRoute["agent route<br/>POST /agent/invoke"]
        Errors["structured error handlers"]
    end

    subgraph Service["Service layer: app/services"]
        AgentService["AgentService<br/>correlation, timing, state mapping"]
    end

    subgraph Workflow["Agent workflow: app/agent (LangGraph)"]
        RouteQuery["route_query"]
        PlanActions["plan_actions"]
        ExecuteTools["execute_tools"]
        RetrieveContext["retrieve_context"]
        Synthesize["synthesize"]
    end

    subgraph Planner["Planner layer: app/agent/planners"]
        Dispatch["dispatch<br/>rule / llm / auto"]
        RulePlanner["rule planner<br/>deterministic, frozen"]
        LLMPlanner["LLM planner<br/>optional"]
    end

    subgraph Tools["Tool layer: app/tools"]
        Registry["registry + names"]
        DeviceTool["get_device_status"]
        AlarmTool["query_alarm_code"]
        ManualTool["search_maintenance_manual"]
    end

    subgraph Integrations["Integration layer: app/integrations"]
        LLMProvider["LLM provider<br/>OpenAI-compatible"]
        RAGProvider["RAG provider<br/>local or http"]
        DeviceSource["device data source<br/>external adapter, optional"]
    end

    DB[("SQLite<br/>Device ORM")]
    AlarmData[("data/alarms.json")]

    Client --> MetaRoutes
    Client --> AgentRoute
    AgentRoute --> Errors
    AgentRoute --> AgentService
    AgentService --> RouteQuery
    RouteQuery --> PlanActions
    PlanActions --> Dispatch
    Dispatch --> RulePlanner
    Dispatch --> LLMPlanner
    LLMPlanner -.-> LLMProvider
    PlanActions --> ExecuteTools
    ExecuteTools --> Registry
    Registry --> DeviceTool
    Registry --> AlarmTool
    Registry --> ManualTool
    DeviceTool --> DB
    AlarmTool --> AlarmData
    ManualTool -.-> RAGProvider
    ExecuteTools --> RetrieveContext
    RetrieveContext --> Synthesize
```

Solid edges are always-present paths. Dotted edges are configured off by default:
the LLM provider requires `PLANNER_MODE=llm` or `auto` plus credentials, and the
RAG provider requires `RAG_PROVIDER` to be configured.

## 3. Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as POST /agent/invoke
    participant S as AgentService
    participant G as LangGraph workflow
    participant P as Planner
    participant T as Tool registry
    participant R as RAG provider

    C->>A: InvokeRequest query
    A->>A: validate request model
    A->>S: invoke query
    S->>G: start state with correlation id
    G->>G: route_query classifies intent
    G->>P: plan_actions with query and tool schemas
    P-->>G: AgentPlan validated against the registry
    G->>T: execute_tools with planned calls
    T-->>G: tool results
    G->>R: retrieve_context when a manual search was planned
    R-->>G: retrieval hits
    G->>G: synthesize renders the answer from results
    G-->>S: final state with evidence
    S-->>A: AgentResponse with latency and correlation id
    A-->>C: 200 with answer, tools used, evidence
```

The path length is fixed at planning time. There is no re-planning loop, so a
plan that names an unavailable tool is reported as a failure rather than retried.

## 4. Planner layer

```mermaid
flowchart LR
    Q["Parsed query<br/>+ device vocabulary<br/>+ tool schemas"] --> Mode{"PLANNER_MODE"}

    Mode -->|rule| Rule["Rule planner<br/>deterministic rules<br/>no network"]
    Mode -->|llm| LLM["LLM planner<br/>provider call"]
    Mode -->|auto| Auto{"LLM plan valid?"}
    Auto -->|yes| LLM
    Auto -->|no| Rule

    Rule --> Plan["AgentPlan<br/>tool calls + arguments"]
    LLM --> Validate{"Validation chain"}
    Validate -->|"schema, registry, arguments all pass"| Plan
    Validate -->|"any gate fails"| Err["PlannerError<br/>five codes, no repair"]
```

Both planners emit the same `AgentPlan` model and both are validated against the
same registry, so the executor cannot tell which planner produced the plan. That
equivalence is what makes the Rule versus LLM comparison meaningful.

The validation chain has four gates and five error codes:

| Gate | Failure code |
| ---- | ------------ |
| Text parses as a JSON object | `INVALID_PLANNER_OUTPUT` |
| Object matches `AgentPlan` | `INVALID_PLANNER_OUTPUT` |
| Every named tool exists | `UNKNOWN_TOOL` |
| Every argument set satisfies the tool input model | `TOOL_ARGUMENT_VALIDATION_FAILED` |
| Provider transport fails or times out | `LLM_PROVIDER_ERROR`, `LLM_TIMEOUT` |

## 5. Tools and evidence

```mermaid
flowchart TB
    Plan["AgentPlan tool calls"] --> Exec["Executor"]
    Exec --> D["get_device_status<br/>seed rows, or an external source"]
    Exec --> A["query_alarm_code<br/>data/alarms.json"]
    Exec --> M["search_maintenance_manual<br/>RAG provider"]

    D --> R1["device fields"]
    A --> R2["alarm description and remedy"]
    M --> R3["document hits with source and score"]

    R1 --> Ev["Evidence model"]
    R2 --> Ev
    R3 --> Ev
    Ev --> Syn["Deterministic synthesis"]
    Syn --> Ans["Answer text grounded in the evidence"]
```

Three deterministic tools are registered at import time. None of them calls an
LLM. `search_maintenance_manual` is the only tool with an optional external
dependency, and when the RAG provider cannot run it returns `found=false` with an
`error` rather than an empty result, so an unavailable knowledge base cannot be
mistaken for a knowledge base with no match.

`get_device_status` is answered by two sources selected on the identifier: the
seeded SQLite rows, which own every identifier they define, and any external
adapter configured for the process, which owns its own identifiers only. An
adapter that claims an identifier but cannot be read produces `found=false` with
an `error`, which keeps three outcomes apart: the device is unknown, the source is
broken, or the device was found. The tool contract, the registered name and the
planner are identical whichever source answers, and detection of the external
device is case-insensitive while the answer reports the source's own canonical
identifier.

## 6. Module boundaries

| Layer | Package | Owns | Must not |
| ----- | ------- | ---- | -------- |
| Transport | `app/api` | HTTP contract, validation, error shaping | contain workflow logic |
| Service | `app/services` | correlation, timing, state translation | construct the workflow graph |
| Workflow | `app/agent` | graph order, state model, rule parsing | build HTTP responses |
| Planner | `app/agent/planners` | plan construction and validation | execute tools |
| Tools | `app/tools` | tool behaviour and argument validation | call planners |
| Integrations | `app/integrations` | provider transports | decide which tool to call |
| Persistence | `app/database` | ORM models, schema creation, seeding | hold business rules |

The dependency direction is one way. `app/services/__init__.py` exports lazily to
break the only cycle that would otherwise form between the service and the agent
package.

## 7. Evaluation framework

The evaluation framework lives beside the application rather than inside
`tests/`, because the two answer different questions. A test asserts that the
code matches the specification. The framework measures whether the plan was the
right plan, against a hand-authored answer key.

```mermaid
flowchart LR
    DS["evaluation/dataset.json<br/>49 cases + answer key"] --> Runner["evaluation/runner.py<br/>planner-only or end-to-end"]
    Runner --> Eval["evaluation/evaluator.py<br/>per-case scoring"]
    Eval --> Metrics["evaluation/metrics.py<br/>metric primitives"]
    Metrics --> Rep["evaluation/reports/<br/>baselines and failures"]
    Rep --> Cmp["evaluation/comparison.py<br/>Rule versus LLM delta"]
```

A metric with a zero denominator is recorded as `null`. Reporting `0.0` or `1.0`
there would invent a measurement that was never taken, and the comparison step
refuses to approximate rather than fill the gap.

## 8. Deliberate non-goals

1. No re-planning from tool output. Planning is single-shot by design.
2. No LLM-authored prose. The answer is rendered from evidence.
3. No authentication on `/agent/invoke`. The service is a localhost development
   surface until an auth layer is added.
4. No caching in the retrieval path. The reported latency states the real cost.
