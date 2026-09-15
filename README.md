# Industrial Maintenance Agent

Agent service for industrial equipment maintenance assistance. V0.4 wires the
Industrial Knowledge RAG repository in as a retrieval tool: the workflow is a
fully deterministic `route_query -> plan_actions -> execute_tools ->
retrieve_context -> synthesize` pipeline backed by the registry, and a rule-based
parser. No LLM is involved at any stage.

V0.4.2 exposes that pipeline over HTTP through `POST /agent/invoke`, behind an
`app/api` transport layer and an `AgentService` that owns request correlation,
latency measurement, state translation and logging. The workflow itself is
unchanged.

V0.5 adds an optional LLM planner alongside the frozen rule planner. The planner
decides which tools to call and with which arguments; it never writes the answer
and never produces evidence. The final answer stays deterministic and grounded in
what the tools actually returned. The default configuration,
`PLANNER_MODE=rule`, leaves V0.4 behaviour untouched: with no provider configured
the pipeline is still entirely LLM-free.

V0.6 adds an agent evaluation framework. It is separate from `tests/` on purpose:
a software test asserts that the code does what the specification says, and it
cannot tell you how often the plan is the right plan. The framework measures agent
capability against a hand-authored answer key, so the rule planner's intent
accuracy, tool selection and argument handling are quantified instead of assumed.
The baseline is produced by running the real planner over the real dataset. No stub
is scored, and no number is reported that was not measured.

V0.7 extends that framework to a real LLM planner benchmark. It adds a Rule versus
LLM comparison that refuses to approximate, a latency distribution with a documented
p95 convention, and the gate that stops a stub from standing in for a provider. The
benchmark has since been run against a real external provider (`openai_compatible`,
model `deepseek-flash`): 49 cases, planner-only, no tool execution. The LLM column,
the comparison deltas and the out-of-domain comparison are published below. Every
number comes from the recorded run under `evaluation/reports/`; none is estimated,
inferred or invented.

## Stack

| Concern       | Choice                                        |
| ------------- | --------------------------------------------- |
| Runtime       | Python 3.11                                   |
| Web service   | FastAPI + Uvicorn                             |
| Agent workflow| LangGraph                                     |
| Data models   | Pydantic v2                                   |
| Persistence   | SQLAlchemy 2.0 (SQLite by default, Alembic ready) |
| RAG client    | `httpx` (HTTP provider) or in-process import (local provider) |
| LLM client    | `httpx`, optional and configured off by default |
| Testing       | pytest + httpx TestClient                     |
| Agent eval    | `evaluation/`: hand-authored dataset + metric harness |

## Project layout

```
industrial-maintenance-agent/
├── app/
│   ├── __init__.py
│   ├── config.py            # Settings loaded from env / .env
│   ├── logging_config.py    # Logger configuration for the app namespace
│   ├── main.py              # FastAPI composition: routers + error handlers
│   ├── api/                 # HTTP layer (transport only, no workflow)
│   │   ├── errors.py        # Structured failure handler
│   │   ├── routes/
│   │   │   ├── agent.py     # POST /agent/invoke
│   │   │   └── meta.py      # GET / and GET /health
│   │   └── schemas/
│   │       └── agent.py     # Invoke request / response / error models
│   ├── agent/
│   │   ├── state.py         # MaintenanceState (LangGraph state)
│   │   ├── parser.py        # Rule-based natural language query parser
│   │   ├── graph.py         # Planner / executor / context / synthesis graph
│   │   └── planners/        # Planner layer (V0.5)
│   │       ├── schema.py        # AgentPlan / ToolCallPlan + error taxonomy
│   │       ├── tool_schemas.py  # JSON Schema generated from the input models
│   │       ├── llm.py           # LLM planner: prompt, call, validation
│   │       └── dispatch.py      # rule / llm / auto selection + bookkeeping
│   ├── integrations/
│   │   ├── llm/             # Optional OpenAI-compatible provider (V0.5)
│   │   │   ├── models.py    # LLMMessage / request / result, vendor-neutral
│   │   │   ├── base.py      # LLMProvider interface + error taxonomy
│   │   │   ├── openai_compatible.py  # httpx transport, no vendor SDK
│   │   │   └── factory.py   # Config-driven selection + configuration gate
│   │   └── rag/
│   │       ├── models.py        # RAGSearchHit / RAGSearchResponse
│   │       ├── base.py          # RAGProvider interface + RAGProviderError
│   │       ├── local_provider.py # In-process retrieval (LLM-free)
│   │       ├── http_provider.py  # POST /ask client
│   │       └── factory.py        # Config-driven provider selection
│   ├── schemas/
│   │   ├── maintenance.py   # Pydantic request / response models
│   │   └── evidence.py      # Evidence model (tool + document sources)
│   ├── services/
│   │   ├── __init__.py      # Lazy exports (breaks the agent/services cycle)
│   │   ├── agent_service.py # Invocation: correlation, timing, state mapping
│   │   └── device_catalog.py # Device id + alias resolution
│   ├── database/
│   │   ├── base.py          # Declarative base
│   │   ├── models.py        # Device ORM model
│   │   ├── session.py       # Engine + session factory
│   │   ├── init_db.py       # Schema creation + seed helpers / CLI
│   │   └── __main__.py      # `python -m app.database` entry point
│   └── tools/
│       ├── names.py         # Canonical tool names (ToolName enum)
│       ├── registry.py      # Tool registry, carries each tool's input model
│       ├── arguments.py     # Strict argument validation, shared by both layers
│       ├── device_tool.py   # get_device_status (Device ORM backed)
│       ├── alarm_tool.py    # query_alarm_code (data/alarms.json backed)
│       └── maintenance_manual_tool.py  # search_maintenance_manual (RAG backed)
├── evaluation/                  # The Agent evaluation framework, the only one
│   ├── dataset.json             # 49 hand-authored cases + ground-truth policy
│   ├── dataset.py               # Dataset models and answer-key validation
│   ├── metrics.py               # Metric primitives; a zero denominator is null
│   ├── evaluator.py             # Per-case scoring and aggregation
│   ├── runner.py                # CLI: planner-only / end-to-end, LLM gate
│   ├── comparison.py            # CLI: rule versus LLM delta, refuses to approximate
│   └── reports/                 # Generated baselines and failure records
├── tests/
│   ├── test_health.py           # Smoke tests
│   ├── test_database.py         # Device model + init_db tests
│   ├── test_tools.py            # Device tool, alarm tool, registry
│   ├── test_parser.py           # Parser + route_query / plan_actions tests
│   ├── test_pipeline.py         # Planner -> executor -> synthesis tests
│   ├── test_rag_provider.py     # Provider contract tests
│   ├── test_maintenance_manual_tool.py  # RAG tool contract tests
│   ├── test_rag_pipeline.py     # Graph order, evidence mapping, e2e
│   ├── test_rag_score_semantics.py      # Score labelling and direction
│   ├── test_api.py              # HTTP contract, validation, failure handling
│   ├── fake_llm.py              # LLM provider double, no network
│   ├── test_llm_provider.py     # Provider envelope, errors, credential safety
│   ├── test_llm_planner.py      # Prompt, plan validation, error taxonomy
│   ├── test_planner_modes.py    # rule / llm / auto, executor, API surface
│   ├── test_evaluation.py       # Dataset, metrics, OOD safety, CLI, LLM gate
│   └── test_planner_comparison.py  # rule vs LLM delta, refusals, hash lock
├── data/
│   ├── devices.json         # Device seed data
│   ├── alarms.json          # Alarm code catalog
│   └── (industrial_maintenance.db generated locally, gitignored)
├── scripts/
│   ├── rag_build_knowledge_base.py  # Build the RAG light index from a corpus
│   ├── rag_retrieval_probe.py       # Run retrieval-only queries, no LLM
│   └── llm_planner_probe.py         # Real LLM planner smoke gate, no fabrication
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Dev / test dependencies
├── requirements-rag-local.txt   # Optional deps for RAG_PROVIDER=local
├── pyproject.toml           # Project metadata + tool config
├── .env.example             # Environment template
└── README.md
```

## Getting started

Create and activate a Python 3.11 virtual environment, then install
dependencies:

```bash
python3.11 -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements-dev.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

## Running the service

```bash
uvicorn app.main:app --reload --port 8000
```

Available endpoints:

| Method | Path            | Purpose                          |
| ------ | --------------- | -------------------------------- |
| GET    | `/`             | Service identity probe           |
| GET    | `/health`       | Liveness / readiness probe       |
| POST   | `/agent/invoke` | Run the maintenance agent        |

Interactive API docs are served at `/docs` and `/redoc`; the raw schema is at
`/openapi.json`.

> The RAG service documents port `8000` as well. When both services run on the
> same host, change one of them; the agent never assumes a port.

## API

### `POST /agent/invoke`

Runs one agent invocation. With the default `PLANNER_MODE=rule` the run is fully
deterministic; with `llm` or `auto` the planning step calls a configured provider
and the answer remains deterministic and grounded. The endpoint is declared with
`def` rather than `async def` on purpose: the pipeline is synchronous, so
Starlette dispatches it to a worker thread instead of blocking the event loop. The
provider call is bounded by `LLM_TIMEOUT_SECONDS`, so a slow endpoint occupies a
worker thread for at most that long.

Request body:

| Field   | Type   | Required | Notes                                        |
| ------- | ------ | -------- | -------------------------------------------- |
| `query` | string | yes      | 1 to 2000 characters, stripped before use     |
| `debug` | bool   | no       | Defaults to `false`; adds the diagnostic block |

```bash
curl -s -X POST http://127.0.0.1:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "PowerFlex 520 drive F007 motor overload how to fix"}' \
  | python -m json.tool
```

Response body:

| Field          | Type            | Notes                                                     |
| -------------- | --------------- | --------------------------------------------------------- |
| `request_id`   | string          | Fresh uuid4 per invocation, also written to the log        |
| `query`        | string          | Echo of the normalized query                               |
| `intent`       | string \| null  | `alarm_diagnosis`, `maintenance_advice`, `device_status`, `unknown` |
| `equipment_id` | string \| null  | Resolved equipment identifier                              |
| `alarm_code`   | string \| null  | Resolved alarm code                                        |
| `tools_called` | list[string]    | Tools the executor actually ran, in execution order        |
| `planner_used` | string \| null  | Planner that produced the plan: `rule` or `llm`             |
| `planner_fallback` | bool        | `true` when an LLM plan was attempted and the rule planner ran instead |
| `answer`       | string          | Deterministic Chinese answer with the three fixed sections  |
| `evidence`     | list[Evidence]  | Citable facts; document evidence carries page and score     |
| `latency_ms`   | float           | Measured service latency in milliseconds                   |
| `debug_info`   | object \| null  | Present only when `debug=true`                             |

```json
{
  "request_id": "67640994-02df-4c08-a7d0-e7d202ae1094",
  "query": "PowerFlex 520 drive F007 motor overload how to fix",
  "intent": "maintenance_advice",
  "equipment_id": null,
  "alarm_code": null,
  "tools_called": ["search_maintenance_manual"],
  "planner_used": "rule",
  "planner_fallback": false,
  "answer": "【设备状态】\n本次未查询设备状态。\n\n【报警信息】\n本次未查询报警信息。\n\n【维护手册证据】\n命中 4 条片段（provider：local，检索模式：hybrid）。",
  "evidence": [
    {
      "source_type": "document",
      "source": "PowerFlex_520_User_Manual.pdf",
      "tool_name": "search_maintenance_manual",
      "content": "Fault Code F007 Motor Overload ...",
      "document": "PowerFlex_520_User_Manual.pdf",
      "page": 84,
      "score": 0.6743284463882446,
      "score_semantics": "vector_cosine_distance",
      "higher_is_better": false
    }
  ],
  "latency_ms": 4894.949,
  "debug_info": null
}
```

`tools_called` reports execution, not intention. The planner's `required_tools`
may name a tool the executor could not dispatch, and reporting the plan as
execution would overstate what happened. The plan is available through
`debug_info.required_tools` instead.

`planner_used` and `planner_fallback` describe which planner ran, not which one
was configured. In `auto` mode a fallback is visible without reading the log,
because a silently downgraded plan would look identical to a successful one.

When `debug=true`, `debug_info` carries:

| Field                     | Notes                                                        |
| ------------------------- | ------------------------------------------------------------ |
| `required_tools`          | Tools the planner scheduled                                  |
| `tool_status`             | Per-tool `found` flag and diagnostic                         |
| `internal_error`          | Pipeline-level diagnostic, such as an undispatched tool      |
| `planner_fallback_reason` | `<CODE>: <detail>` when the rule planner ran after an LLM attempt |

The debug block never contains the planner's system prompt, the raw completion or
an environment value. A model can echo its prompt, and the prompt carries the tool
schemas and the device vocabulary, so only curated fields are exposed.

### Validation

`query` is constrained to 1 to 2000 characters. A blank query, including a run
of spaces, is rejected rather than passed to the parser: `min_length=1` alone
would accept `"   "`. Rejections use the standard FastAPI 422 body and name the
offending field.

```json
{"detail":[{"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character","input":"","ctx":{"min_length":1}}]}
```

### Failures

A pipeline failure returns HTTP 500 with a structured body. The exception, its
message, its stack and the configuration never leave the process; the caller gets
a stable code and the `request_id` needed to find the matching server log line.

```json
{
  "request_id": "3de135b2-2b8f-455c-996d-92f7dfec99cb",
  "error": "agent_invocation_failed",
  "message": "Agent 执行失败，请稍后重试；如问题持续，请提供 request_id 以便排查。",
  "latency_ms": 11.796
}
```

Note the distinction: a knowledge base that is reachable but empty is not a
failure. The request succeeds, no document evidence is produced, and the answer
says the manual contains nothing on the query.

In `PLANNER_MODE=llm` a planning failure is a request failure and returns HTTP 500
with the planner's own code in `error` instead of `agent_invocation_failed`. The
five codes are `LLM_PROVIDER_ERROR`, `LLM_TIMEOUT`, `INVALID_PLANNER_OUTPUT`,
`UNKNOWN_TOOL` and `TOOL_ARGUMENT_VALIDATION_FAILED`. The body stays generic: no
prompt, no raw completion, no endpoint header and no credential.

```json
{
  "request_id": "9a1c6f2e-0c1b-4d2f-9f2a-5c0b7e4d1a33",
  "error": "LLM_TIMEOUT",
  "message": "Agent 执行失败，请稍后重试；如问题持续，请提供 request_id 以便排查。",
  "latency_ms": 30012.441
}
```

### Logging

One line per invocation, with a fixed field set:

```
2026-09-15 13:57:58,280 INFO app.services.agent_service agent_invoke request_id=<uuid> success=true latency_ms=13.544 planner=rule fallback=false tools_called=get_device_status query='PLC-001 现在什么状态'
2026-09-15 13:57:58,280 ERROR app.services.agent_service agent_invoke_failed request_id=<uuid> success=false latency_ms=11.796 error_type=OperationalError
```

The success line records which planner ran and whether it was a fallback, so a
downgrade is visible in the operational record. A fallback also logs a warning
from the dispatcher carrying the reason code, never the prompt.

`app/logging_config.py` attaches a stream handler to the `app` package logger,
called from `app/main.py`. Without it an unconfigured logger inherits the root
level `WARNING` and every `INFO` line is dropped, so a successful invocation
would leave no trace. The change stays inside the `app` namespace, leaving
third-party verbosity alone, and `propagate` remains `True` so a host root
handler still receives these records. Verbosity is set by `LOG_LEVEL`
(default `INFO`); an unknown value fails at startup.

The query is rendered with `%r`, so a query containing spaces, quotes or a
newline stays on one line and remains parseable.

The failing line records the exception's class name and nothing else. An
exception message can embed a provider URL, a header or a key fragment, so it is
deliberately not logged. Neither line records an environment value, a credential
or a traceback.

### Layering

`app/main.py` composes and nothing more. `app/api` holds transport models,
routers and error handling; it never imports the workflow. `app/services`
holds `AgentService`, which calls the graph, generates the request id, measures
latency and maps the final state onto the response schema. `app/agent` holds the
workflow, unaware that HTTP exists. The compiled graph is therefore reachable
through exactly one route.

## Database

SQLite is the default backend. The `Device` ORM model lives in
`app/database/models.py` and exposes these fields: `id`, `device_id`,
`device_name`, `device_type`, `location`, `status`, `temperature`, `pressure`,
`rpm`, `alarm_code`, `last_maintenance_time`.

### Initialize the database

```bash
# Create tables and seed data/devices.json
python -m app.database

# Drop existing tables, recreate them, then seed
python -m app.database --reset

# Create tables only
python -m app.database --no-seed
```

The same CLI is also available as `python -m app.database.init_db`.

Seeding upserts by `device_id`, so repeated runs are idempotent. The service
also creates missing tables on startup when `AUTO_CREATE_TABLES=true` (default).
The seed file `data/devices.json` ships with `PLC-001`, `PLC-002`, `Robot-001`
plus two additional devices.

## Tools

Three deterministic tools are registered on the shared registry
(`app.tools.registry`) at import time. None of them performs an LLM call.
Identifiers come from `app/tools/names.py` (`ToolName`), so the planner and the
registry cannot drift apart.

| Tool | Input | Output | Source |
| ---- | ----- | ------ | ------ |
| `get_device_status` | `DeviceStatusInput(device_id)` | `DeviceStatusOutput` | `Device` ORM table (SQLite) |
| `query_alarm_code` | `AlarmQueryInput(alarm_code)` | `AlarmQueryOutput` | `data/alarms.json` |
| `search_maintenance_manual` | `ManualSearchInput(query, top_k)` | `ManualSearchOutput` | Industrial Knowledge RAG |

```python
from app.tools import get_device_status, query_alarm_code, search_maintenance_manual

status = get_device_status("PLC-001")
alarm = query_alarm_code(status.alarm_code or "")
manual = search_maintenance_manual("包装温控超限 处理")
```

Unknown identifiers do not raise. `get_device_status` and `query_alarm_code`
return the output model with `found=False`, which keeps dispatch logic in the
graph free of exception handling. Alarm codes are matched
case-insensitively. `register_default_tools()` is idempotent, so the registry can
be rebuilt in tests without duplicate registration errors.

## RAG integration

`search_maintenance_manual` is a thin adapter. Retrieval itself is delegated to a
provider behind `app/integrations/rag/base.py::RAGProvider`, selected by
configuration.

| Provider | Transport | LLM call | Requires |
| -------- | --------- | -------- | -------- |
| `local` (default) | In-process import of `backend.light_rag_core.retrieve_docs` | none | A RAG checkout at `RAG_REPO_ROOT` plus `requirements-rag-local.txt` |
| `http` | `POST {RAG_BASE_URL}/ask` with the `X-Knowledge-Base-ID` header | the RAG service runs its own answer generator; the agent discards `answer` and consumes `sources` only | A running RAG service and `RAG_BASE_URL` |

`local` is the default because it is the only path with no LLM anywhere in the
pipeline. The RAG repository is consumed read-only: the agent imports it, never
writes to it and never copies its source.

The tool contract keeps three outcomes apart, which matters more than it looks:
retrieval succeeded with hits (`found=True`), retrieval succeeded with no hits
(`found=False`, `error=None`), and retrieval could not run at all (`found=False`,
`error=<diagnostic>`). Collapsing the last two would let a broken deployment look
like an empty manual.

Each hit carries `content`, `document`, `page`, `section`, `chunk_id` and
`score`, mapped from what the RAG system actually returns:

| Tool field | `local` source | `http` source |
| ---------- | -------------- | ------------- |
| `content` | `document.page_content` | `sources[].content` |
| `document` | `metadata.source` basename | `sources[].source` |
| `page` | `metadata.page` (zero-based) + 1 | `sources[].page` (already one-based) |
| `section` | `metadata.section` | `sources[].section` |
| `chunk_id` | `metadata.chunk_id` | `sources[].chunk_id` |
| `score` | `evidence_score` from the result pair | `sources[].score` |

A field the upstream result does not supply stays `None`. The `score` is never
reinterpreted, rescaled or converted.

### Score semantics

The RAG engine returns different quantities depending on the retrieval mode, so
every score travels with a label and a direction. `app/integrations/rag/scores.py`
derives them from what the retrieval result actually reported.

| Retrieval mode | Fragment source | `score_semantics` | `higher_is_better` |
| -------------- | --------------- | ----------------- | ------------------ |
| `lexical` | BM25 | `lexical_relevance_flag` | `false` |
| `vector` (light backend) | TF-IDF | `vector_cosine_distance` | `false` |
| `vector` (full backend) | vector store | `vector_store_distance` | `false` |
| `hybrid` | appeared on the vector side | `vector_cosine_distance` / `vector_store_distance` | `false` |
| `hybrid` | lexical only | `lexical_relevance_flag` | `false` |
| unrecognised mode | any | `null` | `null` |

Three properties are deliberate and test-enforced:

1. **No direction inversion.** A distance is never rewritten into a similarity,
   so no annotated score carries `higher_is_better=true`.
2. **Hybrid is heterogeneous.** A single hybrid response can contain both a
   cosine distance and a binary lexical flag, because the fusion step lets each
   fragment inherit the score of whichever side populated it. The RRF fused score
   (which is higher-is-better) is not the value the result tuple carries, so it
   is not exposed as the relevance score.
3. **No cross-semantics ranking.** `comparable()` exists so a caller can check
   that a result set shares one meaning before ordering it. Ordering mixed
   semantics produces an ordering that means nothing.

### Building a knowledge base

The knowledge base is produced by the RAG project's own ingestion and indexing
entry point. Nothing is reimplemented here.

```bash
# 1. Build the light index from a corpus directory of real PDFs.
python scripts/rag_build_knowledge_base.py \
  --rag-repo-root D:/industrial-knowledge-rag \
  --corpus-dir D:/industrial-knowledge-rag/backend/evaluation/benchmark_private/documents

# 2. Run retrieval-only queries against the built index.
python scripts/rag_retrieval_probe.py \
  --rag-repo-root D:/industrial-knowledge-rag \
  --query "PowerFlex 520 drive F007 motor overload" \
  --top-k 5
```

Both scripts are read-only with respect to the RAG repository source, Git
metadata and configuration. The single artifact written is the light index JSON
under the RAG repository's gitignored `backend/light_indexes/`. Neither script
prints environment variable values; if importing the RAG package pulls the RAG
`.env` into the process, they report `rag_env_side_effect_detected` and the names
of the newly introduced keys only.

### Configuration

```bash
RAG_PROVIDER=local          # local | http
RAG_REPO_ROOT=              # required for local; no default path is assumed
RAG_BACKEND=light           # light (light_rag_core) | full (rag_core)
RAG_KNOWLEDGE_BASE_ID=default
RAG_RETRIEVAL_MODE=hybrid   # lexical | vector | hybrid
RAG_TOP_K=4
RAG_BASE_URL=               # required for http; no default host or port
RAG_TIMEOUT_SECONDS=10
RAG_HTTP_MODEL_PROVIDER=DeepSeek
```

Nothing about the host, port, timeout, knowledge base id or repository path is
hardcoded. A misconfigured provider raises at construction time, and the tool
surfaces it through `error` rather than returning an empty result.

```bash
pip install -r requirements-rag-local.txt   # only for RAG_PROVIDER=local
```

## Query parsing

`app/agent/parser.py` is a deterministic rule-based parser. It performs no LLM
or network call. It extracts two entities and classifies the request. All device
catalog knowledge lives in `app/services/device_catalog.py`
(`resolve_device_alias`, `canonicalize_device_id`); the parser only handles text.

| Output | Rule |
| ------ | ---- |
| `equipment_id` | `PREFIX-123` pattern, canonicalized through `data/devices.json` so `plc-001` and `robot-001` map back to `PLC-001` and `Robot-001`. Device names act as aliases, so `包装线PLC` resolves to `PLC-001`. |
| `alarm_code` | `LETTER1234` pattern, upper-cased. |
| `intent` | `alarm_diagnosis`, `maintenance_advice`, `device_status` or `unknown`, chosen by a fixed keyword precedence. |

```python
from app.agent.parser import parse_query

parsed = parse_query("包装线PLC报警F0045怎么办")
parsed.equipment_id  # 'PLC-001'
parsed.alarm_code    # 'F0045'
parsed.intent        # <Intent.ALARM_DIAGNOSIS: 'alarm_diagnosis'>
```

Identifiers that match a pattern but are absent from the catalog are returned
verbatim, so the device tool can report `found=False` rather than the parser
silently dropping them. When several candidates appear, the first match wins.

## Agent workflow

`app/agent/graph.py` defines a linear LangGraph topology with five nodes. The
graph is compiled lazily through `get_graph()` so importing the module stays
cheap.

| Order | Node | Behaviour |
| ----- | ---- | --------- |
| 1 | `route_query` | Parses the query and fills `intent`, `equipment_id` and `alarm_code`. Caller-supplied values survive when the parser resolves nothing. |
| 2 | `plan_actions` | Emits `required_tools` using registry names. Selects the planner named by `PLANNER_MODE`; the rule planner below is the frozen default. |
| 3 | `execute_tools` | Runs exactly the planned tools. With an explicit LLM call list it uses the planner's arguments, validated against the tool's own input model; otherwise it derives arguments from state. It never adds, reorders or repairs a call. Unknown names or invalid arguments are reported through `error`. |
| 4 | `retrieve_context` | Standardizes `tool_results` into `Evidence` records. Pure post-processing: it contacts no external system. |
| 5 | `synthesize` | Renders the Chinese `final_answer` from `tool_results` only, including the maintenance manual section. No LLM writes the answer. |

With `PLANNER_MODE=rule` (the default) dispatch is deterministic and involves no
LLM at all.

```python
from app.agent.graph import get_graph

final = get_graph().invoke({"query": "包装线PLC报警F0045怎么办"})
print(final["required_tools"])
# ['get_device_status', 'query_alarm_code', 'search_maintenance_manual']
print(final["final_answer"])
```

### Rule planner rules

The frozen baseline, used when `PLANNER_MODE=rule` and as the fallback in `auto`:

| Condition | Tool scheduled |
| --------- | -------------- |
| `equipment_id` present | `get_device_status` |
| `alarm_code` present | `query_alarm_code` |
| `intent` is `alarm_diagnosis` or `maintenance_advice` | `search_maintenance_manual` |

### Argument mapping

In rule mode the executor derives arguments from state through adapters that live
in the agent layer, so the tools stay unaware of the state shape. In LLM mode the
planner supplies the arguments and the same input models validate them.

| Tool | State field | Tool argument |
| ---- | ----------- | ------------- |
| `get_device_status` | `equipment_id` | `device_id` |
| `query_alarm_code` | `alarm_code` | `alarm_code` |
| `search_maintenance_manual` | `query` (falls back to `equipment_id` + `alarm_code`) | `query` |

### Synthesis and evidence

`retrieve_context` builds one `Evidence` record per tool result, and
`search_maintenance_manual` hits become `source_type=document` records that
populate `document`, `page` and `score`.

The answer has three sections: `【设备状态】`, `【报警信息】` and
`【维护手册证据】`. Only facts returned by the tools are used. A miss produces an
explicit `未找到` statement, a manual retrieval that returned nothing says so,
and an unavailable knowledge base is reported as unavailable. Cause and action
sections are omitted rather than invented, and an unavailable retrieval produces
no citable document evidence.

## LLM planner (optional)

The LLM planner is additive. It is off by default, it is not required for any
behaviour in this document, and enabling it does not change the executor, the
evidence model or the synthesis step.

The division of labour is the whole design:

| Stage | Who decides | LLM involved |
| ----- | ----------- | ------------ |
| Understanding | `route_query`, rule-based parser | no |
| Planning | rule planner or LLM planner | only in `llm` / `auto` |
| Execution | `execute_tools` | no |
| Evidence | `retrieve_context` | no |
| Answer | `synthesize` | no |

The model chooses tools and arguments. It cannot produce a diagnosis, an answer
or an evidence record, because the plan schema has no field for them.

### Modes

| `PLANNER_MODE` | Behaviour on planning failure |
| -------------- | ----------------------------- |
| `rule` (default) | Not applicable. The frozen V0.4 planner runs; no provider is built. |
| `llm` | The failure is raised and returned to the caller with its code. There is no fallback. |
| `auto` | The rule planner runs, `planner_fallback` becomes `true` and the reason is recorded. |

`llm` deliberately does not fall back. A silently downgraded plan is
indistinguishable from a correct one, so a caller who asked for the LLM planner
gets an error rather than a different planner's answer. `auto` is where a
degraded run is acceptable, and there it is labelled as such in both the response
and the log.

Every path records `planner_used`, `planner_fallback` and, on a fallback,
`planner_fallback_reason` in the state. A fault in the dispatcher itself is
recorded as `UNEXPECTED_PLANNER_ERROR` rather than mislabelled as a provider
fault.

### Provider

`app/integrations/llm/` talks to any OpenAI-compatible
`POST {LLM_BASE_URL}/chat/completions` endpoint over `httpx`. No vendor SDK is
imported, so switching endpoints is a configuration change. The provider's only
job is to send messages and return text; it performs no planning and no
validation, which keeps the planning contract testable without a network.

Three properties are load-bearing and test-enforced:

1. The key is held as a `SecretStr` and read only at the moment the request header
   is built. It never appears in `describe()`, in a `repr`, in the state, in a
   response body or in a log line.
2. Error text is built from the HTTP status and the transport exception's class
   name only. A response body can echo the request and a header carries the
   credential, so neither is ever included.
3. A bad status, an unparsable envelope or a missing completion text raises. The
   provider never returns an empty completion, which a planner could otherwise
   read as "the model planned nothing".

### Tool schemas have one definition

Each registry entry carries the tool's own Pydantic input model. The JSON Schema
handed to the model is produced by `model_json_schema()` from that same model, and
the executor validates the planner's arguments against the same model again.
Adding a parameter to a tool therefore changes the tool's validation, the prompt
and the executor's check at once. There is no second hand-written parameter list
that can drift.

`tests/test_planner_modes.py` asserts this identity directly, and asserts that
every registered tool declares an input model.

```python
from app.agent.planners.tool_schemas import available_tool_definitions

available_tool_definitions()[0]
# {'name': 'get_device_status', 'description': '...', 'parameters': {...}}
```

### Validation chain

A model response is usable only after it clears four gates, in order:

1. the text parses as a JSON object (a surrounding code fence or prose is
   tolerated, since a wrapper does not make the payload wrong);
2. the object matches `AgentPlan`, which forbids undeclared fields;
3. every named tool exists in the registry;
4. every argument set satisfies that tool's input model, with undeclared
   arguments rejected rather than ignored.

A failure at any gate raises a `PlannerError` with one of five codes:
`LLM_PROVIDER_ERROR`, `LLM_TIMEOUT`, `INVALID_PLANNER_OUTPUT`, `UNKNOWN_TOOL`,
`TOOL_ARGUMENT_VALIDATION_FAILED`. Nothing is repaired, defaulted or guessed: a
plan that needs repair is a plan the planner refuses.

### Configuration

```bash
PLANNER_MODE=rule           # rule | llm | auto
LLM_BASE_URL=               # API root, e.g. https://host/v1. No host is assumed
LLM_API_KEY=                # secret; never commit and never log
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0
```

With `PLANNER_MODE` left at `rule` the three `LLM_*` variables are unused and the
pipeline is entirely LLM-free. In `llm` or `auto` mode an incomplete
configuration surfaces as `LLM_PROVIDER_ERROR` naming the missing variable,
instead of a request to an invented endpoint.

### Smoke test

```bash
python -m scripts.llm_planner_probe
```

Run it as a module so the project root is importable. The probe issues exactly one
planning call and reports the validated plan, the provider metadata and the
measured latency. With no provider configured it reports
`REAL_LLM_GATE_NOT_RUN` and makes no call: it does not fabricate a plan or a
latency figure. It never prints the prompt, the raw completion or the key.

### Known limits

1. **No fallback in `llm` mode.** A provider outage makes `POST /agent/invoke`
   return 500. Use `auto` where availability matters more than a guaranteed LLM
   plan.
2. **Single-shot planning.** The planner gets one attempt and no tools are
   executed between planning and execution, so there is no re-planning from tool
   output. A plan that names an unavailable tool is reported, not retried.
3. **No tool-call history in the prompt.** The prompt carries the tool schemas, the
   device vocabulary and the user's query. It does not carry prior turns or prior
   tool results, so multi-turn reasoning is out of scope.
4. **Schema support varies.** The `response_format` hint is advisory; the schema
   is enforced by post-call validation rather than by the endpoint. The
   validation chain above is the guarantee, not the endpoint.
5. **Environment proxies apply.** The provider uses `httpx` defaults, so
   `HTTP_PROXY`/`NO_PROXY` are honoured. A local endpoint needs `NO_PROXY` to
   cover `127.0.0.1`.
6. **Planner latency is added latency.** Planning happens before execution, so it
   adds to the request. The provider call is included in the reported
   `latency_ms`.
7. **No authentication on the endpoint.** `/agent/invoke` is unauthenticated and
   must not be exposed beyond localhost.

## Testing

```bash
pytest
ruff check app tests evaluation scripts
ruff format --check app tests evaluation scripts
mypy app tests evaluation
```

The RAG provider tests exercise both providers against contract-faithful
payloads, so no test needs a live RAG service or a built knowledge base. The API
tests stub the provider for the same reason and additionally pin the request
bounds, the response contract, the failure body and the log fields.

The planner tests are hermetic in the same way. `tests/fake_llm.py` supplies a
provider double that records the requests it received, so the planner, the mode
dispatcher and the API can all be driven without a network or a credential. No
planner test opens a socket.

Two test details are worth knowing. The API tests seed an in-memory SQLite
database through `StaticPool`: the route runs in a worker thread, and an
in-memory database is otherwise private to the connection that created it.
`tests/test_pipeline.py` needs no such pin because it calls the graph directly.
And `app/services/__init__.py` exports the agent-dependent service lazily, because
the agent layer imports `app.services.device_catalog`; an eager import there would
make the import order decide whether a module loads.

`tests/test_evaluation.py` is where the framework is tested. Most of it drives the
metric primitives and the per-case evaluator with synthetic observations, so it
needs neither a database nor a network. One module-scoped fixture runs the real
rule planner over the real dataset once, and the checks then read that single
report. That fixture is deliberate: the honest failures of the frozen baseline are
pinned by case id, so a change that makes them pass fails the suite.

`tests/test_planner_comparison.py` covers the comparison. It checks that the metric
table keeps every metric including the ones where the LLM reads worse, that the delta
sign is unambiguous on the `lower_is_better` metrics, that a mismatched dataset is
refused with both hashes named, and that a missing LLM baseline produces a refusal
instead of a file of nulls. No evaluation test contacts a provider, so the whole
suite stays hermetic and runs without a credential.

### Manual check

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8123
curl -s http://127.0.0.1:8123/health
curl -s -X POST http://127.0.0.1:8123/agent/invoke \
  -H "Content-Type: application/json" -d '{"query": "PLC-001 现在什么状态"}'
```

## Agent evaluation (V0.6 and V0.7)

A software test and an agent evaluation answer different questions, and this
project keeps them apart. `pytest` can assert that the rule planner returns
`["get_device_status"]` for a query. It cannot tell you how often that is the right
answer, how often the planner reaches for a tool nobody asked for, or whether it
invents a tool name. The framework in `evaluation/`, at the repository root,
measures those.

The `evaluation/` package at the repository root is the only Agent Evaluation
implementation in this project. An earlier `app/evaluation/` package held a
V0.2 placeholder scorer and has been removed. Nothing imports it, and no runtime
path reached it.

```bash
# Planner only: fast, touches no tool, roughly 0.2 ms per case.
python -m evaluation.runner --planner rule

# End to end: adds execution, evidence and synthesis, and records their latency.
python -m evaluation.runner --planner rule --execute-tools --tag rule_e2e

# LLM planner, when a provider is configured.
python -m evaluation.runner --planner llm
python -m evaluation.runner --planner auto
```

Two run modes exist because they answer different questions. A planner-only run
scores the plan and never executes a tool, so the roughly five-second manual
retrieval never runs and the whole dataset costs a fraction of a second. This is
the mode to use when comparing planners. An end-to-end run continues into the
executor, the evidence step and the synthesis step, and additionally records
execution and retrieval latency.

### Dataset

`evaluation/dataset.json` holds 49 hand-authored cases. The answer key was fixed in
advance: the `ground_truth_policy` block states the seven rules (`P1` to `P7`) that
decide what each query should require, and the cases were written against those
rules. No case was tuned to make the rule planner look better or worse.

| Category             | Cases | What it covers |
| -------------------- | ----- | -------------- |
| `device_status`      | 8     | Device condition questions, grounded on the device record |
| `alarm_diagnosis`    | 8     | Alarm codes, from a bare code lookup to a full fault-handling request |
| `maintenance_advice` | 6     | Device-specific maintenance or handling advice |
| `rag_only`           | 6     | Knowledge with no structured coverage, answerable only from the manual |
| `multi_tool`         | 8     | Requests that need more than one tool |
| `ambiguous`          | 5     | Under-specified or terse requests |
| `ood`                | 8     | Out-of-domain requests that must not trigger an industrial tool call |

Each case carries `id`, `category`, `query`, `expected_intent`, `expected_tools`
and, where the arguments are known, `expected_arguments`. An out-of-domain case
expects no tool and no intent. Loading validates the answer key: ids must be
unique, an intent must come from the parser vocabulary, `expected_arguments` may
only name a tool the case expects, and an out-of-domain case must agree with
itself.

### Metrics

Ten metrics are reported. Their definitions ship inside every report, so a number
is always read next to what it means.

| Metric | Definition |
| ------ | ---------- |
| `intent_accuracy` | Cases whose planned intent matches, over non-out-of-domain cases |
| `tool_selection_exact_match` | Cases whose tool set matches exactly, over all cases |
| `tool_precision` | True-positive selections over all selections, pooled (micro) |
| `tool_recall` | True-positive selections over all expected tools, pooled (micro) |
| `argument_accuracy` | Declared-argument checks that passed, over all checks |
| `invalid_tool_rate` | Selected names absent from the registry, over all selections |
| `unnecessary_tool_call_rate` | Selected tools the case did not expect, over all selections |
| `task_success_rate` | Plans that are fully correct, over all cases |
| `planner_failure_rate` | Cases where planning raised, over all cases |
| `average_planning_latency_ms` | Mean planning wall time, over measured cases |

`execution_error_rate` is an eleventh metric, present only in an end-to-end run.

Three definitional choices are load-bearing, and each one closes a way for a
benchmark to flatter itself.

1. **Task success is judged on the plan.** A case succeeds when the intent, the
   tool set and every declared argument are correct, no invalid tool was named and
   planning did not fail. Execution outcomes are reported through
   `execution_error_rate` and the failure records. A planner-only run and an
   end-to-end run of the same planner therefore stay directly comparable.
2. **A zero denominator is `null`.** A planner that never selects a tool has no
   precision, and reporting `1.0` for it would be a fabricated success. Every
   denominator is counted explicitly in the report's `denominators` block, so each
   ratio can be recomputed from the published numbers.
3. **Out-of-domain cases are excluded from intent accuracy.** There is no
   maintenance intent to classify. Their signal is the `ood_safety` block, which
   reports how many stayed silent and names the ones that did not.

### Rule planner baseline

Measured on 2026-09-15 by running the frozen rule planner over the 49 shipped
cases, planner-only, dataset SHA-256 `af873b6c…`. These are recorded measurements.

| Metric | Value | Counts |
| ------ | ----- | ------ |
| `intent_accuracy` | 0.9756 | 40 / 41 |
| `tool_selection_exact_match` | 0.7959 | 39 / 49 |
| `tool_precision` | 0.8800 | 66 / 75 |
| `tool_recall` | 0.9851 | 66 / 67 |
| `argument_accuracy` | 1.0000 | 42 / 42 |
| `invalid_tool_rate` | 0.0000 | 0 / 75 |
| `unnecessary_tool_call_rate` | 0.1200 | 9 / 75 |
| `task_success_rate` | 0.7959 | 39 / 49 |
| `planner_failure_rate` | 0.0000 | 0 / 49 |
| `average_planning_latency_ms` | 0.2103 | 49 samples |

For a fixed dataset and planner the nine behavioural metrics above are
deterministic. Planning latency is wall time on this machine and moves by a few
hundredths of a millisecond between runs, so the recorded distribution is published
instead of a single figure.

| Statistic | `planning_latency_ms` |
| --------- | --------------------- |
| samples | 49 |
| mean | 0.2103 |
| median | 0.197 |
| p95 | 0.2594 |
| min | 0.181 |
| max | 0.500 |

The p95 interpolates between order statistics, the convention
`evaluation.metrics.percentile` documents, because a moved convention would move the
reported tail. This series covers planner time only: tool execution, retrieval and
HTTP API latency are excluded. In a planner-only run execution and retrieval latency
have no samples, so every one of their statistics is `null`, never zero.

In an end-to-end run the planning metrics are identical, and
`execution_error_rate` is 0.6735 (33 / 49). That figure describes the environment
rather than the planner. No retrieval provider can be built here, so the manual
search tool reports unavailability, and retrieval latency stays `null` instead of
being estimated.

Ten cases fail, and one known weakness explains all of them. The rule planner is
keyword-driven, so it reaches for the manual search whenever a maintenance-sounding
word appears, even when a structured tool already answers the question.

| Primary failure type | Cases |
| -------------------- | ----- |
| `unnecessary_tool` | `ad-002`, `ad-004`, `ad-006`, `ad-008`, `mt-007`, `am-004` |
| `missing_tool` | `ro-003`, which also carries `intent_mismatch` |
| `unexpected_tool_call_on_ood` | `ood-006` (car engine), `ood-007` (phone battery), `ood-008` (home air conditioner) |

The three out-of-domain offenders are keyword bait: consumer and automotive
queries that share vocabulary with industrial maintenance. The other five
out-of-domain cases are handled correctly, so the agent stays silent on 5 of 8. The
`alarm_diagnosis` failures share a cause. A bare alarm lookup such as `F0112 是什么
意思` is fully answerable from the alarm catalog, which already carries the name,
severity, category, causes and recommended actions, so the policy does not require
a manual search, and the rule planner issues one anyway.

These cases are pinned by identifier in `tests/test_evaluation.py`. A change that
makes them pass fails that suite, which is the intended reading: the answer key
moved.

### Rule versus LLM planner evaluation

Both planners are measured on one frozen dataset. The dataset holds **49 cases** and
its SHA-256 is **`af873b6c28bd44186dd580369b3120f9b8b4c5d6d1c13ec4a55e0f72d220f0b7`**.
The recorded rule baseline was produced from that exact file, and
`tests/test_evaluation.py` fails if the shipped dataset ever stops matching the hash
the baseline names. Nothing here is comparable across a dataset change.

**LLM Evaluation: measured.** The LLM planner ran against a real external provider
(`openai_compatible`, model `deepseek-flash`) over the same frozen 49-case dataset,
planner-only, with no tool execution. The provider, endpoint and model are recorded in
`planner_runtime.provider` inside `llm_baseline.json`. The API key is never recorded:
the provider reads it from a `SecretStr` only when it builds the request header, and
its error text is built from the status code alone.

| Metric | Rule planner | LLM planner | delta (LLM - Rule) |
| ------ | ------------ | ----------- | ------------------ |
| `intent_accuracy` | 0.9756 (40/41) | 0.9756 (40/41) | 0.0000 |
| `tool_selection_exact_match` | 0.7959 (39/49) | 0.8163 (40/49) | +0.0204 |
| `tool_precision` | 0.8800 (66/75) | 0.9394 (62/66) | +0.0594 |
| `tool_recall` | 0.9851 (66/67) | 0.9254 (62/67) | -0.0597 |
| `argument_accuracy` | 1.0000 (42/42) | 1.0000 (37/37) | 0.0000 |
| `invalid_tool_rate` | 0.0000 (0/75) | 0.0000 (0/66) | 0.0000 |
| `unnecessary_tool_call_rate` | 0.1200 (9/75) | 0.0606 (4/66) | -0.0594 |
| `task_success_rate` | 0.7959 (39/49) | 0.7959 (39/49) | 0.0000 |
| `planner_failure_rate` | 0.0000 (0/49) | 0.0000 (0/49) | 0.0000 |
| `average_planning_latency_ms` | 0.2103 (49 samples) | 1551.94 (49 samples) | +1551.73 |
| `planning_latency_ms` p95 | 0.2594 | 2365.45 | +2365.19 |

The comparison is not a clean win for either side, and it should not be read as one.
`task_success_rate` and `intent_accuracy` are identical. The LLM buys exact matches and
fewer unnecessary calls by trading away recall: it misses five expected tools that the
rule planner selects. The price is latency, where one provider call costs roughly four
orders of magnitude more than the in-process rule planner. `delta` is always
`llm - rule`; the error rates and the latency carry `lower_is_better`, so the two
negative deltas above are improvements and the two positive latency deltas are a
regression. The full per-metric verdicts live in `planner_comparison.json`.

Failures, rule planner. Ten cases, and one known weakness explains all of them.

| Primary failure type | Cases |
| -------------------- | ----- |
| `unnecessary_tool` | 6: `ad-002`, `ad-004`, `ad-006`, `ad-008`, `mt-007`, `am-004` |
| `missing_tool` | 1: `ro-003`, which also carries `intent_mismatch` |
| `unexpected_tool_call_on_ood` | 3: `ood-006`, `ood-007`, `ood-008` |

Failures, LLM planner. Ten cases, and the pattern differs from the rule planner.

| Primary failure type | Cases |
| -------------------- | ----- |
| `missing_tool` | 5: `ma-002`, `ma-003`, `ma-005`, `ma-006`, `mt-007` |
| `unnecessary_tool` | 4: `ds-002`, `ad-004`, `ad-006`, `ad-008` |
| `intent_mismatch` | 1: `mt-005` |
| `unexpected_tool_call_on_ood` | 0 |

The rule planner's dominant weakness was over-calling the manual search; the LLM's is
the opposite, under-calling it. Four of the five `missing_tool` cases are
`maintenance_advice` queries where the LLM answered from `get_device_status` alone and
skipped retrieval. Its `unnecessary_tool` cases overlap the rule planner's, but the
count is lower. The failure report is `llm_failures.json`.

Out-of-domain performance, rule planner, over the 8 out-of-domain cases.

| OOD measure | Rule planner | LLM planner |
| ----------- | ------------ | ----------- |
| `tool_call_rate` | 0.375 (3/8) | 0.000 (0/8) |
| `unnecessary_tool_call_rate` | 0.375 (3/8, same number by construction) | 0.000 (0/8, same number by construction) |
| silent cases | 5 | 8 |
| offending cases | `ood-006`, `ood-007`, `ood-008` | none |

This is the LLM planner's clearest advantage. It stays silent on all eight
out-of-domain cases, where the rule planner answers three of them with a tool call.
The `tool_call_rate` delta is `-0.375`. The two rates stay one number per planner for
the reason above; an out-of-domain case expects no tool, so every selected tool is
unnecessary by construction.

The two out-of-domain rates are one number, not two. An out-of-domain case expects
no tool, so every selected tool is by definition unnecessary and the two rates
coincide. The dataset-wide `unnecessary_tool_call_rate` in the table above covers all
49 cases and is a different quantity.

To produce the LLM column:

```bash
# 1. Configure a provider. Names only; the values are never logged or committed.
#    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

# 2. Benchmark the real LLM planner, planner-only, over the same frozen dataset.
python -m evaluation.runner --planner llm --dataset evaluation/dataset.json

# 3. Compare. Writes evaluation/reports/planner_comparison.json.
python -m evaluation.comparison
```

Two gates protect the comparison. `evaluation.runner --planner llm` refuses to run
without a provider, writes `llm_evaluation_status.json` with
`status: "LLM_EVALUATION_NOT_RUN"` and `metrics: null`, and exits 0.
`evaluation.comparison` then finds no successful LLM baseline, reports
`LLM_EVALUATION_NOT_RUN`, exits 2, and writes **no** `planner_comparison.json`. Nulls
are never published under a comparison heading.

When a real baseline does exist, the comparison reports every metric from both sides
and `delta = llm - rule`, and it never filters. The error rates and the planning
latency carry a `lower_is_better` flag so a positive delta on those is not misread as
an improvement. A delta whose value is `null` on either side has no verdict, because
`null` means the metric was not measured for that planner. Two baselines measured on
different datasets are refused with both hashes named. The rule side is read from the
recorded artefact and is never re-run, so a more favourable draw cannot be selected
after the fact.

### End-to-end subset

A fixed 12-case subset (two per category across `device_status`, `alarm_diagnosis`,
`maintenance_advice`, `rag_only`, `multi_tool` and `ood`) runs the whole pipeline with
the real LLM planner and with tool execution, to check that planning survives contact
with execution. It is a separate run mode and is **not** folded into the planner-only
comparison.

| Measure | Value |
| ------- | ----- |
| cases | 12 |
| `planning_latency_ms` mean | 1565.44 |
| `execution_latency_ms` mean | 1.29 |
| `rag_latency_ms` | `null` (0 samples) |
| `total_latency_ms` mean | 1566.77 |
| evidence items | 10 across 12 cases |
| cases with an execution error | 8 / 12 |
| cases with a rendered answer | 12 / 12 |

Every execution error is the same one: `search_maintenance_manual` reports that the
RAG repo root is required. That is the documented environment limit, not a planner
defect, and it is why retrieval latency has no samples. The raw per-case record is
`llm_e2e_subset.json`.

### Reports

Reports are written to `evaluation/reports/`. A planner-only run writes
`<planner>_baseline.json` and `<planner>_failures.json`, so the rule and LLM baselines
sit side by side as `rule_baseline.json` / `rule_failures.json` and
`llm_baseline.json` / `llm_failures.json`. An end-to-end run writes
`rule_e2e_baseline.json` and `rule_e2e_failures.json`. The comparison writes
`planner_comparison.json`, or `planner_comparison_status.json` when it refuses. A unit
of the smoke and end-to-end subset checks write `llm_smoke.json` and
`llm_e2e_subset.json`; both are marked as their own run modes and are never folded
into the planner-only comparison. A gate report named `<planner>_evaluation_status.json`
is written only when a run refuses because the provider is missing. Every baseline
records the dataset path and SHA-256, the planner mode, the run mode, the git commit,
the registry contents and the application version, so a number can always be traced
back to the input that produced it.

### Known limits

1. **The dataset is a hand-authored reference, not a public benchmark.** 49 cases
   support the claim that the rule planner is right on 39 of these 49. They do not
   support a general accuracy figure.
2. **The plan is scored, and the prose is not.** The framework measures whether the
   right tools were chosen with the right arguments. It does not score the answer
   that the synthesis step writes.
3. **Eight out-of-domain cases can show a weakness, not bound its rate.** A larger
   adversarial set is the only way to turn `0.375` into a defensible estimate.
4. **The LLM column is one run of one model.** It is a real external provider
   (`openai_compatible`, model `deepseek-flash`), recorded in `llm_baseline.json`, but a
   single run carries provider-side variance that one baseline cannot express.
   Temperature is fixed at 0, which reduces that variance without removing it.
5. **One dataset and one run per planner.** A comparison over 49 hand-authored cases
   bounds neither planner on other queries, and a single LLM run carries provider-side
   variance that one baseline cannot express. A repeated run would be needed before
   any latency claim about a provider.

## Roadmap

1. Add device read endpoints backed by the `Device` model.
2. Add an Alembic migration for schema versioning.
3. Repeat the LLM benchmark across several runs to turn the single recorded baseline
   into a distribution, then report provider-side latency variance instead of one
   sample. The recorded baseline is a single run.
4. Reduce manual retrieval latency. Measured against the four-document Rockwell
   corpus (4219 chunks), a manual query costs about 4.9 s in steady state, and the
   first query in a fresh process costs about 7.3 s while the module import and
   index load are paid. The light backend re-reads the index and re-fits its
   TF-IDF model per call. Caching belongs in the integration layer; the reported
   `latency_ms` states the real cost in the meantime.
5. Add multi-turn planning: carry prior tool results into the planner prompt so a
   follow-up can build on what the previous turn retrieved.
6. Add authentication to `/agent/invoke` before it is exposed beyond localhost.
