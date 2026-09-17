# Observability

How the service reports what it is doing, and what it deliberately refuses to
report. Three capabilities, all of them optional except the log line that was
always there:

| Capability | Switch | Default | Endpoint or sink |
|---|---|---|---|
| Structured logging | `LOG_FORMAT` | `text` | stdout, via `app/logging_config.py` |
| Prometheus metrics | `METRICS_ENABLED` | `true` | `GET /metrics` |
| OpenTelemetry tracing | `OTEL_ENABLED` | `false` | OTLP/HTTP, only when an endpoint is named |

The layer is **decoupled from the agent's decisions**. It observes the pipeline
from the outside. It does not plan, does not choose tools, does not render an
answer and does not alter state the pipeline reads back. The rule this release
was built to satisfy is that the frozen benchmark results are byte-identical with
the layer on and off, which
[`docs/evaluation/observability_overhead.md`](evaluation/observability_overhead.md)
demonstrates rather than asserts.

## 1. Architecture

```
app/main.py                    configure_logging() -> configure_observability()
                                        |
app/observability/__init__.py  ---------+-------------------------------+
                                        |                               |
                          app/observability/metrics.py        app/observability/tracing.py
                          (private CollectorRegistry)         (deferred OTel import)
                                        |                               |
                          app/observability/labels.py         app/observability/context.py
                          (frozen vocabularies)               (request_id ContextVar)
                                        \                              /
                              app/observability/instrumentation.py
                                        |
        +--------------+----------------+----------------+----------------+
        |              |                |                |                |
  agent_service    planners/        agent/graph.py   tools/manual      integrations/llm
  (request)        dispatch, llm    (tools, synth)   (RAG)             (provider, tokens)
```

Business code imports exactly one observability module,
`app.observability.instrumentation`, and only its functions and context managers.
No planner, tool, service or integration module imports `prometheus_client` or
`opentelemetry`. Three consequences follow, and each one was the reason for the
arrangement:

1. **A metric name or a label changes in one file.** `metrics.py` holds the
   declaration; `instrumentation.py` holds the mapping from an application event
   onto it. Neither is duplicated.
2. **A call site cannot invent a label.** Every label value passes through
   `labels.bounded()`, which maps anything outside a frozen vocabulary onto
   `other`. The alternative, letting call sites pass values through and testing
   for cardinality afterwards, means a leaky label reaches production before a
   test sees it.
3. **Disabling metrics removes the work, not the evidence.** The log call is
   unconditional and the metric recording is gated, so `METRICS_ENABLED=false`
   drops the series work while leaving the audit trail that predates this
   release.

`app/observability/` sits at the same level as `app/tools/` and
`app/integrations/`. It may read application state; no other layer may depend on
it for correctness. The layering rules for the rest of the tree are unchanged and
are described in the README.

## 2. Structured logging

One call site produces both output shapes.

`LOG_FORMAT=text` is the default and is **byte-for-byte the format this service
emitted before this release**: `%(asctime)s %(levelname)s %(name)s %(message)s`.
It is not a new output shape, it is the old one. Three tests pin its rendering,
so an accidental change to it fails a build rather than an operator's grep.

`LOG_FORMAT=json` emits one JSON object per line with a fixed field set:

```json
{"timestamp": "2026-09-16T10:52:31+0800", "level": "INFO", "event": "agent.request.completed", "logger": "app.observability", "message": "agent_request_completed status=success planner=rule intent=device_status duration_ms=0.907", "request_id": "6f1c...", "planner": "rule", "intent": "device_status", "status": "success", "duration_ms": 0.907}
```

The field set is closed. `FIELD_ORDER` in `app/observability/logging.py` names
every field an event may carry; a field outside it is dropped rather than
serialized, so a typo loses a field instead of adding an undocumented one.

| Field | Meaning |
|---|---|
| `timestamp` | Record creation time, `%Y-%m-%dT%H:%M:%S%z` |
| `level` | Log level name |
| `event` | Event name, from the enumerable set below |
| `logger` | Logger name |
| `message` | Human-readable rendering, scrubbed of user text |
| `request_id` | Correlation id of the invocation |
| `planner` | `rule`, `llm`, `none` or `other` |
| `intent` | Coarse intent classification |
| `tool` | Registry tool name |
| `provider` | Integration provider |
| `status` | `success`, `not_found`, `unavailable`, `error` or `other` |
| `duration_ms` | Measured duration of the operation the event describes |
| `error_type` | Exception class name, or a stable application code |
| `count` | A count that belongs to the event, such as hits or tools planned |
| `reason` | Closed failure reason, for `planner.failed` |
| `token_type` | `prompt` or `completion`, for token accounting |

### Event vocabulary

Every event this application emits is named in `logging.py`, so the set is
enumerable rather than emergent.

| Event | Emitted when |
|---|---|
| `agent.request.started` | An invocation is accepted |
| `agent.request.completed` | An invocation finishes, successfully or not |
| `planner.started` | Planning begins |
| `planner.completed` | A plan is produced |
| `planner.failed` | Planning raises, with a closed `reason` |
| `tool.started` | A tool call begins |
| `tool.completed` | A tool call returns |
| `tool.failed` | A tool call raises |
| `rag.started` | A retrieval begins |
| `rag.completed` | A retrieval returns, with or without hits |
| `rag.unavailable` | A retrieval could not run |
| `llm.started` | A provider call begins |
| `llm.completed` | A provider call returns a completion |
| `llm.failed` | A provider call fails, times out or is unreachable |

A deployment that does not enable the LLM planner never emits an `llm.*` event.
The vocabulary is what is available, not what is guaranteed per request.

## 3. Metrics catalog

`GET /metrics` renders the Prometheus text exposition format
(`text/plain; version=0.0.4; charset=utf-8`). All 11 metrics are declared in one
table, `METRIC_SPECS` in `app/observability/metrics.py`, and both the collectors
and the cardinality tests are built from it. A metric therefore cannot acquire a
label the test does not inspect.

Latency is recorded in **seconds**, following Prometheus convention, with the
unit in the name. The application measures and reports milliseconds everywhere
else; the conversion happens at exactly one place, `instrumentation._seconds`.

The `prometheus_client` library appends a `_created` timestamp gauge to a metric
family once that family has been observed at least once. A warm container
therefore exposes 18 families where 11 are declared here: the eleven below plus
seven `_created` gauges, for `requests`, `request_duration`, `planner_duration`,
`tool_calls`, `tool_duration`, `rag_requests` and `rag_duration`. The four
families with no observation yet, `planner_failures`, `llm_requests`,
`llm_duration` and `llm_tokens`, carry their `HELP` and `TYPE` lines and no
`_created` gauge.

Those gauges are generated by the client library and are not part of
`METRIC_SPECS`. Nothing in this project reads or asserts on them.

| Metric | Type | Labels | What it answers |
|---|---|---|---|
| `industrial_agent_requests_total` | counter | `planner`, `intent`, `status` | How many invocations, classified how, ending how |
| `industrial_agent_request_duration_seconds` | histogram | `planner`, `status` | Wall-clock time of one invocation, HTTP entry to response |
| `industrial_agent_planner_duration_seconds` | histogram | `planner`, `status` | Wall-clock time of the planning step alone |
| `industrial_agent_planner_failures_total` | counter | `planner`, `reason` | Planning failures by closed reason |
| `industrial_agent_tool_calls_total` | counter | `tool`, `status` | Executed tool calls by tool and outcome |
| `industrial_agent_tool_duration_seconds` | histogram | `tool`, `status` | Wall-clock time of one tool execution |
| `industrial_agent_rag_requests_total` | counter | `provider`, `status` | Retrieval calls by provider and outcome |
| `industrial_agent_rag_duration_seconds` | histogram | `provider`, `status` | Wall-clock time of one retrieval |
| `industrial_agent_llm_requests_total` | counter | `provider`, `status` | Provider calls by provider and outcome |
| `industrial_agent_llm_duration_seconds` | histogram | `provider`, `status` | Wall-clock time of one provider call |
| `industrial_agent_llm_tokens_total` | counter | `provider`, `type` | Provider-reported token usage by direction |

### Histogram buckets

Chosen per metric rather than shared, because the interesting resolution differs
by an order of magnitude between a rule-planner decision and an LLM round trip.

| Metric family | Buckets (seconds) |
|---|---|
| Request | 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30 |
| Planner | 0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 30 |
| Tool | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30 |
| RAG | 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30 |
| LLM | 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30 |

The planner row is the one that carries unusual resolution, deliberately. The
rule planner plans in about a third of a millisecond
([measurement](evaluation/observability_overhead.md)), so buckets at 100 µs and
500 µs are where its distribution actually lives. An LLM planner's planning step
is three orders of magnitude larger and lands in the `1` to `2.5` buckets.

### Status semantics

Four outcomes, applied identically to every metric so that one rule covers every
dashboard.

| Status | Meaning | Example |
|---|---|---|
| `success` | The operation completed | A device status was read |
| `not_found` | It completed and legitimately found nothing | The device id is not in the inventory; the alarm code is unknown; retrieval returned no hit |
| `unavailable` | A backing source could not be read or reached | The RAG provider raised; the device source raised; the LLM endpoint failed or timed out |
| `error` | The operation raised, or produced an unusable result | An unhandled tool exception |

The distinction between `not_found` and `unavailable` is the one most worth
keeping. A query about a device the deployment does not carry is a correct answer
to a question the service was asked. Reporting it as an error would turn a
working service into an alarming dashboard, and would make a real outage
indistinguishable from normal operation. `tool_status_from_payload()` derives the
status from the payload the tool itself returned (`error` set means
`unavailable`, `found: false` means `not_found`) rather than from an exception,
because these tools answer questions rather than raise on a miss.

Two nuances that a reader of a dashboard will otherwise have to work out.

**`requests_total` carries `success` and `error` only.** An invocation that
correctly answers "no such device" completed successfully from the caller's point
of view, and the HTTP response is `200`. The finer outcome belongs to the tool or
the retrieval that produced it, which is where `not_found` and `unavailable`
appear. A rule that watches `requests_total` for failures therefore stays quiet
during normal operation, which is the intended reading.

**A retrieval that could not be built is attributed to `unknown`.** When no
provider can be constructed at all, the code has no `provider_id` to report and
records `provider="unknown"` with `status="unavailable"` rather than naming a
provider it never reached. The reason is in the `rag.unavailable` log event, not
in the label, because a label is published to every scraper and the reason string
is not bounded.

## 4. Label and cardinality policy

A Prometheus label is a dimension of a time series. Any label value that can grow
without bound multiplies the series count, and a value taken from user input lets
a caller choose how much memory the process spends. The design used here is
**allow-listing, not escaping**.

Each label has one frozen set of legal values, declared in
`app/observability/labels.py`. `bounded(value, vocabulary)` returns the value
when it is in the set and `other` otherwise. The `other` sentinel is itself a
member of every vocabulary except `TOKEN_TYPE_LABELS`, so the fallback is always
a legal value and never a surprise series.

| Label | Vocabulary | Size |
|---|---|---|
| `planner` | `rule`, `llm`, `none`, `other` | 4 |
| `intent` | `alarm_diagnosis`, `maintenance_advice`, `device_status`, `unknown`, `none`, `other` | 6 |
| `status` | `success`, `not_found`, `unavailable`, `error`, `other` | 5 |
| `tool` | `get_device_status`, `query_alarm_code`, `search_maintenance_manual`, `none`, `other` | 5 |
| `provider` | `local`, `http`, `openai_compatible`, `none`, `unknown`, `other` | 6 |
| `reason` | `invalid_output`, `provider_error`, `timeout`, `configuration`, `unknown`, `other` | 6 |
| `type` | `prompt`, `completion` | 2 |

**Series ceiling.** The largest metric is
`industrial_agent_requests_total` at 4 × 6 × 5 = **120 series**. Because every
value is drawn from a fixed set, no request can raise the count. The total series
count for the whole registry is bounded by these products and does not depend on
traffic.

### What is forbidden, and why

`FORBIDDEN_LABEL_NAMES` in `metrics.py` names the label names that must never
appear, kept next to the specs so the prohibition and the declaration are read
together.

| Forbidden | Why |
|---|---|
| `request_id` | One series per request. Unbounded by construction. |
| `device_id`, `equipment_id` | Unbounded, and identifies which equipment a user is asking about. |
| `query`, `prompt`, `answer` | User content. Unbounded cardinality and a privacy leak in one. |
| `document`, `document_path`, `document_name`, `page`, `chunk_id` | Unbounded, and discloses which manual was read. |
| `exception`, `exception_message`, `error_message` | Unbounded, and an exception message can embed a URL and a header. |
| `url` | An LLM provider exception message carries the request URL; some transports carry the headers with it. |
| `session_id` | Unbounded. |
| `model` | Unbounded across deployments, and not needed to answer any of the questions above. |

A planner failure therefore records `reason="provider_error"`, never the provider
message. That is a deliberate loss of diagnostic detail in the metric, and the
detail is not lost overall: the class name travels in the `error_type` log field
and, at `DEBUG`, the underlying error is available where it is already handled.

`tests/test_observability.py` enforces the policy from three directions: the
forbidden names are asserted absent from every spec, each vocabulary is checked
against the set its producer can actually emit, and a request carrying a
conspicuous sentinel query, device id and document name is asserted not to appear
in the rendered exposition output.

## 5. Tracing

Off by default, with three properties that make "off" mean off.

**No import when disabled.** `OTEL_ENABLED=false` means neither
`opentelemetry-api` nor `opentelemetry-sdk` is imported. A deployment that does
not trace does not need `requirements-otel.txt` installed, and the default
container image does not carry it. Enabling tracing without the SDK installed
logs `otel_sdk_missing tracing_disabled=true` and leaves tracing off rather than
failing to start: an observability feature must not be able to stop the service.

**No implied collector.** `OTEL_EXPORTER_OTLP_ENDPOINT` has no default. Pointing
one at `localhost:4317` would turn an optional feature into a runtime dependency
on a process the operator may not have started, and the SDK's retry behaviour
would turn that into latency in the request path. An empty endpoint keeps spans
in the process, which is what the overhead measurement uses.

**Span hierarchy is the execution hierarchy.**

```
agent.invoke                       attributes: planner.type
├── planner.plan                   attributes: planner.type
├── tool.execute                   attributes: tool.name   (one per planned call)
├── rag.retrieve                   attributes: provider    (only if the tool retrieves)
├── llm.request                    attributes: provider    (only if the planner called a provider)
└── synthesize                     attributes: evidence.count
```

No span is invented for a step that did not run. A rule-planned request that
touches no RAG produces neither a `rag.retrieve` nor an `llm.request` span.

Span attributes are allow-listed by `SPAN_ATTRIBUTE_NAMES`, which is exactly
`planner.type`, `intent`, `tool.name`, `provider`, `status` and `evidence.count`.
Values pass through the same redaction the JSON logger uses.
`record_error` records an error **type name** and a redacted message; the
exception object is never handed to `record_exception`, because that call
serializes the exception's arguments and an LLM provider error's arguments carry
the URL and, on some transports, the credential-bearing headers.

## 6. Configuration

| Variable | Type | Default | Effect |
|---|---|---|---|
| `LOG_FORMAT` | `text` \| `json` | `text` | Log rendering. An unrecognised value falls back to `text`. |
| `METRICS_ENABLED` | bool | `true` | When `false`, `GET /metrics` answers `404` and no series is recorded. |
| `OTEL_ENABLED` | bool | `false` | When `false`, nothing is imported and no connection is attempted. |
| `OTEL_SERVICE_NAME` | str | `industrial-maintenance-agent` | `service.name` resource attribute. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | str | empty | OTLP/HTTP endpoint. Empty keeps spans in the process. |

`/metrics` returns `404` rather than an empty body or a `200` with zeroed series
because a scrape target that answers `200` is treated as healthy by every
collector. An operator who disabled metrics wants the target gone, and `404` is
how a target says so.

Note also what is **not** configurable: metric names, label names, label
vocabularies and buckets are code, not environment. A deployment that could
rename a metric could break the dashboard silently.

## 7. Security

The policy is that no prompt, query, answer, document body, header or credential
reaches a log, a metric, a span or a response.

**Enforced at the source.** No call site in this application passes user text or
a credential to an event. `request_started` passes `query_chars`, a length, which
answers "was this a one-word lookup or a paragraph" without carrying the text.

**Second line of defence.** `logging.sanitize()` redacts, before truncating to
200 characters, patterns for bearer tokens, `sk-` keys, GitHub tokens, AWS access
key ids, `api_key=`/`secret=`/`password=`/`token=` assignments and URLs with
embedded credentials. A non-scalar value is reduced to its type name, so a nested
provider payload cannot reach the output.

**The JSON logger scrubs the message.** The text format renders the query, and
that rendering is the historical operator-facing contract that three tests pin.
A JSON pipeline has a different contract, and the requirement there is that no
user text is carried. Both hold because `JsonFormatter` applies
`redact_message()` to every message it serializes and the text formatter does
not.

**Tracebacks are not serialized.** The JSON formatter emits the exception class
name and stops. A provider traceback can carry the request headers.

**Error text is never a label.** The reason vocabulary is closed, so an exception
message cannot become a series.

## 8. Using it

Start the service and scrape it.

```bash
docker run --rm -p 8000:8000 industrial-maintenance-agent:0.9.0
```

```bash
curl -s http://127.0.0.1:8000/metrics | head -20
```

```text
# HELP industrial_agent_requests_total Agent invocations by planner, intent and outcome.
# TYPE industrial_agent_requests_total counter
industrial_agent_requests_total{intent="device_status",planner="rule",status="success"} 5.0
industrial_agent_requests_total{intent="alarm_diagnosis",planner="rule",status="success"} 2.0
industrial_agent_requests_total{intent="maintenance_advice",planner="rule",status="success"} 1.0
# HELP industrial_agent_request_duration_seconds Wall-clock time of one agent invocation, HTTP entry to response.
# TYPE industrial_agent_request_duration_seconds histogram
industrial_agent_request_duration_seconds_bucket{le="0.005",planner="rule",status="success"} 7.0
industrial_agent_request_duration_seconds_count{planner="rule",status="success"} 8.0
industrial_agent_request_duration_seconds_sum{planner="rule",status="success"} 0.019218
```

That excerpt is copied from a running container and not retyped.

Generate traffic and watch a counter move.

```bash
curl -s -X POST http://127.0.0.1:8000/agent/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the status of PLC-001?"}' >/dev/null

curl -s http://127.0.0.1:8000/metrics \
  | grep '^industrial_agent_requests_total{.*planner="rule"'
```

Structured logs, one object per line:

```bash
docker run --rm -p 8000:8000 -e LOG_FORMAT=json industrial-maintenance-agent:0.9.0
```

A minimal scrape configuration, if you run a Prometheus server:

```yaml
scrape_configs:
  - job_name: industrial-maintenance-agent
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

No collector is required for the service to start, and none is contacted unless
an endpoint is named.

## 9. What this layer does not do

Stated so the boundary is not mistaken for an omission.

- **No dashboards or alert rules ship with the repository.** The metrics are
  produced; what to alert on is a deployment decision.
- **No log shipping.** Records go to stdout; collection is the platform's job.
- **No tracing backend.** Spans are recorded in-process unless an OTLP endpoint
  is configured, and the repository does not assume one exists.
- **No change to the agent's behaviour.** The layer observes stages the pipeline
  already had. It adds no retry, no re-plan, no fallback and no decision. The
  benchmark report in `evaluation/reports/` is unchanged and
  [the integrity check](evaluation/observability_overhead.md#5-benchmark-integrity)
  shows all nine functional metrics are identical with the layer enabled.
