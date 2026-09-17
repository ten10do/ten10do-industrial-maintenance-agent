# Demo guide

This document backs the three images in the README. It records which runtime
produced each one, the exact commands to reproduce it, and the boundary of what
the capture does and does not demonstrate.

Every value in the images comes from a real v0.9.0 container. Nothing was typed by
hand, no number was adjusted, and no field was back-filled.

## 1. Demo environment

| Item | Value |
| ---- | ----- |
| Version | 0.9.0 (`ec71eb80fd621696f4776c03a6f81df71e51bb46`) |
| Planner | `PLANNER_MODE=rule`, the deterministic default, no LLM provider configured |
| Images | `industrial-maintenance-agent:0.9.0` (384 MB) and `industrial-maintenance-agent:0.9.0-rag` (760 MB) |
| Containers | `agent` on :8000, `agent-rag` on :8001, `agent-real-data` on :8002, plus one standalone run on :8003 for the JSON log view |
| Host | Windows 11, Docker Engine 29.6.2, a laptop CPU; latencies are this machine's |

The three images share one presentation: dark terminal, one monospace stack, one
1200 px width. They are syntax-highlighted **renderings of captured text**, not
photographs of a screen. The text is what the service returned; the only
interventions were token colouring, a fixed width, a normalised font, and soft
wrapping of lines longer than 104 display columns. No value was edited.

## 2. Real-data demo

Captured from the `real-data` profile, which mounts the dataset read-only.

```bash
# The dataset is not redistributed by this repository. Obtain it first:
#   see docs/data/metropt3.md
export METROPT3_HOST_DIR=./data/external/metropt3

docker compose --profile real-data up -d agent-real-data
curl -s http://127.0.0.1:8002/health

curl -s -X POST http://127.0.0.1:8002/agent/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query": "METRO-APU-001 当前设备状态怎么样？"}'
```

What the capture shows, as returned:

| Field | Value |
| ----- | ----- |
| `intent` | `device_status` |
| `equipment_id` | `METRO-APU-001` |
| `tools_called` | `["get_device_status"]` |
| `planner_used` | `rule` |
| `evidence[0].source` | `uci_metropt3` |
| source timestamp | `2020-09-01T03:59:50` |
| `oil_temperature_c` | `59.475` |
| `motor_current_a` | `0.0424999999999995` |
| pressures | `TP2_bar -0.0139999999999993`, `TP3_bar 8.86`, `H1_bar 8.848`, `DV_pressure_bar -0.0219999999999984`, `reservoir_pressure_bar 8.864` |
| derived status | `stopped`, with the derivation stated in the answer |

The Tool schema, the tool name and the planner are unchanged from V0.8.4. The
external source is consulted first and answers only for its own identifiers, so a
seeded device such as `PLC-001` still returns its SQLite row. What that looks like
is documented in
[docs/evaluation/real_data_integration.md](evaluation/real_data_integration.md).

## 3. RAG demo

Captured from the `rag` profile, whose image carries the optional local retrieval
dependencies and mounts an external Industrial Knowledge RAG checkout read-only.

```bash
export RAG_HOST_REPO=/path/to/industrial-knowledge-rag
export RAG_HOST_PORT=8001

docker compose --profile rag up -d agent-rag

curl -s -X POST http://127.0.0.1:8001/agent/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I repair a PowerFlex 520 drive motor overload?"}'
```

What the capture shows, as returned:

| Field | Value |
| ----- | ----- |
| `intent` | `maintenance_advice` |
| `tools_called` | `["search_maintenance_manual"]` |
| provider / mode | `local`, `hybrid` |
| hits | 4 |
| hit 1 | `PowerFlex_527_User_Manual.pdf`, page 112, `Chapter 7 Troubleshooting`, score `0.7198033332824707` |
| hit 2 | `PowerFlex_520_User_Manual.pdf`, page 161, `Chapter 4 Troubleshooting`, score `0.6929119825363159` |
| hit 3 | `PowerFlex_520_User_Manual.pdf`, page 84, `Chapter 3 Programming and Parameters`, score `0.720583975315094` |
| hit 4 | `PowerFlex_520_User_Manual.pdf`, page 7, `Appendix K`, score `0.7283151149749756` |
| score semantics | `vector_cosine_distance`, lower is better |

The passage bodies are **not** reproduced in the image, in this repository or in
the response shown. The corpus is a licensed vendor publication, so the demo
carries the retrieval structure instead: document, page, section, chunk id and
score. The image marks each omitted body explicitly rather than trimming silently.

The first call in a fresh process pays the index load, so the `2824.753 ms`
latency is a cold call. The steady-state figure is in the README API example.

## 4. Observability demo

Captured from a standalone container running with the JSON log formatter.

```bash
# The compose "agent" service is identical except that LOG_FORMAT defaults to text.
docker run -d --name ima-demo-obs -p 8003:8000 \
  -e LOG_FORMAT=json \
  -e LOG_LEVEL=INFO \
  -e PLANNER_MODE=rule \
  -e METRICS_ENABLED=true \
  -e DATABASE_URL=sqlite:///./data/industrial_maintenance.db \
  -e AUTO_CREATE_TABLES=true \
  -v ima-demo-data:/app/data \
  industrial-maintenance-agent:0.9.0

docker exec ima-demo-obs python -m app.database      # seed the demo device rows

curl -s -X POST http://127.0.0.1:8003/agent/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query": "PLC-001 现在什么状态？"}'

curl -s http://127.0.0.1:8003/metrics
docker logs ima-demo-obs
```

One request, from one process, and the three views agree:

| View | Value |
| ---- | ----- |
| API response `latency_ms` | `7.568` |
| `industrial_agent_request_duration_seconds_sum` | `0.007567999999999999` s |
| `tool.completed` log `duration_ms` | `1.977` |
| `industrial_agent_tool_duration_seconds_sum` | `0.001977` s |
| `industrial_agent_requests_total` | `{intent="device_status",planner="rule",status="success"} 1.0` |
| samples advanced | 53, across the request, planner and tool families |

The log record carries `query=<redacted>`; the raw query text is not written to
the log line and the size is recorded instead as `query_chars`. Metric labels come
from a frozen vocabulary, so no request id, device id or query text can become a
time series. The metric catalog and the cardinality bound are in
[observability.md](observability.md).

Metrics are held in the process. Restarting the container resets every series, so
a fresh process lists the families with their `HELP` and `TYPE` lines and no
samples until something is observed.

## 5. What each demo proves

**Real device data.** That the production tool answers a real identifier from a
real industrial dataset, end to end through the HTTP surface, and that the
provenance travels with the answer: the source name, the upstream timestamp and
the reading names all appear in the response and its `evidence` record.

**Industrial RAG.** That a natural-language maintenance question is planned into
the retrieval tool, that retrieval returns page-level hits from a real vendor
manual, and that the answer is rendered from those hits with the document, page
and score attached rather than generated from memory. The score semantics and
direction are reported alongside the number, so a reader cannot mistake a distance
for a confidence.

**Observability.** That one request is traceable across logs and metrics at the
same instant, that the numbers in the three views reconcile, and that the
instrumentation is bolted on from outside: the planner, tool, graph and
integration modules import no metrics or tracing library.

**Quality and delivery.** That the change reached `main` through a branch, a pull
request and the required `Quality gates (Python 3.11)` check, with no direct push
to a protected branch.

## 6. What each demo does NOT prove

**Real device data is not condition monitoring.** The MetroPT-3 readings are real;
the `status` value is derived from the dataset's documented control contacts and
is labelled as derived in the answer. The dataset publishes no ground-truth
equipment state and carries no fault label, so nothing here is fault detection,
predictive maintenance or a health score, and none of it should be read as one.
The V0.8.5 integration gates check a single record against the file. See
[docs/evaluation/real_data_integration.md](evaluation/real_data_integration.md).

**RAG is not a correctness guarantee.** The demo shows retrieval evidence, not
verified maintenance advice. Whether the retrieved page is the right page for a
particular drive, revision and fault is bounded by the corpus that was indexed,
and the corpus is a snapshot. A page-level citation makes an answer auditable; it
does not make it authoritative, and the service never claims a repair is safe.

**Observability is not an SRE platform.** This is runtime instrumentation inside
one service. There is no alerting, no dashboard, no long-term metric retention, no
log aggregation backend, no sampling policy and no trace backend configured by
default. Traces are emitted only when a collector endpoint is supplied and the
optional SDK is present in the image. Enabling `OTEL_ENABLED=true` on the default
image logs `otel_sdk_missing` and stays disabled rather than failing.

**The benchmark narrows what any planner can claim.** The 49 cases are
hand-authored for this repository. They bound the two planners on this dataset,
this registry and this provider endpoint. The demo does not extend those bounds.

## 7. Reproducing the images

The capture and rendering pipeline is not part of the tracked repository. What is
required to reproduce a capture is a running container and the commands above.
Latencies and metric values will differ on another machine, and a fresh process
starts every counter at zero, so the screenshots are a record of one run rather
than a target to match.
