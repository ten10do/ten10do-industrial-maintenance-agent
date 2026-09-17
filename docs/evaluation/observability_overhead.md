# Observability overhead

What the observability layer costs on a deterministic local workload, measured
rather than asserted. This document is intentionally separate from the frozen
benchmark reports in `evaluation/reports/`, which it does not replace or amend.

## 1. Question

`METRICS_ENABLED` and `OTEL_ENABLED` add work to the request path: a counter
increment, a histogram observation, a span. This release also adds per-stage log
events, which `METRICS_ENABLED` does not gate. Three questions follow.

1. Does that work change what the agent decides, or what the frozen benchmark
   measures? Answered separately and negatively in section 5.
2. How much wall-clock time does it add, at which stage, and is the cost
   resolvable above the noise of this host?
3. The logging is the part that is always on. How much does it cost on its own,
   and does it explain why a planning figure measured here is larger than the
   published one? Section 3.4 and section 4.2.

## 2. Method

The harness is `scripts/observability_overhead.py`. One process per
configuration, because settings are cached: switching `METRICS_ENABLED` inside a
running interpreter would measure a process that is half-reconfigured.

```
python -m scripts.observability_overhead --config off
python -m scripts.observability_overhead --config metrics
python -m scripts.observability_overhead --config metrics+tracing
```

| Configuration | `METRICS_ENABLED` | `OTEL_ENABLED` | Tracing active |
|---|---|---|---|
| `off` | false | false | no |
| `metrics` | true | false | no |
| `metrics+tracing` | true | true | yes, in-process, no exporter |

Three properties of the method matter for reading the numbers.

**The workload is frozen.** The 49 cases of `evaluation/dataset.json` at digest
`af873b6c28bd44186dd580369b3120f9b8b4c5d6d1c13ec4a55e0f72d220f0b7`, driven
through `PLANNER_MODE=rule`. Each run executes 1,519 requests: one warm-up pass
over 49 cases followed by 30 measured repeats of the same 49.

**The boundaries are the benchmark's boundaries.** Planning is `route_query`
followed by `dispatch_plan`; total adds `execute_tools`, `retrieve_context` and
`synthesize`. These are the same two boundaries `evaluation.runner` measures.
Percentiles come from `evaluation.metrics.percentile`, the harness's own
linear-interpolation definition, so a tail figure is comparable across runs.

Sharing the boundaries is necessary but not sufficient for the absolute figures to
be comparable with the published ones. This release also added per-stage log
events whose emission happens **inside** those boundaries, and a log call costs
more than a counter increment. An absolute planning mean measured here is
therefore not the same quantity as the 0.2103 ms published in
`evaluation/reports/rule_baseline.json`, even though both are labelled planning
latency. Section 4.2 isolates that difference and reconciles the two figures
rather than leaving the reader to wonder which one is wrong.

**Two things are held constant.** The retrieval provider is an in-process
deterministic double, so no configuration pays for the real retrieval engine and
no index cache warms up mid-run. Logging is configured identically in all three
configurations, at `LOG_LEVEL=INFO` and `LOG_FORMAT=text`, because log emission is
not gated by `METRICS_ENABLED`. The measured delta therefore isolates metric
recording and span creation. That the log volume really is constant is checked
directly: all three configurations emit **6.408 log records per request**
(9,734 records over 1,519 requests).

Timing is `perf_counter`, wall clock, on the host below. The rule planner's work
is sub-millisecond, so this is a high-noise regime and the run-to-run spread is
reported rather than hidden.

| Host fact | Value |
|---|---|
| Workload | 49 frozen cases, rule planner, in-memory SQLite, deterministic retrieval double |
| Requests per run | 1,519 |
| Runs per configuration | 3 |
| Python | 3.11.9 |
| Platform | Windows, local developer machine |

## 3. Results

### 3.1 Per run

Milliseconds. Planning is the planning step; total is the whole pipeline.

| Config | Run | Planning mean | Planning median | Planning p95 | Total mean | Total median | Total p95 |
|---|---|---|---|---|---|---|---|
| off | 1 | 0.3387 | 0.3265 | 0.4440 | 0.7006 | 0.6765 | 1.1751 |
| off | 2 | 0.3601 | 0.3300 | 0.6157 | 0.7524 | 0.7140 | 1.3283 |
| off | 3 | 0.3292 | 0.3200 | 0.4310 | 0.6878 | 0.6790 | 1.1375 |
| metrics | 1 | 0.3420 | 0.3320 | 0.4560 | 0.7266 | 0.7220 | 1.2247 |
| metrics | 2 | 0.3582 | 0.3420 | 0.5113 | 0.7516 | 0.7390 | 1.2770 |
| metrics | 3 | 0.3274 | 0.3240 | 0.3880 | 0.6962 | 0.6710 | 1.1540 |
| metrics+tracing | 1 | 0.4039 | 0.3920 | 0.5195 | 0.9016 | 0.8540 | 1.4851 |
| metrics+tracing | 2 | 0.3892 | 0.3790 | 0.4845 | 0.8696 | 0.8395 | 1.4020 |
| metrics+tracing | 3 | 0.3836 | 0.3800 | 0.4625 | 0.8551 | 0.7895 | 1.4055 |

### 3.2 Run-level mean

The mean of the three run means. Individual per-case samples from different runs
are never merged into one distribution; only the run summaries are averaged, and
that is stated here so the figure is not mistaken for a pooled sample.

| Config | Planning mean | Planning median | Planning p95 | Total mean | Total median | Total p95 |
|---|---|---|---|---|---|---|
| off | 0.3427 | 0.326 | 0.4969 | 0.7136 | 0.690 | 1.2136 |
| metrics | 0.3425 | 0.333 | 0.4518 | 0.7248 | 0.711 | 1.2185 |
| metrics+tracing | 0.3922 | 0.384 | 0.4889 | 0.8754 | 0.828 | 1.4309 |

### 3.3 Delta against `off`

| Config | Planning mean | Planning % | Total mean | Total % |
|---|---|---|---|---|
| metrics | -0.0001 ms | -0.0% | +0.0112 ms | +1.6% |
| metrics+tracing | +0.0496 ms | +14.5% | +0.1618 ms | +22.7% |

### 3.4 The same workload with the per-stage log events silenced

A fourth configuration, ``off`` at `LOG_LEVEL=WARNING`, exists for one question:
how much of the absolute figure is the logging this release added? At `WARNING`
the per-stage events are not emitted and **0.000 log records per request** are
produced, which is the instrumentation level the frozen baseline was measured at.
Everything else is identical.

| Config | Run | Planning mean | Planning median | Planning p95 | Total mean | Total median | Total p95 | Log records / request |
|---|---|---|---|---|---|---|---|---|
| off, WARNING | 1 | 0.2451 | 0.239 | 0.3175 | 0.5264 | 0.521 | 0.6560 | 0.000 |
| off, WARNING | 2 | 0.2479 | 0.242 | 0.3215 | 0.5302 | 0.524 | 0.8140 | 0.000 |
| off, WARNING | 3 | 0.2504 | 0.243 | 0.3340 | 0.5340 | 0.528 | 0.8160 | 0.000 |
| off, INFO (shipped) | run-level mean | 0.3427 | 0.326 | 0.4969 | 0.7136 | 0.690 | 1.2136 | 6.408 |

Run-level mean of the three quiet runs: planning 0.2478 ms, total 0.5302 ms.

| Comparison | Planning mean | Total mean |
|---|---|---|
| Per-stage logging (`off`, INFO minus `off`, WARNING) | **+0.0949 ms** | **+0.1834 ms** |
| Metric recording (`metrics` minus `off`, both INFO) | -0.0001 ms | +0.0112 ms |
| Span creation (`metrics+tracing` minus `metrics`, both INFO) | +0.0497 ms | +0.1506 ms |

## 4. Interpretation

### 4.1 What each component costs

**Metric recording is below the noise floor of this host.** The `off` planning
mean varies by 0.031 ms across the three runs (0.3292 to 0.3601). The measured
cost of enabling metrics is 0.0001 ms, an order of magnitude smaller than that
spread, so this workload cannot resolve it in either direction. Reporting it as
"free" would overstate the evidence; reporting it as a percentage would produce a
number dominated by which run was sampled.

**Span creation is measurable and small.** Enabling tracing adds 0.05 ms to
planning and 0.16 ms to the whole pipeline, and the sign is consistent across all
three runs: every `metrics+tracing` run is slower than every `off` run on both
boundaries. This is the real cost of the observability layer, and it is why
tracing is off by default.

**The absolute figures are what matter.** A 22.7% relative increase reads
alarming until the base is named: the rule planner plans in a third of a
millisecond. Against the stages that dominate a real request, the same 0.16 ms
is negligible. These comparisons are derived from previously published
artifacts, not measured in this run, and are labelled as such.

| Stage | Published latency | 0.1618 ms as a share | Source |
|---|---|---|---|
| LLM planner call | 1551.94 ms mean | 0.010% | `docs/releases/v0.7.md`, provider `openai_compatible` |
| Local RAG retrieval | ~4880 ms | 0.003% | `README.md`, local light retrieval steady state |

**No quantity-of-magnitude regression exists.** The rule that governs here is
that overhead must be explicable and must not change behaviour. Both hold:
metric recording is a dictionary lookup plus an atomic increment on an existing
child, span creation is the only allocation, and neither touches a decision.

The logging this release added is the largest single cost, and that ordering is
worth stating plainly. It is also the cost the shipped default pays, since
`LOG_LEVEL` defaults to `INFO`: the per-stage events are not opt-in. An operator
who wants the latency of the pre-0.9 pipeline back can have it, at the price of
the per-stage operational record, by raising `LOG_LEVEL` to `WARNING`. That is a
real trade and the measurement above is what makes it an informed one rather than
a guess.

### 4.2 Reconciling the absolute figure with the published baseline

Two numbers describing planning latency appear in this repository and they differ.
Neither is wrong, and the difference is fully accounted for.

| Figure | Value | Sample | Source |
|---|---|---|---|
| Published rule-planner planning latency | 0.2103 ms mean (median 0.197, p95 0.2594) | 49, single pass | `evaluation/reports/rule_baseline.json` |
| This harness, `off` at `INFO` | 0.3427 ms run-level mean | 1,470, 30 passes | section 3.2 |
| This harness, `off` at `WARNING` | 0.2478 ms run-level mean | 1,470, 30 passes | section 3.4 |

The measured boundary is genuinely the same in both. `evaluation.runner.run_case`
times `perf_counter()`, calls `route_query`, calls `dispatch_plan`, and stops; the
harness quotes that code path exactly. So the difference is not a differently
drawn boundary, which is the failure mode worth ruling out first.

**The named part is the per-stage logging, +0.0949 ms.** `dispatch_plan` now emits
`planner.started` and `planner.completed` inside the measured region, and the
`agent.request.*` events follow the same pattern. Silencing them at `WARNING`
recovers most of the gap: 0.3427 falls to 0.2478, which is 0.0375 ms above the
published figure.

**The residual 0.0375 ms is not attributed, and is not claimed as observability
cost.** Three candidates remain and this measurement does not separate them. The
published figure is a single 49-sample pass while the harness averages 30 passes,
so it carries more slow outliers and any drift across a longer run. `dispatch_plan`
gained a wrapper and a timing call of its own. And the two runs were made at
different times on a developer machine, with the ambient variation that implies;
the `off` configuration alone varies by 0.031 ms between runs, which is the same
order as the residual.

That last point is the reason the headline claims in this document rest on the
**delta between configurations measured by one harness in one session**, where the
host, the logging level and the code path are held fixed, and none rest on the
cross-harness absolute figure. The reconciliation is reported because a reader who
compares 0.3427 with 0.2103 deserves an answer, not because the comparison
supports a claim.

## 5. Benchmark integrity

The frozen planner benchmark was run before and after the change, in the same
environment, over the same 49 cases.

| Metric | Before | After |
|---|---|---|
| intent_accuracy | 0.975609756097561 | 0.975609756097561 |
| tool_selection_exact_match | 0.7959183673469388 | 0.7959183673469388 |
| tool_precision | 0.88 | 0.88 |
| tool_recall | 0.9850746268656716 | 0.9850746268656716 |
| argument_accuracy | 1.0 | 1.0 |
| invalid_tool_rate | 0.0 | 0.0 |
| unnecessary_tool_call_rate | 0.12 | 0.12 |
| task_success_rate | 0.7959183673469388 | 0.7959183673469388 |
| planner_failure_rate | 0.0 | 0.0 |

All nine are identical. The failure records are byte-identical, over the same ten
cases (`ad-002`, `ad-004`, `ad-006`, `ad-008`, `ro-003`, `mt-007`, `am-004`,
`ood-006`, `ood-007`, `ood-008`), which is a stronger statement than matching
aggregates: not one case changed outcome.

`evaluation/dataset.json` still hashes to
`af873b6c28bd44186dd580369b3120f9b8b4c5d6d1c13ec4a55e0f72d220f0b7`, and
`evaluation/reports/` is untouched. Latency figures from the frozen reports are
not reproduced here, because a planning latency is a wall-clock quantity that
varies between runs and must be read from the artifact of the run it belongs to.

## 6. Reproducing

```
python -m scripts.observability_overhead --config off             --repeats 30
python -m scripts.observability_overhead --config metrics         --repeats 30
python -m scripts.observability_overhead --config "metrics+tracing" --repeats 30

# The logging cost, from section 3.4. Not part of the metric comparison.
python -m scripts.observability_overhead --config off --log-level WARNING --repeats 30
```

Return to the default configuration with `METRICS_ENABLED=true` and
`OTEL_ENABLED=false`, which is what `.env.example` ships.

## 7. Limitations

- One host, one process layout. These are not container measurements; the Docker
  runtime validation in `docs/releases/v0.9.0.md` checks that the metrics are
  produced, not how long they take.
- The workload is the rule planner, which is what makes it deterministic and
  therefore measurable. An LLM-planned request spends its time in a network call
  that is two to three orders of magnitude larger, and the observability share
  there is proportionally smaller.
- The `off` configuration still emits the structured log events added in this
  release, because `METRICS_ENABLED` does not gate logging. What is measured is
  `off` as the configuration actually ships, not a hypothetical build with no
  instrumentation code at all.
- Tail figures at p95 with a sub-millisecond sample are sensitive to a single slow
  outlier; the run-level table is the more reliable summary.
