# Real-data integration and the evaluation boundary

This document records what adding the MetroPT-3 real-world device source does
and does not establish. It exists because an integration test that returns the
right answer is easy to mistake for an evaluation result, and the two are not the
same thing.

Nothing here replaces or supersedes the frozen evaluation reports under
`evaluation/reports/`. Those describe the 49-case Agent benchmark and are
unchanged by this work.

## The frozen benchmark did not move

The device integration was required not to disturb the planner, the prompt, the
tool schemas or the dataset. One change was necessary and it is disclosed below;
everything else is additive.

The 49-case dataset is unchanged. Its digest is still
`af873b6c28bd44186dd580369b3120f9b8c5d6d1c13ec4a55e0f72d220f0b7`.

### The one change, and the drift measurement

The rule planner's identifier pattern was `([A-Za-z]{2,}-\d{3,})`. Against
`METRO-APU-001` that pattern cannot start at the first letter, skips forward and
matches `APU-001`, so the device would have been looked up under a truncated
identifier. The pattern was widened to allow one optional middle segment:

```
([A-Za-z]{2,}(?:-[A-Za-z]{2,})?-\d{3,})
```

This is a lexer change, not a planning change: it does not alter which tools are
selected, how they are ordered, or how the answer is synthesised. Its impact was
measured rather than asserted. The rule planner was run over the frozen dataset
immediately before and immediately after the change, in the same environment:

| Metric | Before | After | Drift |
| ------ | ------ | ----- | ----- |
| `intent_accuracy` | 0.9756 | 0.9756 | none |
| `tool_selection_exact_match` | 0.7959 | 0.7959 | none |
| `tool_precision` | 0.8800 | 0.8800 | none |
| `tool_recall` | 0.9851 | 0.9851 | none |
| `argument_accuracy` | 1.0000 | 1.0000 | none |
| `invalid_tool_rate` | 0.0000 | 0.0000 | none |
| `unnecessary_tool_call_rate` | 0.1200 | 0.1200 | none |
| `task_success_rate` | 0.7959 | 0.7959 | none |
| `planner_failure_rate` | 0.0000 | 0.0000 | none |

Nine of nine metrics identical, to the full precision reported. No case changed
its outcome, in either direction. The dataset contains no two-segment identifier,
so the widened branch is never taken on a benchmark case. `tests/test_parser.py`
pins both halves: that the full identifier is recovered, and that every
single-segment form parses as before.

## What the real-data gates do not measure

The gates ask one question: for the latest record of the real dataset, does the
service return the right structured answer? They are integration checks.

They are **not** an accuracy measurement, for three reasons worth stating plainly:

1. **There is one record under test.** The gate reads the last record of the file.
   It does not evaluate the adapter over 1.5 million rows, and it says nothing
   about how the derived status rule behaves across the dataset's full range. A
   single passing record is evidence that the wiring works, not evidence that the
   rule is right in general.
2. **There is no ground truth to compare against.** The dataset publishes sensor
   readings, not equipment states. The status is derived from documented control
   contacts, so the gate can only check that the derivation ran and that it
   declares itself derived. It cannot check that the label is correct, because
   nothing in the dataset says what correct is.
3. **No benchmark case covers this device.** The frozen 49 cases use the seeded
   demo identifiers. Adding `METRO-APU-001` does not add a case, so no reported
   metric changed and none should. Editing the dataset to cover it would
   invalidate the hash lock that makes every published report comparable.

For the same reason no new metric is published here. A metric computed from one
record over a denominator of one would be a number without meaning, and reporting
it beside the frozen benchmark would invite a comparison that is not available.

## What was measured

Recorded from the validation run against the copy described in
`docs/data/metropt3.md`. The row count and digest are properties of the file; the
latency figures are single observations on one developer machine, are not
benchmarks, and will differ on other hardware.

### Service-level gate

| Fact | Value |
| ---- | ----- |
| Verdict | `REAL_DATA_GATE: PASS` |
| File size | 218,300,507 bytes |
| Data rows | 1,516,948 |
| File SHA-256 | `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24` |
| Selected record | `2020-09-01 03:59:50` |
| Cold query, adapter read | 2.562 ms |
| Warm query, cache hit | 0.376 ms |

The cold figure is the first read of a 208 MiB file, and it is milliseconds
because the adapter reads a header and a bounded tail window rather than the
file. The warm figure is the same query served from the stamped cache. Both are
reported so the cost of a query is visible and bounded, not because either is a
target.

Selected readings from that record, reported as the file carries them:

| Reading | Value |
| ------- | ----- |
| `TP2_bar` | -0.014 |
| `TP3_bar` | 8.86 |
| `H1_bar` | 8.848 |
| `DV_pressure_bar` | -0.022 |
| `reservoir_pressure_bar` | 8.864 |
| `oil_temperature_c` | 59.475 |
| `motor_current_a` | 0.0425 |
| Derived status | `stopped` |

Two pressure channels read slightly negative. Those are the file's own values,
reproduced without clamping or sign correction. Smoothing a real sensor artefact
into a plausible-looking zero would make the output easier to read and less
truthful, so it was not done.

The derived `stopped` label is consistent with the record: `COMP` is active,
`DV_eletric` is inactive, and the motor current is 0.0425 A, well inside the
published "off" band.

### Agent gate over HTTP

| Route | Planner | Latency | Verdict |
| ----- | ------- | ------- | ------- |
| Host service | `rule` (default) | 17.588 ms | `REAL_DATA_AGENT_E2E_GATE: PASS` |
| Host service | `llm` (local override) | 2542.293 ms | passed the same checks |
| Container, release image | `rule` | 16.016 ms | `REAL_DATA_DOCKER_GATE: PASS` |

The LLM run is recorded because it is informative, not because it is comparable:
it is one query, it used the developer's own provider, and the device
integration does not depend on which planner runs. Its latency is dominated by
the provider call, which is the expected relationship and the reason the default
is `rule`.

Both planners produced the same plan for the new identifier: `get_device_status`
and nothing else. The answer attributed its evidence to `uci_metropt3` rather
than `sqlite:devices`, carried the upstream timestamp, showed the readings under
their own names, and reported `报警码：无` because the dataset carries no alarm
code. It did not populate the demo record's temperature, pressure or rpm fields.

### Container facts

| Fact | Value |
| ---- | ----- |
| Image | `industrial-maintenance-agent:0.8.5`, id `9265cdc33d0c` |
| Size | 383 MB on disk, 83,646,066 bytes compressed |
| Dataset inside image | Absent |
| `numpy` / `scikit-learn` / `pypdf` | Absent, as intended for the standalone shape |
| Mounted CSV digest inside container | `db30ccb4…793e24`, identical to the host |

The last row is the point of the container gate. The service in the container
answered from bytes whose digest matches the host file exactly, and the image it
ran from contains no CSV at all. That combination is what shows the data path is
the read-only mount and nothing else.

### A defect this validation caught

The first build of this release produced a 1.37 GB image. The Dockerfile copied
`data` as a directory, so placing the dataset in `data/external/metropt3/` for
validation put 218 MB of CSV and archive into the build context, and the
`chown -R app:app /app` that follows duplicated it into a second layer. No
instruction named the dataset, so nothing in the build output pointed at the
cause; the only symptom was the size.

Both halves are fixed: `.dockerignore` excludes `data/external`, and the
Dockerfile copies `data/devices.json` and `data/alarms.json` by name instead of
taking the directory. `tests/test_docker_config.py` pins the exclusion, the
named copies, and the absence of a directory-wide data copy, so the regression
cannot return silently. The sizes in the table above are from the corrected
build and match the 383 MB the standalone shape carried before this release.

## Regression coverage added

`tests/test_device_data.py` covers parsing, status derivation across all four
branches, cache invalidation, every error path, the tool dispatch, and that the
demo SQLite devices are untouched. It runs against a synthetic five-record
fixture, so no test requires the 208 MiB download.

`tests/test_docker_config.py` gained the real-data service contract: the profile
placement, the read-only mount, the absence of a hardcoded host path, the reuse
of the standalone image, the separate database volume, and the two halves of the
dataset-exclusion fix.

`tests/test_parser.py` gained the two-segment identifier cases described above.

## Reproducing

```bash
python -m scripts.prepare_metropt3 --input "/path/MetroPT3(AirCompressor).csv"
python -m scripts.real_data_gate --csv "/path/MetroPT3(AirCompressor).csv" \
  --expect-rows 1516948 --expect-sha256 db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24
python -m scripts.real_data_agent_gate --base-url http://127.0.0.1:8000
```

The first script validates a local copy and downloads nothing. The second and
third print a JSON report and a verdict line.

To confirm the frozen benchmark is untouched, run the rule planner before and
after any change you make to the identifier handling and compare the reports:

```bash
python -m evaluation.runner --planner rule --output-dir /tmp/before --tag before
# make the change
python -m evaluation.runner --planner rule --output-dir /tmp/after --tag after
```

Write these to a directory outside `evaluation/reports/`. The reports there are
published artefacts of specific past runs, and regenerating them in place would
rewrite history that other documents link to.
