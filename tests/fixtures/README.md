# Test fixtures

Synthetic inputs used by the test suite. Nothing here is real data.

## `metropt3_sample.csv`

**TEST FIXTURE.** A five-record stand-in for the UCI MetroPT-3 CSV. It exists so
the adapter's header validation, parsing and last-record selection can be tested
without the 208 MiB download, and so `pytest` never depends on a file that is not
in the repository.

It is not a sample of the real dataset. The values are round numbers chosen to
hit each branch of the documented status rule:

| Row | `COMP` | `DV_eletric` | `Motor_current` | Derived status |
| --- | ------ | ------------ | --------------- | -------------- |
| 1 | 1.0 | 0.0 | 0.0 | `stopped` |
| 2 | 1.0 | 0.0 | 4.0 | `running_offloaded` |
| 3 | 0.0 | 0.0 | 0.0 | `observed` |
| 4 | 1.0 | 0.0 | 4.0 | `running_offloaded` |
| 5 | 0.0 | 1.0 | 7.0 | `running_under_load` |

The column layout matches the published file, including the unnamed leading
index column and the dataset's `DV_eletric` spelling, because the adapter
validates the real header.

The real file is never committed. See `docs/data/metropt3.md`. The gate that
exercises the real CSV is run separately and is reported in
`docs/evaluation/real_data_integration.md`.
