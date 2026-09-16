# MetroPT-3 real-world sensor data

This document covers the optional external device data source that backs the
`METRO-APU-001` device. It is an integration with a public real-world industrial
sensor dataset. It is not data from any private installation.

## Dataset

| Field | Value |
| ----- | ----- |
| Name | MetroPT-3 |
| Publisher | UCI Machine Learning Repository |
| Dataset id | 791 |
| DOI | [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Landing page | https://archive.ics.uci.edu/dataset/791/metropt+3+dataset |
| Contents | `MetroPT3(AirCompressor).csv` and `Data Description_Metro.pdf` |

Source: UCI Machine Learning Repository, dataset 791, "MetroPT-3"
(https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).

### What the data is

Readings from an Air Production Unit (APU) compressor on a metro train in an
operational context, logged at 1 Hz between February and August 2020. The
recorded signals are pressures, motor current, oil temperature and the
electrical signals of the air intake valves. The dataset publisher describes it
as multivariate time series from analogue and digital sensors installed on the
compressor.

### Creators

As stated on the UCI dataset page and in the accompanying data description:

- Narjes Davari, INESC TEC, Laboratory of Artificial Intelligence and Decision
  Support
- Bruno Veloso, INESC TEC and Faculty of Economics, University of Porto
- Rita P. Ribeiro, INESC TEC and Faculty of Sciences, University of Porto
- Joao Gama, INESC TEC and Faculty of Economics, University of Porto

### Accompanying publications

- Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., Gama, J. Predictive
  maintenance based on anomaly detection using deep learning for air production
  unit in the railway industry. 2021 IEEE 8th International Conference on Data
  Science and Advanced Analytics (DSAA), pp. 1-10.
  DOI: 10.1109/DSAA53316.2021.9564181
- Veloso, B., Ribeiro, R. P., Pereira, P. M., Gama, J. The MetroPT dataset for
  predictive maintenance. Scientific Data 9, no. 1 (2022): 764.
  DOI: 10.1038/s41597-022-01877-3

## This repository does not redistribute the data

The extracted CSV is about 208 MiB and the archive about 208 MiB. Neither is
covered by this project's MIT license, and neither is committed here:

- the archive stays outside version control;
- every payload under `data/external/metropt3/` is ignored by `.gitignore`;
- the running service never downloads anything. It reads only the path given in
  `METROPT3_CSV_PATH`.

You obtain the data yourself, from the publisher, under the terms of CC BY 4.0.
If you redistribute it or publish results derived from it, comply with the
attribution requirement of that license and cite the dataset and its creators
as listed above.

## Preparing a local copy

```bash
# Download the archive from UCI (about 208 MiB) and extract the CSV.
# Then point the service at it and run the verification script:
export METROPT3_CSV_PATH=/absolute/path/to/MetroPT3(AirCompressor).csv
python -m scripts.prepare_metropt3 --input "$METROPT3_CSV_PATH"
```

The script is optional. Its purpose is to check that a file really is the
MetroPT-3 CSV before you rely on it, and to print row count, timestamp range and
provenance so a mistake is caught at preparation time rather than mid-query. It
performs no download. `--download-info` prints the official location and stops;
fetching the archive stays a deliberate manual step.

## What is published versus what is derived

The dataset is unlabeled with respect to equipment state. It publishes sensor
readings and a separate table of company failure reports, and it does not carry
a per-record `running` / `warning` / `failed` label.

The adapter therefore reports `status` as an **operational state derived** from
the documented control contacts, and it carries the basis in
`status_derivation` on every response so a derived label can never be mistaken
for a published one:

- `DV_eletric` is documented as active while the compressor runs under load;
- `COMP` is documented as active while there is no air intake, meaning the
  compressor is off or operating offloaded;
- `Motor_current` is documented with bands of about 0 A when off, 4 A when
  offloaded, 7 A under load and 9 A while starting.

The resulting labels are `running_under_load`, `running_offloaded`, `stopped`
and, when the contacts are in a state the documentation does not describe,
`observed`. Across bounded windows of the real file the two contacts were
mutually exclusive and agreed with the published current bands, which is what
makes the mapping defensible.

No fault, degradation, anomaly or remaining-useful-life claim is made. Those
would require the failure table and a model, and neither is part of this
integration. Fields the dataset does not carry, such as rotational speed or an
alarm code, are reported as `null` and never synthesised.

## Measured facts of the copy used for validation

Recorded from the file that the integration gate was run against, for
reproducibility:

| Fact | Value |
| ---- | ----- |
| Archive size | 218,381,995 bytes |
| Archive SHA-256 | `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a` |
| Extracted CSV size | 218,300,507 bytes |
| Extracted CSV SHA-256 | `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24` |
| Data rows | 1,516,948 (plus one header line) |
| Columns | 17 (an unnamed index column, `timestamp`, then 15 sensor columns) |
| First timestamp | 2020-02-01 00:00:00 |
| Last timestamp | 2020-09-01 03:59:50 |

The CSV digest is the value the server-side gate checks when it is passed
`--expect-sha256`, so a validation run can state which revision of the file it
read rather than only that some file was present. The digest is of the CSV, not
of the archive: extracting the same archive again reproduces it.

The UCI data description states "Number of Instances: 15169480". The CSV as
shipped holds 1,516,948 data rows, with timestamps about ten seconds apart, and
its unnamed index column runs to 15,169,470. The row count above is what the
file actually contains and is the figure this integration relies on; the
publisher's stated instance count is recorded here so the difference is visible
rather than silently reconciled.

## Field mapping

Analogue readings are exposed under unit-bearing keys. Units are those the
publisher states.

| Output key | Source column | Unit | Publisher description |
| ---------- | ------------- | ---- | --------------------- |
| `TP2_bar` | `TP2` | bar | Pressure on the compressor |
| `TP3_bar` | `TP3` | bar | Pressure generated at the pneumatic panel |
| `H1_bar` | `H1` | bar | Pressure from the pressure drop when the cyclonic separator filter discharges |
| `DV_pressure_bar` | `DV_pressure` | bar | Pressure drop when the towers discharge air dryers; zero indicates operation under load |
| `reservoir_pressure_bar` | `Reservoirs` | bar | Downstream pressure of the reservoirs |
| `oil_temperature_c` | `Oil_temperature` | degrees Celsius | Oil temperature on the compressor |
| `motor_current_a` | `Motor_current` | ampere | Current of one phase of the three-phase motor |

Digital contact signals keep their published column names, since renaming them
would invent a vocabulary the dataset does not use: `COMP`, `DV_eletric`,
`Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses`.

Note that the published column is spelled `DV_eletric`. That spelling is kept
verbatim so it matches the file; it is the dataset's spelling, not a typo
introduced here.
