# External datasets

Payloads in this directory are **not** part of the repository and are ignored by
`.gitignore`. Only this README is tracked.

## metropt3/

Holds a local copy of the UCI MetroPT-3 dataset when the optional external
device source (`METRO-APU-001`) is in use.

Expected files after you download and extract the archive from UCI:

- `MetroPT3(AirCompressor).csv` (about 208 MiB, the file the service reads)
- `Data Description_Metro.pdf` (publisher's data description)
- the archive you downloaded

None of them are committed. The CSV is about 208 MiB and is not covered by this
project's MIT license, so it stays out of Git history entirely.

Set `METROPT3_CSV_PATH` to the absolute path of the extracted CSV. The running
service never downloads anything and reads no other file here.

See [`docs/data/metropt3.md`](../../../docs/data/metropt3.md) for the source,
DOI, license, creator attribution and field mapping.
