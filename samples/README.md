# Sample Data

Example benchmark data for testing and demonstrating `asv` commands.

## Directory Structure

```
samples/
    native/                     # ASV native result format (v2)
        numlib/                 # Small project: 5 benchmarks, 1 machine, 2 commits
        dataflow/               # Large project: 11 benchmarks, 2 machines, 4 commits
    external/
        BMF/                    # Bencher Metric Format (for `asv import`)
            webperf/            # Web performance benchmarks, 3 commits
```

## Native Format (`samples/native/`)

Each native project directory contains:

- `asv.conf.json` -- project configuration
- `results/benchmarks.json` -- benchmark metadata
- `results/<machine>/machine.json` -- machine hardware info
- `results/<machine>/<hash>-<env>.json` -- result files (one per commit+environment)

### numlib

A small numeric library project with:

- **Benchmarks**: matrix multiply, eigenvalues, FFT (parameterized), memory, and a tracked metric
- **Machine**: `developer-laptop` (Intel i7, 32G RAM)
- **Commits**: 2 (showing slight performance improvement)

### dataflow

A larger data pipeline project with:

- **Benchmarks**: CSV/Parquet I/O, groupby (multi-parameter), merge (parameterized),
  sort, pivot table, memory usage, throughput tracking, error rate tracking
- **Machines**: `build-server` (AMD EPYC 128-core) and `ci-runner` (ARM Neoverse)
- **Commits**: 4 on build-server, 2 on ci-runner (showing optimization over time)

## BMF Format (`samples/external/BMF/`)

Files in the [Bencher Metric Format](https://bencher.dev/docs/reference/bencher-metric-format/)
for importing into ASV via `asv import --format bencher`.

### webperf

Web performance benchmarks with built-in Bencher measures:

- `latency` (nanoseconds) -- converted to seconds on import
- `throughput` (operations/second)
- `file-size` (bytes)
- `build-time` (seconds)

**Files**: one JSON per commit (`commit_<hash>.json`), each containing 9 benchmarks.

### Importing BMF data

To import a single file:

```bash
asv import samples/external/BMF/webperf/commit_a1f3e2b7.json \
    --format bencher \
    --commit a1f3e2b7 \
    --date 1704067200000 \
    --machine test-server
```

To import all files in a directory (merges all JSON files):

```bash
asv import samples/external/BMF/webperf/ \
    --format bencher \
    --commit a1f3e2b7 \
    --machine test-server
```
