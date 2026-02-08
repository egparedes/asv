# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Import source for the Bencher Metric Format (BMF).

The BMF JSON structure is::

    {
        "<benchmark_name>": {
            "<measure>": {
                "value": <float>,
                "lower_value": <float>,   // optional
                "upper_value": <float>    // optional
            },
            ...
        },
        ...
    }

Built-in Bencher measures and their units:

- ``latency``: nanoseconds
- ``throughput``: operations/second
- ``file-size``: bytes
- ``build-time``: seconds

See https://bencher.dev/docs/reference/bencher-metric-format/
"""

import datetime
import json
import os
import platform
import socket

from asv import import_source, results, util
from asv.console import log


# Mapping of well-known Bencher measure slugs to ASV benchmark metadata.
# (asv_type, asv_unit, value_conversion_factor)
#   conversion factor converts from the Bencher unit to the ASV unit.
_KNOWN_MEASURES = {
    "latency": ("time", "seconds", 1e-9),       # ns -> s
    "throughput": ("track", "ops/s", 1.0),
    "file-size": ("memory", "bytes", 1.0),
    "build-time": ("time", "seconds", 1.0),
}


class BencherImportSource(import_source.ImportSource):
    """Import benchmark results in the Bencher Metric Format (BMF)."""

    name = "bencher"

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--commit",
            default=None,
            help=(
                "Commit hash to associate with the imported results. "
                "If not given, asv attempts to read it from the repo HEAD."
            ),
        )
        parser.add_argument(
            "--date",
            default=None,
            help=(
                "Commit date as an ISO-8601 string or Unix timestamp in "
                "milliseconds.  Defaults to the current time."
            ),
        )
        parser.add_argument(
            "--machine",
            default=None,
            help="Machine name.  Defaults to the current hostname.",
        )
        parser.add_argument(
            "--env-name",
            default=None,
            help="Environment name.  Defaults to 'imported'.",
        )

    @classmethod
    def import_results(cls, path, conf, args):
        # ----------------------------------------------------------
        # Load the BMF JSON
        # ----------------------------------------------------------
        if os.path.isdir(path):
            bmf_data = cls._load_directory(path)
        else:
            bmf_data = cls._load_file(path)

        if not bmf_data:
            return [], {}

        # ----------------------------------------------------------
        # Resolve metadata
        # ----------------------------------------------------------
        commit_hash = getattr(args, "commit", None) or cls._resolve_commit(conf)
        date_ms = cls._resolve_date(getattr(args, "date", None))
        machine = getattr(args, "machine", None) or socket.gethostname()
        env_name = getattr(args, "env_name", None) or "imported"

        params = {
            "machine": machine,
            "arch": platform.machine(),
            "cpu": "",
            "os": f"{platform.system()} ({platform.release()})",
            "ram": "",
        }

        # ----------------------------------------------------------
        # Convert BMF entries to ASV results
        # ----------------------------------------------------------
        res = results.Results(
            params=params,
            requirements={},
            commit_hash=commit_hash,
            date=date_ms,
            python="",
            env_name=env_name,
            env_vars={},
        )

        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        new_benchmarks = {}

        for bench_name, measures in bmf_data.items():
            if not isinstance(measures, dict):
                log.warning(f"Skipping non-object benchmark entry: {bench_name}")
                continue

            for measure_name, metric in measures.items():
                if not isinstance(metric, dict) or "value" not in metric:
                    log.warning(
                        f"Skipping invalid metric for {bench_name}/{measure_name}"
                    )
                    continue

                value = metric["value"]
                lower = metric.get("lower_value")
                upper = metric.get("upper_value")

                asv_type, asv_unit, factor = _KNOWN_MEASURES.get(
                    measure_name, ("track", "unit", 1.0)
                )

                converted_value = value * factor
                converted_lower = lower * factor if lower is not None else None
                converted_upper = upper * factor if upper is not None else None

                # Build the ASV benchmark name
                # If there is only one measure, use the bench name directly;
                # otherwise qualify with the measure.
                if len(measures) == 1:
                    asv_name = bench_name
                else:
                    asv_name = f"{bench_name}.{measure_name}"

                # Store in the result object's internal dicts
                res._results[asv_name] = [converted_value]
                res._benchmark_params[asv_name] = []
                res._benchmark_version[asv_name] = None
                res._started_at[asv_name] = now_ms

                # Stats from lower/upper bounds
                if converted_lower is not None or converted_upper is not None:
                    stats = {}
                    if converted_lower is not None:
                        stats["ci_99_a"] = converted_lower
                        stats["q_25"] = (converted_value + converted_lower) / 2
                    if converted_upper is not None:
                        stats["ci_99_b"] = converted_upper
                        stats["q_75"] = (converted_value + converted_upper) / 2
                    res._stats[asv_name] = [stats]
                else:
                    res._stats[asv_name] = None

                # Benchmark metadata for benchmarks.json
                new_benchmarks[asv_name] = {
                    "name": asv_name,
                    "param_names": [],
                    "params": [],
                    "timeout": 60.0,
                    "type": asv_type,
                    "unit": asv_unit,
                }

        # Ensure machine.json exists
        cls._ensure_machine_json(conf.results_dir, machine, params)

        log.info(
            f"Parsed {len(new_benchmarks)} benchmark(s) for "
            f"commit {commit_hash[:8]}"
        )
        return [res], new_benchmarks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_file(cls, path):
        with open(path) as f:
            return json.load(f)

    @classmethod
    def _load_directory(cls, path):
        """Load and merge all .json files in a directory."""
        merged = {}
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".json"):
                with open(os.path.join(path, fn)) as f:
                    data = json.load(f)
                merged.update(data)
        return merged

    @classmethod
    def _resolve_commit(cls, conf):
        """Try to get HEAD commit hash from the repo."""
        try:
            from asv.repo import get_repo

            repo = get_repo(conf)
            return repo.get_hash_from_name(None)
        except Exception:
            raise util.UserError(
                "Cannot determine commit hash.  Use --commit to specify one."
            )

    @classmethod
    def _resolve_date(cls, date_str):
        """Parse a date argument into millisecond JS timestamp."""
        if date_str is None:
            return int(
                datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
            )

        # Try integer (ms timestamp)
        try:
            return int(date_str)
        except ValueError:
            pass

        # Try ISO-8601
        dt = datetime.datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1000)

    @classmethod
    def _ensure_machine_json(cls, results_dir, machine_name, params):
        """Create the machine.json file if it does not exist."""
        machine_dir = os.path.join(results_dir, machine_name)
        os.makedirs(machine_dir, exist_ok=True)
        machine_json = os.path.join(machine_dir, "machine.json")
        if not os.path.isfile(machine_json):
            from asv.machine import Machine

            data = {
                "machine": machine_name,
                "arch": params.get("arch", ""),
                "cpu": params.get("cpu", ""),
                "os": params.get("os", ""),
                "ram": params.get("ram", ""),
                "num_cpu": "",
                "version": Machine.api_version,
            }
            util.write_json(machine_json, data, Machine.api_version)
