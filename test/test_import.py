# Licensed under a 3-clause BSD style license - see LICENSE.rst

import json
import os
from os.path import join

import pytest

from asv import config, results, util
from asv.benchmarks import Benchmarks
from asv.commands.import_results import Import
from asv.plugins.bencher import BencherImportSource

from . import tools

DEFAULT_BRANCH = f"{util.git_default_branch()}"


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def import_conf(tmpdir):
    """Minimal config pointing at a temporary results dir."""
    tmpdir = str(tmpdir)
    dvcs = tools.generate_test_repo(tmpdir, values=[0], dvcs_type="git")
    conf = config.Config.from_json({
        "repo": dvcs.path,
        "results_dir": join(tmpdir, "results"),
        "project": "test-project",
    })
    return conf, dvcs


@pytest.fixture
def bmf_file(tmpdir):
    """Write a sample BMF JSON file and return its path."""
    data = {
        "bench_algo_sort": {
            "latency": {
                "value": 88.0,
                "lower_value": 85.0,
                "upper_value": 91.0,
            }
        },
        "bench_algo_search": {
            "latency": {
                "value": 200.5,
            },
            "throughput": {
                "value": 5000000.0,
                "lower_value": 4800000.0,
                "upper_value": 5200000.0,
            },
        },
        "bench_memory_usage": {
            "file-size": {
                "value": 1048576.0,
            }
        },
        "bench_build": {
            "build-time": {
                "value": 12.5,
                "lower_value": 11.0,
                "upper_value": 14.0,
            }
        },
    }
    path = join(str(tmpdir), "bencher_results.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def bmf_dir(tmpdir):
    """Write multiple BMF JSON files in a directory."""
    d = join(str(tmpdir), "bmf_dir")
    os.makedirs(d)

    file1 = {
        "bench_a": {
            "latency": {"value": 100.0}
        }
    }
    file2 = {
        "bench_b": {
            "latency": {"value": 200.0}
        }
    }
    with open(join(d, "results1.json"), "w") as f:
        json.dump(file1, f)
    with open(join(d, "results2.json"), "w") as f:
        json.dump(file2, f)
    return d


# ---------------------------------------------------------------
# Tests for BencherImportSource
# ---------------------------------------------------------------

class TestBencherImportSource:

    def test_single_measure_benchmark(self, import_conf, bmf_file):
        """Benchmarks with a single measure use the benchmark name directly."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine", env_name="imported")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_file, conf, args
        )

        assert len(result_list) == 1
        res = result_list[0]
        assert res.commit_hash == commit

        # bench_algo_sort has one measure -> name is "bench_algo_sort"
        assert "bench_algo_sort" in benchmarks
        assert benchmarks["bench_algo_sort"]["type"] == "time"
        assert benchmarks["bench_algo_sort"]["unit"] == "seconds"
        # 88 ns -> 88e-9 seconds
        assert res._results["bench_algo_sort"] == [pytest.approx(88e-9)]

    def test_multi_measure_benchmark(self, import_conf, bmf_file):
        """Benchmarks with multiple measures get qualified names."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_file, conf, args
        )

        # bench_algo_search has two measures -> qualified names
        assert "bench_algo_search.latency" in benchmarks
        assert "bench_algo_search.throughput" in benchmarks
        assert benchmarks["bench_algo_search.latency"]["type"] == "time"
        assert benchmarks["bench_algo_search.throughput"]["type"] == "track"
        assert benchmarks["bench_algo_search.throughput"]["unit"] == "ops/s"

        res = result_list[0]
        assert res._results["bench_algo_search.latency"] == [pytest.approx(200.5e-9)]
        assert res._results["bench_algo_search.throughput"] == [pytest.approx(5000000.0)]

    def test_file_size_measure(self, import_conf, bmf_file):
        """file-size measure maps to memory type in bytes."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_file, conf, args
        )

        assert "bench_memory_usage" in benchmarks
        assert benchmarks["bench_memory_usage"]["type"] == "memory"
        assert benchmarks["bench_memory_usage"]["unit"] == "bytes"
        assert result_list[0]._results["bench_memory_usage"] == [pytest.approx(1048576.0)]

    def test_build_time_measure(self, import_conf, bmf_file):
        """build-time measure maps to time type in seconds."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_file, conf, args
        )

        assert "bench_build" in benchmarks
        assert benchmarks["bench_build"]["type"] == "time"
        assert benchmarks["bench_build"]["unit"] == "seconds"
        assert result_list[0]._results["bench_build"] == [pytest.approx(12.5)]

    def test_stats_from_bounds(self, import_conf, bmf_file):
        """lower_value/upper_value are stored as CI stats."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_file, conf, args
        )

        res = result_list[0]
        # bench_algo_sort: 88ns, lower=85ns, upper=91ns
        stats = res._stats["bench_algo_sort"]
        assert stats is not None
        assert stats[0]["ci_99_a"] == pytest.approx(85e-9)
        assert stats[0]["ci_99_b"] == pytest.approx(91e-9)

        # bench_memory_usage: no bounds -> stats is None
        assert res._stats["bench_memory_usage"] is None

    def test_directory_import(self, import_conf, bmf_dir):
        """Importing from a directory merges all .json files."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            bmf_dir, conf, args
        )

        assert "bench_a" in benchmarks
        assert "bench_b" in benchmarks
        res = result_list[0]
        assert res._results["bench_a"] == [pytest.approx(100e-9)]
        assert res._results["bench_b"] == [pytest.approx(200e-9)]

    def test_custom_measure(self, import_conf, tmpdir):
        """Unknown measures default to 'track' type with unit='unit'."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        data = {
            "my_bench": {
                "custom_metric": {"value": 42.0}
            }
        }
        path = join(str(tmpdir), "custom.json")
        with open(path, "w") as f:
            json.dump(data, f)

        args = _make_args(commit=commit, machine="test-machine")
        result_list, benchmarks = BencherImportSource.import_results(
            path, conf, args
        )

        assert "my_bench" in benchmarks
        assert benchmarks["my_bench"]["type"] == "track"
        assert benchmarks["my_bench"]["unit"] == "unit"
        assert result_list[0]._results["my_bench"] == [pytest.approx(42.0)]

    def test_date_iso_format(self, import_conf, bmf_file):
        """ISO-8601 date strings are parsed correctly."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(
            commit=commit, machine="test-machine",
            date="2024-06-15T12:00:00+00:00",
        )
        result_list, _ = BencherImportSource.import_results(bmf_file, conf, args)
        # 2024-06-15T12:00:00 UTC -> 1718452800000 ms
        assert result_list[0].date == 1718452800000

    def test_date_timestamp(self, import_conf, bmf_file):
        """Integer timestamp (ms) passed via --date."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(commit=commit, machine="test-machine", date="1700000000000")
        result_list, _ = BencherImportSource.import_results(bmf_file, conf, args)
        assert result_list[0].date == 1700000000000


# ---------------------------------------------------------------
# Tests for Import command (end-to-end)
# ---------------------------------------------------------------

class TestImportCommand:

    def test_import_creates_result_files(self, import_conf, bmf_file):
        """Full end-to-end: import writes result files and benchmarks.json."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        args = _make_args(
            commit=commit, machine="test-machine",
            env_name="imported", format="bencher",
            config=None, verbose=False,
        )
        Import.run(conf=conf, path=bmf_file, fmt="bencher", args=args)

        # Results directory exists
        assert os.path.isdir(conf.results_dir)

        # benchmarks.json was created
        bench_path = Benchmarks.get_benchmark_file_path(conf.results_dir)
        assert os.path.isfile(bench_path)
        bench_data = util.load_json(bench_path, api_version=Benchmarks.api_version)
        assert "bench_algo_sort" in bench_data

        # machine.json was created
        machine_json = join(conf.results_dir, "test-machine", "machine.json")
        assert os.path.isfile(machine_json)

        # Result file was created
        short_hash = commit[:8]
        result_file = join(
            conf.results_dir, "test-machine",
            f"{short_hash}-imported.json"
        )
        assert os.path.isfile(result_file)

        # The result can be loaded back
        loaded = results.Results.load(result_file, machine_name="test-machine")
        assert loaded.commit_hash == commit
        assert "bench_algo_sort" in loaded.get_all_result_keys()

    def test_import_unknown_format(self, import_conf, bmf_file):
        """Raise UserError for unknown format."""
        conf, _ = import_conf
        args = _make_args(format="nonexistent")
        with pytest.raises(util.UserError, match="Unknown import format"):
            Import.run(conf=conf, path=bmf_file, fmt="nonexistent", args=args)

    def test_import_nonexistent_path(self, import_conf):
        """Raise UserError for missing path."""
        conf, _ = import_conf
        args = _make_args(format="bencher")
        with pytest.raises(util.UserError, match="Path does not exist"):
            Import.run(conf=conf, path="/nonexistent/path", fmt="bencher", args=args)

    def test_import_merge_benchmarks(self, import_conf, tmpdir):
        """Importing twice merges benchmarks.json entries."""
        conf, dvcs = import_conf
        commit = dvcs.get_hash(DEFAULT_BRANCH)

        # First import
        data1 = {"bench_one": {"latency": {"value": 100.0}}}
        path1 = join(str(tmpdir), "bmf1.json")
        with open(path1, "w") as f:
            json.dump(data1, f)

        args = _make_args(commit=commit, machine="test-machine", env_name="env1")
        Import.run(conf=conf, path=path1, fmt="bencher", args=args)

        # Second import
        data2 = {"bench_two": {"throughput": {"value": 5000.0}}}
        path2 = join(str(tmpdir), "bmf2.json")
        with open(path2, "w") as f:
            json.dump(data2, f)

        args = _make_args(commit=commit, machine="test-machine", env_name="env2")
        Import.run(conf=conf, path=path2, fmt="bencher", args=args)

        # Both benchmarks present
        bench_path = Benchmarks.get_benchmark_file_path(conf.results_dir)
        bench_data = util.load_json(bench_path, api_version=Benchmarks.api_version)
        assert "bench_one" in bench_data
        assert "bench_two" in bench_data


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

class _Namespace:
    """Minimal argparse.Namespace stand-in."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_args(**kwargs):
    defaults = {
        "commit": None,
        "date": None,
        "machine": None,
        "env_name": None,
        "format": None,
    }
    defaults.update(kwargs)
    return _Namespace(**defaults)
