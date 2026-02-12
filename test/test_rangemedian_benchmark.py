# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Benchmark and correctness tests comparing four RangeMedian implementations:
  1. C++ extension  (_rangemedian)
  2. Numba JIT      (_rangemedian_numba)
  3. NumPy-only     (_rangemedian_numpy)
  4. Pure Python    (_rangemedian_python)

Run benchmarks:
    pytest test/test_rangemedian_benchmark.py -v --benchmark-enable

Run correctness tests only:
    pytest test/test_rangemedian_benchmark.py -v -k TestCorrectness
"""

import random

import numpy as np
import pytest

_rangemedian = pytest.importorskip(
    "asv._rangemedian", reason="C++ _rangemedian extension not built"
)

from asv._rangemedian_numba import RangeMedian as NumbaRangeMedian
from asv._rangemedian_numba import warmup as numba_warmup
from asv._rangemedian_numpy import RangeMedian as NumpyRangeMedian
from asv._rangemedian_python import RangeMedian as PythonRangeMedian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMPLEMENTATIONS = {
    "cpp": _rangemedian.RangeMedian,
    "numba": NumbaRangeMedian,
    "numpy": NumpyRangeMedian,
    "python": PythonRangeMedian,
}
"""All available implementations keyed by short name."""

INPUT_SIZES = [10, 50, 100, 500, 1000]
BENCHMARK_SIZES_APPROX = [100, 500, 1000]
# For the pure-python / numpy find_best_partition benchmarks we keep sizes
# small to avoid extremely long runs.
BENCHMARK_SIZES_SLOW = [10, 50, 100]

# Implementations that are fast enough for large benchmark sizes.
FAST_IMPLS = {"cpp": _rangemedian.RangeMedian, "numba": NumbaRangeMedian}
# Implementations that need smaller sizes due to interpreter overhead.
SLOW_IMPLS = {"numpy": NumpyRangeMedian, "python": PythonRangeMedian}
ALL_IMPLS = {**FAST_IMPLS, **SLOW_IMPLS}


@pytest.fixture(scope="session", autouse=True)
def _warmup_numba():
    """Ensure Numba JIT compilation is done before benchmarks."""
    numba_warmup()


def _make_data(n, seed=42):
    """Generate reproducible test data of size n."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).tolist()
    w = np.abs(rng.standard_normal(n) + 0.1).tolist()
    w = [max(x, 0.01) for x in w]
    return y, w


def _make_step_data(n, seed=42):
    """Generate step-function data with noise (typical use case)."""
    rng = np.random.default_rng(seed)
    n_steps = max(1, n // 20)
    y0 = np.zeros(n, dtype=np.float64)
    for i in range(1, n_steps):
        pos = int(n * i / n_steps)
        y0[pos:] += rng.standard_normal() * 0.5
    y = (y0 + 0.05 * rng.standard_normal(n)).tolist()
    w = [1.0] * n
    return y, w


# ---------------------------------------------------------------------------
# Correctness tests — every implementation must match C++ exactly
# ---------------------------------------------------------------------------

# Parameterize over all *alternative* implementations (not C++ itself)
ALT_IMPL_IDS = ["numba", "numpy", "python"]
ALT_IMPLS = [IMPLEMENTATIONS[k] for k in ALT_IMPL_IDS]


class TestCorrectness:
    """Verify that every implementation matches the C++ reference."""

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_mu_dist_match(self, ImplCls, n):
        """mu() and dist() produce identical results for random ranges."""
        y, w = _make_data(n)
        ref = _rangemedian.RangeMedian(y, w)
        alt = ImplCls(y, w)

        rng = random.Random(123)
        for _ in range(min(500, n * n)):
            l = rng.randint(0, n - 1)
            r = rng.randint(l, n - 1)
            assert ref.mu(l, r) == pytest.approx(alt.mu(l, r), abs=1e-12), (
                f"mu mismatch at ({l}, {r})"
            )
            assert ref.dist(l, r) == pytest.approx(alt.dist(l, r), abs=1e-12), (
                f"dist mismatch at ({l}, {r})"
            )

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_find_best_partition_match(self, ImplCls, n):
        """find_best_partition() produces identical partitions."""
        y, w = _make_data(n)
        for gamma in [0.1, 0.5, 2.0]:
            for min_size, max_size in [
                (1, min(20, n)),
                (2, min(10, n)),
                (5, min(50, n)),
            ]:
                if min_size > max_size or min_size > n:
                    continue
                ref = _rangemedian.RangeMedian(y, w)
                alt = ImplCls(y, w)
                ref_p = ref.find_best_partition(gamma, min_size, max_size, 0, n)
                alt_p = alt.find_best_partition(gamma, min_size, max_size, 0, n)
                assert ref_p == alt_p, (
                    f"partition mismatch: n={n}, gamma={gamma}, "
                    f"min_size={min_size}, max_size={max_size}"
                )

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    def test_edge_single_element(self, ImplCls):
        ref = _rangemedian.RangeMedian([42.0], [1.0])
        alt = ImplCls([42.0], [1.0])
        assert ref.mu(0, 0) == alt.mu(0, 0)
        assert ref.dist(0, 0) == alt.dist(0, 0)

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    def test_edge_uniform_step_data(self, ImplCls):
        y = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        w = [1.0] * 9
        ref = _rangemedian.RangeMedian(y, w)
        alt = ImplCls(y, w)
        for l in range(9):
            for r in range(l, 9):
                assert ref.mu(l, r) == pytest.approx(alt.mu(l, r), abs=1e-12)
                assert ref.dist(l, r) == pytest.approx(alt.dist(l, r), abs=1e-12)

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    def test_edge_even_count_midpoint(self, ImplCls):
        y = [1.0, 2.0, 3.0, 4.0]
        w = [1.0, 1.0, 1.0, 1.0]
        ref = _rangemedian.RangeMedian(y, w)
        alt = ImplCls(y, w)
        assert ref.mu(0, 3) == pytest.approx(alt.mu(0, 3), abs=1e-12)
        assert ref.dist(0, 3) == pytest.approx(alt.dist(0, 3), abs=1e-12)

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    def test_edge_varied_weights(self, ImplCls):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        w = [0.1, 0.1, 0.1, 0.1, 10.0]
        ref = _rangemedian.RangeMedian(y, w)
        alt = ImplCls(y, w)
        for l in range(5):
            for r in range(l, 5):
                assert ref.mu(l, r) == pytest.approx(alt.mu(l, r), abs=1e-12)
                assert ref.dist(l, r) == pytest.approx(alt.dist(l, r), abs=1e-12)

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    @pytest.mark.parametrize("n", [20, 100])
    def test_solve_potts_integration(self, ImplCls, n):
        from asv.step_detect import solve_potts

        y, w = _make_step_data(n)
        ref_md = _rangemedian.RangeMedian(y, w)
        alt_md = ImplCls(y, w)

        right_ref, values_ref, dists_ref = solve_potts(
            y, w=w, gamma=0.1, mu_dist=ref_md
        )
        right_alt, values_alt, dists_alt = solve_potts(
            y, w=w, gamma=0.1, mu_dist=alt_md
        )

        assert right_ref == right_alt
        for vr, va in zip(values_ref, values_alt):
            assert vr == pytest.approx(va, abs=1e-12)
        for dr, da in zip(dists_ref, dists_alt):
            assert dr == pytest.approx(da, abs=1e-12)

    @pytest.mark.parametrize("ImplCls", ALT_IMPLS, ids=ALT_IMPL_IDS)
    @pytest.mark.parametrize("n", [100, 500])
    def test_solve_potts_approx_integration(self, ImplCls, n):
        from asv.step_detect import solve_potts_approx

        y, w = _make_step_data(n)
        ref_md = _rangemedian.RangeMedian(y, w)
        alt_md = ImplCls(y, w)

        right_ref, values_ref, dists_ref = solve_potts_approx(
            y, w=w, gamma=0.5, mu_dist=ref_md
        )
        right_alt, values_alt, dists_alt = solve_potts_approx(
            y, w=w, gamma=0.5, mu_dist=alt_md
        )

        assert right_ref == right_alt
        for vr, va in zip(values_ref, values_alt):
            assert vr == pytest.approx(va, abs=1e-12)
        for dr, da in zip(dists_ref, dists_alt):
            assert dr == pytest.approx(da, abs=1e-12)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _make_ranges(n, count=200, seed=999):
    """Pre-generate random (left, right) pairs for mu/dist benchmarks."""
    rng = random.Random(seed)
    ranges = []
    for _ in range(count):
        l = rng.randint(0, n - 1)
        r = rng.randint(l, n - 1)
        ranges.append((l, r))
    return ranges


def _bench_mu_dist(benchmark, ImplCls, impl_name, n):
    """Run mu/dist benchmark for a given implementation and size."""
    y, w = _make_data(n)
    rm = ImplCls(y, w)
    ranges = _make_ranges(n)

    def run():
        for l, r in ranges:
            rm.mu(l, r)
            rm.dist(l, r)

    benchmark.extra_info.update(impl=impl_name, n=n)
    benchmark(run)


def _bench_find_best_partition(benchmark, ImplCls, impl_name, n):
    """Run find_best_partition benchmark for a given implementation and size."""
    y, w = _make_step_data(n)

    def run():
        rm = ImplCls(y, w)
        rm.find_best_partition(0.5, 1, min(20, n), 0, n)

    benchmark.extra_info.update(impl=impl_name, n=n)
    benchmark(run)


def _bench_solve_potts(benchmark, ImplCls, impl_name, n):
    """Run solve_potts benchmark for a given implementation and size."""
    from asv.step_detect import solve_potts

    y, w = _make_step_data(n)

    def run():
        rm = ImplCls(y, w)
        solve_potts(y, w=w, gamma=0.5, mu_dist=rm)

    benchmark.extra_info.update(impl=impl_name, n=n)
    benchmark(run)


def _bench_solve_potts_approx(benchmark, ImplCls, impl_name, n):
    """Run solve_potts_approx benchmark for a given implementation and size."""
    from asv.step_detect import solve_potts_approx

    y, w = _make_step_data(n)

    def run():
        rm = ImplCls(y, w)
        solve_potts_approx(y, w=w, gamma=0.5, mu_dist=rm)

    benchmark.extra_info.update(impl=impl_name, n=n)
    benchmark(run)


# ---------------------------------------------------------------------------
# Benchmark: mu / dist
# ---------------------------------------------------------------------------


class TestBenchmarkMuDist:
    """Benchmark individual mu() + dist() calls (200 random ranges)."""

    @pytest.mark.benchmark(group="mu_dist")
    @pytest.mark.parametrize("impl_name,ImplCls", ALL_IMPLS.items(), ids=ALL_IMPLS.keys())
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_mu_dist(self, benchmark, impl_name, ImplCls, n):
        _bench_mu_dist(benchmark, ImplCls, impl_name, n)


# ---------------------------------------------------------------------------
# Benchmark: find_best_partition (core DP, max_size=20)
# ---------------------------------------------------------------------------


class TestBenchmarkFindBestPartition:
    """Benchmark find_best_partition with max_size=20 (typical workload)."""

    @pytest.mark.benchmark(group="find_best_partition")
    @pytest.mark.parametrize("impl_name,ImplCls", FAST_IMPLS.items(), ids=FAST_IMPLS.keys())
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_fast(self, benchmark, impl_name, ImplCls, n):
        _bench_find_best_partition(benchmark, ImplCls, impl_name, n)

    @pytest.mark.benchmark(group="find_best_partition")
    @pytest.mark.parametrize("impl_name,ImplCls", SLOW_IMPLS.items(), ids=SLOW_IMPLS.keys())
    @pytest.mark.parametrize("n", BENCHMARK_SIZES_SLOW)
    def test_slow(self, benchmark, impl_name, ImplCls, n):
        _bench_find_best_partition(benchmark, ImplCls, impl_name, n)


# ---------------------------------------------------------------------------
# Benchmark: solve_potts (exact solver, max_size=n)
# ---------------------------------------------------------------------------


class TestBenchmarkSolvePotts:
    """Benchmark solve_potts (exact solver, max_size=n)."""

    @pytest.mark.benchmark(group="solve_potts")
    @pytest.mark.parametrize("impl_name,ImplCls", ALL_IMPLS.items(), ids=ALL_IMPLS.keys())
    @pytest.mark.parametrize("n", [50, 100])
    def test_solve_potts(self, benchmark, impl_name, ImplCls, n):
        _bench_solve_potts(benchmark, ImplCls, impl_name, n)


# ---------------------------------------------------------------------------
# Benchmark: solve_potts_approx (typical workload, max_size bounded)
# ---------------------------------------------------------------------------


class TestBenchmarkSolvePottsApprox:
    """Benchmark solve_potts_approx (typical workload, max_size bounded)."""

    @pytest.mark.benchmark(group="solve_potts_approx")
    @pytest.mark.parametrize("impl_name,ImplCls", ALL_IMPLS.items(), ids=ALL_IMPLS.keys())
    @pytest.mark.parametrize("n", BENCHMARK_SIZES_APPROX)
    def test_solve_potts_approx(self, benchmark, impl_name, ImplCls, n):
        _bench_solve_potts_approx(benchmark, ImplCls, impl_name, n)
