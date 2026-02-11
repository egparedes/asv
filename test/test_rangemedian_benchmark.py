# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Benchmark and correctness tests comparing the C++ _rangemedian extension
with the Numba implementation (_rangemedian_numba).

Run benchmarks:
    pytest test/test_rangemedian_benchmark.py -v --benchmark-enable

Run correctness tests only:
    pytest test/test_rangemedian_benchmark.py -v -k TestCorrectness
"""

import random

import numpy as np
import pytest

from asv import _rangemedian
from asv._rangemedian_numba import RangeMedian as NumbaRangeMedian
from asv._rangemedian_numba import warmup as numba_warmup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_SIZES = [10, 50, 100, 500, 1000]


@pytest.fixture(scope="session", autouse=True)
def _warmup_numba():
    """Ensure Numba JIT compilation is done before benchmarks."""
    numba_warmup()


def _make_data(n, seed=42):
    """Generate reproducible test data of size n."""
    rng = np.random.RandomState(seed)
    y = rng.randn(n).tolist()
    w = np.abs(rng.randn(n) + 0.1).tolist()
    w = [max(x, 0.01) for x in w]
    return y, w


def _make_step_data(n, seed=42):
    """Generate step-function data with noise (typical use case)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    n_steps = max(1, n // 20)
    y0 = np.zeros(n, dtype=np.float64)
    for i in range(1, n_steps):
        pos = int(n * i / n_steps)
        y0[pos:] += rng.randn() * 0.5
    y = (y0 + 0.05 * rng.randn(n)).tolist()
    w = [1.0] * n
    return y, w


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestCorrectness:
    """Verify that the Numba implementation matches C++ exactly."""

    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_mu_dist_match(self, n):
        """mu() and dist() produce identical results for random ranges."""
        y, w = _make_data(n)
        cpp = _rangemedian.RangeMedian(y, w)
        nb = NumbaRangeMedian(y, w)

        rng = random.Random(123)
        for _ in range(min(500, n * n)):
            l = rng.randint(0, n - 1)
            r = rng.randint(l, n - 1)
            assert cpp.mu(l, r) == pytest.approx(nb.mu(l, r), abs=1e-12), (
                f"mu mismatch at ({l}, {r})"
            )
            assert cpp.dist(l, r) == pytest.approx(nb.dist(l, r), abs=1e-12), (
                f"dist mismatch at ({l}, {r})"
            )

    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_find_best_partition_match(self, n):
        """find_best_partition() produces identical partitions."""
        y, w = _make_data(n)
        for gamma in [0.1, 0.5, 2.0]:
            for min_size, max_size in [(1, min(20, n)), (2, min(10, n)), (5, min(50, n))]:
                if min_size > max_size or min_size > n:
                    continue
                cpp = _rangemedian.RangeMedian(y, w)
                nb = NumbaRangeMedian(y, w)
                cpp_p = cpp.find_best_partition(gamma, min_size, max_size, 0, n)
                nb_p = nb.find_best_partition(gamma, min_size, max_size, 0, n)
                assert cpp_p == nb_p, (
                    f"partition mismatch: n={n}, gamma={gamma}, "
                    f"min_size={min_size}, max_size={max_size}"
                )

    def test_edge_single_element(self):
        """Single element edge case."""
        cpp = _rangemedian.RangeMedian([42.0], [1.0])
        nb = NumbaRangeMedian([42.0], [1.0])
        assert cpp.mu(0, 0) == nb.mu(0, 0)
        assert cpp.dist(0, 0) == nb.dist(0, 0)

    def test_edge_uniform_step_data(self):
        """Uniform weights, step function data."""
        y = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        w = [1.0] * 9
        cpp = _rangemedian.RangeMedian(y, w)
        nb = NumbaRangeMedian(y, w)
        for l in range(9):
            for r in range(l, 9):
                assert cpp.mu(l, r) == pytest.approx(nb.mu(l, r), abs=1e-12)
                assert cpp.dist(l, r) == pytest.approx(nb.dist(l, r), abs=1e-12)

    def test_edge_even_count_midpoint(self):
        """Even number of elements where cumulative weight == midpoint exactly."""
        y = [1.0, 2.0, 3.0, 4.0]
        w = [1.0, 1.0, 1.0, 1.0]
        cpp = _rangemedian.RangeMedian(y, w)
        nb = NumbaRangeMedian(y, w)
        assert cpp.mu(0, 3) == pytest.approx(nb.mu(0, 3), abs=1e-12)
        assert cpp.dist(0, 3) == pytest.approx(nb.dist(0, 3), abs=1e-12)

    def test_edge_varied_weights(self):
        """Non-uniform weights that shift the median."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        w = [0.1, 0.1, 0.1, 0.1, 10.0]  # heavily weighted last element
        cpp = _rangemedian.RangeMedian(y, w)
        nb = NumbaRangeMedian(y, w)
        for l in range(5):
            for r in range(l, 5):
                assert cpp.mu(l, r) == pytest.approx(nb.mu(l, r), abs=1e-12)
                assert cpp.dist(l, r) == pytest.approx(nb.dist(l, r), abs=1e-12)

    @pytest.mark.parametrize("n", [20, 100])
    def test_solve_potts_integration(self, n):
        """Full solve_potts produces identical results with Numba backend."""
        from asv.step_detect import solve_potts

        y, w = _make_step_data(n)
        cpp_md = _rangemedian.RangeMedian(y, w)
        nb_md = NumbaRangeMedian(y, w)

        right_cpp, values_cpp, dists_cpp = solve_potts(y, w=w, gamma=0.1, mu_dist=cpp_md)
        right_nb, values_nb, dists_nb = solve_potts(y, w=w, gamma=0.1, mu_dist=nb_md)

        assert right_cpp == right_nb
        for vc, vn in zip(values_cpp, values_nb):
            assert vc == pytest.approx(vn, abs=1e-12)
        for dc, dn in zip(dists_cpp, dists_nb):
            assert dc == pytest.approx(dn, abs=1e-12)

    @pytest.mark.parametrize("n", [100, 500])
    def test_solve_potts_approx_integration(self, n):
        """solve_potts_approx (typical usage) produces identical results."""
        from asv.step_detect import solve_potts_approx

        y, w = _make_step_data(n)
        cpp_md = _rangemedian.RangeMedian(y, w)
        nb_md = NumbaRangeMedian(y, w)

        right_cpp, values_cpp, dists_cpp = solve_potts_approx(
            y, w=w, gamma=0.5, mu_dist=cpp_md
        )
        right_nb, values_nb, dists_nb = solve_potts_approx(
            y, w=w, gamma=0.5, mu_dist=nb_md
        )

        assert right_cpp == right_nb
        for vc, vn in zip(values_cpp, values_nb):
            assert vc == pytest.approx(vn, abs=1e-12)
        for dc, dn in zip(dists_cpp, dists_nb):
            assert dc == pytest.approx(dn, abs=1e-12)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestBenchmarkMuDist:
    """Benchmark individual mu() and dist() calls."""

    @pytest.mark.benchmark(group="mu_dist")
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_cpp_mu_dist(self, benchmark, n):
        """Benchmark C++ mu() + dist() calls over random ranges."""
        y, w = _make_data(n)
        rm = _rangemedian.RangeMedian(y, w)
        rng = random.Random(999)
        ranges = [(rng.randint(0, n - 1), 0) for _ in range(200)]
        ranges = [(l, rng.randint(l, n - 1)) for l, _ in ranges]

        def run():
            for l, r in ranges:
                rm.mu(l, r)
                rm.dist(l, r)

        benchmark.extra_info["impl"] = "cpp"
        benchmark.extra_info["n"] = n
        benchmark(run)

    @pytest.mark.benchmark(group="mu_dist")
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_numba_mu_dist(self, benchmark, n):
        """Benchmark Numba mu() + dist() calls over random ranges."""
        y, w = _make_data(n)
        rm = NumbaRangeMedian(y, w)
        rng = random.Random(999)
        ranges = [(rng.randint(0, n - 1), 0) for _ in range(200)]
        ranges = [(l, rng.randint(l, n - 1)) for l, _ in ranges]

        def run():
            for l, r in ranges:
                rm.mu(l, r)
                rm.dist(l, r)

        benchmark.extra_info["impl"] = "numba"
        benchmark.extra_info["n"] = n
        benchmark(run)


class TestBenchmarkFindBestPartition:
    """Benchmark find_best_partition (the core DP algorithm)."""

    @pytest.mark.benchmark(group="find_best_partition")
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_cpp_find_best_partition(self, benchmark, n):
        """Benchmark C++ find_best_partition with max_size=20 (typical)."""
        y, w = _make_step_data(n)

        def run():
            rm = _rangemedian.RangeMedian(y, w)
            rm.find_best_partition(0.5, 1, min(20, n), 0, n)

        benchmark.extra_info["impl"] = "cpp"
        benchmark.extra_info["n"] = n
        benchmark(run)

    @pytest.mark.benchmark(group="find_best_partition")
    @pytest.mark.parametrize("n", INPUT_SIZES)
    def test_numba_find_best_partition(self, benchmark, n):
        """Benchmark Numba find_best_partition with max_size=20 (typical)."""
        y, w = _make_step_data(n)

        def run():
            rm = NumbaRangeMedian(y, w)
            rm.find_best_partition(0.5, 1, min(20, n), 0, n)

        benchmark.extra_info["impl"] = "numba"
        benchmark.extra_info["n"] = n
        benchmark(run)


class TestBenchmarkSolvePotts:
    """Benchmark solve_potts (exact solver, max_size=n)."""

    @pytest.mark.benchmark(group="solve_potts")
    @pytest.mark.parametrize("n", [50, 100, 500])
    def test_cpp_solve_potts(self, benchmark, n):
        """Benchmark solve_potts with C++ backend."""
        from asv.step_detect import solve_potts

        y, w = _make_step_data(n)

        def run():
            rm = _rangemedian.RangeMedian(y, w)
            solve_potts(y, w=w, gamma=0.5, mu_dist=rm)

        benchmark.extra_info["impl"] = "cpp"
        benchmark.extra_info["n"] = n
        benchmark(run)

    @pytest.mark.benchmark(group="solve_potts")
    @pytest.mark.parametrize("n", [50, 100, 500])
    def test_numba_solve_potts(self, benchmark, n):
        """Benchmark solve_potts with Numba backend."""
        from asv.step_detect import solve_potts

        y, w = _make_step_data(n)

        def run():
            rm = NumbaRangeMedian(y, w)
            solve_potts(y, w=w, gamma=0.5, mu_dist=rm)

        benchmark.extra_info["impl"] = "numba"
        benchmark.extra_info["n"] = n
        benchmark(run)


class TestBenchmarkSolvePottsApprox:
    """Benchmark solve_potts_approx (typical workload, max_size bounded)."""

    @pytest.mark.benchmark(group="solve_potts_approx")
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_cpp_solve_potts_approx(self, benchmark, n):
        """Benchmark solve_potts_approx with C++ backend."""
        from asv.step_detect import solve_potts_approx

        y, w = _make_step_data(n)

        def run():
            rm = _rangemedian.RangeMedian(y, w)
            solve_potts_approx(y, w=w, gamma=0.5, mu_dist=rm)

        benchmark.extra_info["impl"] = "cpp"
        benchmark.extra_info["n"] = n
        benchmark(run)

    @pytest.mark.benchmark(group="solve_potts_approx")
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_numba_solve_potts_approx(self, benchmark, n):
        """Benchmark solve_potts_approx with Numba backend."""
        from asv.step_detect import solve_potts_approx

        y, w = _make_step_data(n)

        def run():
            rm = NumbaRangeMedian(y, w)
            solve_potts_approx(y, w=w, gamma=0.5, mu_dist=rm)

        benchmark.extra_info["impl"] = "numba"
        benchmark.extra_info["n"] = n
        benchmark(run)
