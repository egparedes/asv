# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Numba-accelerated implementation of the RangeMedian class,
equivalent to the C++ _rangemedian extension.

Falls back to NumPy-only or pure Python implementations when Numba is not
available.  Import this module and use ``RangeMedian`` — the best available
backend is selected automatically.

Provides:
    mu(l, r) = median(y[l:r+1], weights=w[l:r+1])
    dist(l, r) = sum(w*abs(x - mu(l, r)) for x, w in zip(y[l:r+1], w[l:r+1]))
    find_best_partition(gamma, min_size, max_size, min_pos, max_pos)
"""

try:
    import numpy as np
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

if not HAVE_NUMBA:
    # Re-export the best available fallback as RangeMedian so that
    # ``from asv._rangemedian_numba import RangeMedian`` always works.
    try:
        from asv._rangemedian_numpy import RangeMedian  # noqa: F401

        def warmup():
            """No-op: NumPy backend needs no JIT warm-up."""

        BACKEND = "numpy"
    except ImportError:
        from asv._rangemedian_python import RangeMedian  # noqa: F401

        def warmup():
            """No-op: pure-Python backend needs no JIT warm-up."""

        BACKEND = "python"
else:
    BACKEND = "numba"

    # -----------------------------------------------------------------------
    # Numba JIT-compiled implementation (only defined when numba is available)
    # -----------------------------------------------------------------------

    # Threshold below which insertion sort is used instead of merge sort.
    _SORT_THRESHOLD = 32

    @njit(cache=True)
    def _insertion_sort_pairs(vals, wts, n):
        """In-place insertion sort of (vals, wts) by vals. O(n^2) but fast for small n."""
        for i in range(1, n):
            key_v = vals[i]
            key_w = wts[i]
            j = i - 1
            while j >= 0 and vals[j] > key_v:
                vals[j + 1] = vals[j]
                wts[j + 1] = wts[j]
                j -= 1
            vals[j + 1] = key_v
            wts[j + 1] = key_w

    @njit(cache=True)
    def _merge_halves(vals, wts, buf_v, buf_w, lo, mid, hi):
        """Merge two sorted runs [lo, mid) and [mid, hi). Half-copy optimization."""
        left_size = mid - lo
        for i in range(left_size):
            buf_v[i] = vals[lo + i]
            buf_w[i] = wts[lo + i]

        i = 0
        j = mid
        k = lo
        while i < left_size and j < hi:
            if buf_v[i] <= vals[j]:
                vals[k] = buf_v[i]
                wts[k] = buf_w[i]
                i += 1
            else:
                vals[k] = vals[j]
                wts[k] = wts[j]
                j += 1
            k += 1
        while i < left_size:
            vals[k] = buf_v[i]
            wts[k] = buf_w[i]
            i += 1
            k += 1
        # Right half remainder is already in place — no copy needed.

    @njit(cache=True)
    def _hybrid_sort_pairs(vals, wts, n, buf_v, buf_w):
        """
        Hybrid sort: insertion sort for small blocks, bottom-up merge sort for larger.
        Sorts (vals, wts) pairs by vals. O(n*log(n)) with low overhead for small n.
        Only copies the left half during merge (half-copy optimization).
        """
        block = _SORT_THRESHOLD
        for start in range(0, n, block):
            end = min(start + block, n)
            for i in range(start + 1, end):
                key_v = vals[i]
                key_w = wts[i]
                j = i - 1
                while j >= start and vals[j] > key_v:
                    vals[j + 1] = vals[j]
                    wts[j + 1] = wts[j]
                    j -= 1
                vals[j + 1] = key_v
                wts[j + 1] = key_w

        size = block
        while size < n:
            for lo in range(0, n, 2 * size):
                mid = min(lo + size, n)
                hi = min(lo + 2 * size, n)
                if mid < hi:
                    _merge_halves(vals, wts, buf_v, buf_w, lo, mid, hi)
            size *= 2

    @njit(cache=True)
    def _compute_weighted_median(y, w, left, right, tmp_vals, tmp_wts, buf_v, buf_w):
        """
        Compute weighted median and L1 distance for the range [left, right].

        Uses pre-allocated temporary arrays to avoid allocations in hot loops.
        Matches the C++ compute_weighted_median behavior exactly.
        """
        n = right - left + 1
        if n == 0:
            return 0.0, 0.0

        # Copy to temp arrays
        for i in range(n):
            tmp_vals[i] = y[left + i]
            tmp_wts[i] = w[left + i]

        # Hybrid sort: insertion sort for small n, merge sort for large n
        if n <= _SORT_THRESHOLD:
            _insertion_sort_pairs(tmp_vals, tmp_wts, n)
        else:
            _hybrid_sort_pairs(tmp_vals, tmp_wts, n, buf_v, buf_w)

        # Compute midpoint = sum(weights) / 2
        midpoint = 0.0
        for i in range(n):
            midpoint += tmp_wts[i]
        midpoint *= 0.5

        # Find weighted median
        wsum = 0.0
        mu = tmp_vals[n - 1]  # fallback (matches C++ error path)
        for i in range(n):
            wsum += tmp_wts[i]
            if wsum >= midpoint:
                mu = tmp_vals[i]
                if wsum == midpoint and i + 1 < n:
                    mu = (mu + tmp_vals[i + 1]) * 0.5
                break

        # Compute L1 distance using original (unsorted) data
        dist = 0.0
        for i in range(left, right + 1):
            dist += w[i] * abs(y[i] - mu)

        return mu, dist


    @njit(cache=True)
    def _find_best_partition_numba(y, w, gamma, min_size, max_size, min_pos, max_pos):
        """
        Find best partition using Bellman recursion (dynamic programming).

        Equivalent to RangeMedian::find_best_partition in _rangemedian.cpp.
        """
        size = max_pos - min_pos
        B = np.empty(size + 1, dtype=np.float64)
        p = np.zeros(size, dtype=np.int64)

        # Pre-allocate temp arrays for the weighted median sort
        tmp_vals = np.empty(max_size, dtype=np.float64)
        tmp_wts = np.empty(max_size, dtype=np.float64)
        buf_v = np.empty(max_size, dtype=np.float64)
        buf_w = np.empty(max_size, dtype=np.float64)

        B[0] = -gamma

        for right in range(min_pos, max_pos):
            B[right + 1 - min_pos] = np.inf

            aa = right + 1 - max_size
            if aa < min_pos:
                aa = min_pos
            bb = right + 1 - min_size + 1
            if bb < min_pos:
                bb = min_pos

            for left in range(aa, bb):
                mu, dist = _compute_weighted_median(
                    y, w, left, right, tmp_vals, tmp_wts, buf_v, buf_w
                )

                b = B[left - min_pos] + gamma + dist
                if b <= B[right + 1 - min_pos]:
                    B[right + 1 - min_pos] = b
                    p[right - min_pos] = left - 1

        return p

    class RangeMedian:
        """
        Numba-accelerated range median computation.

        Drop-in replacement for _rangemedian.RangeMedian (C++ extension).
        Provides the same interface: mu(l, r), dist(l, r),
        find_best_partition(gamma, min_size, max_size, min_pos, max_pos).
        """

        def __init__(self, y, w):
            if len(y) != len(w):
                raise ValueError("y and w must have same length")
            self._y = np.asarray(y, dtype=np.float64)
            self._w = np.asarray(w, dtype=np.float64)
            self._cache = {}
            # Pre-allocate temp arrays to avoid per-call allocation overhead
            n = len(y)
            self._tmp_vals = np.empty(n, dtype=np.float64)
            self._tmp_wts = np.empty(n, dtype=np.float64)
            self._buf_v = np.empty(n, dtype=np.float64)
            self._buf_w = np.empty(n, dtype=np.float64)

        def _get_mu_dist(self, left, right):
            key = (left, right)
            result = self._cache.get(key)
            if result is None:
                size = len(self._y)
                if left < 0 or right < 0 or left >= size or right >= size:
                    raise ValueError("argument out of range")
                result = _compute_weighted_median(
                    self._y, self._w, left, right,
                    self._tmp_vals, self._tmp_wts, self._buf_v, self._buf_w,
                )
                self._cache[key] = result
            return result

        def mu(self, left, right):
            return self._get_mu_dist(left, right)[0]

        def dist(self, left, right):
            return self._get_mu_dist(left, right)[1]

        def find_best_partition(self, gamma, min_size, max_size, min_pos, max_pos):
            size = len(self._y)
            if not (0 < min_size <= max_size and 0 <= min_pos <= max_pos <= size):
                raise ValueError("invalid input indices")
            p = _find_best_partition_numba(
                self._y, self._w, gamma, min_size, max_size, min_pos, max_pos
            )
            return p.tolist()

        def cleanup_cache(self):
            if len(self._cache) < 500000:
                return
            self._cache.clear()

    def warmup():
        """Pre-compile all numba functions to avoid JIT overhead in benchmarks."""
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        tmp = np.empty(3, dtype=np.float64)
        buf = np.empty(3, dtype=np.float64)
        _compute_weighted_median(y, w, 0, 2, tmp, tmp.copy(), buf, buf.copy())
        _find_best_partition_numba(y, w, 0.1, 1, 3, 0, 3)
