# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
NumPy-based implementation of the RangeMedian class,
equivalent to the C++ _rangemedian extension.

Requires only NumPy (no Numba).

Provides:
    mu(l, r) = median(y[l:r+1], weights=w[l:r+1])
    dist(l, r) = sum(w*abs(x - mu(l, r)) for x, w in zip(y[l:r+1], w[l:r+1]))
    find_best_partition(gamma, min_size, max_size, min_pos, max_pos)
"""

import numpy as np


def _compute_weighted_median(y, w, left, right):
    """
    Compute weighted median and L1 distance for the range [left, right].

    Uses NumPy vectorized operations for sorting and distance computation.
    Matches the C++ compute_weighted_median behavior exactly.
    """
    n = right - left + 1
    if n == 0:
        return 0.0, 0.0

    y_slice = y[left:right + 1]
    w_slice = w[left:right + 1]

    # Sort by value using argsort
    order = np.argsort(y_slice, kind='quicksort')
    sorted_y = y_slice[order]
    sorted_w = w_slice[order]

    # Compute midpoint = sum(weights) / 2
    midpoint = sorted_w.sum() * 0.5

    # Find weighted median via cumulative weight sum
    cumsum = np.cumsum(sorted_w)
    # First index where cumulative weight >= midpoint
    idx = np.searchsorted(cumsum, midpoint, side='left')
    if idx >= n:
        # Error/fallback path (matches C++ behavior)
        idx = n - 1

    mu = sorted_y[idx]
    if cumsum[idx] == midpoint and idx + 1 < n:
        mu = (mu + sorted_y[idx + 1]) * 0.5

    # Compute L1 distance vectorized
    dist = float(np.sum(w_slice * np.abs(y_slice - mu)))

    return float(mu), dist


def _find_best_partition(y, w, gamma, min_size, max_size, min_pos, max_pos):
    """
    Find best partition using Bellman recursion (dynamic programming).

    Uses NumPy arrays for B and p but calls _compute_weighted_median per cell.
    """
    size = max_pos - min_pos
    B = np.empty(size + 1, dtype=np.float64)
    p = np.zeros(size, dtype=np.int64)

    B[0] = -gamma

    for right in range(min_pos, max_pos):
        B[right + 1 - min_pos] = np.inf

        aa = max(right + 1 - max_size, min_pos)
        bb = max(right + 1 - min_size + 1, min_pos)

        for left in range(aa, bb):
            _, dist = _compute_weighted_median(y, w, left, right)

            b = B[left - min_pos] + gamma + dist
            if b <= B[right + 1 - min_pos]:
                B[right + 1 - min_pos] = b
                p[right - min_pos] = left - 1

    return p


class RangeMedian:
    """
    NumPy-based range median computation.

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

    def _get_mu_dist(self, left, right):
        key = (left, right)
        result = self._cache.get(key)
        if result is None:
            size = len(self._y)
            if left < 0 or right < 0 or left >= size or right >= size:
                raise ValueError("argument out of range")
            result = _compute_weighted_median(self._y, self._w, left, right)
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
        p = _find_best_partition(
            self._y, self._w, gamma, min_size, max_size, min_pos, max_pos
        )
        return p.tolist()

    def cleanup_cache(self):
        if len(self._cache) < 500000:
            return
        self._cache.clear()
