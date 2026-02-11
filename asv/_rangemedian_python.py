# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Pure Python implementation of the RangeMedian class,
equivalent to the C++ _rangemedian extension.

No external dependencies beyond the standard library.

Provides:
    mu(l, r) = median(y[l:r+1], weights=w[l:r+1])
    dist(l, r) = sum(w*abs(x - mu(l, r)) for x, w in zip(y[l:r+1], w[l:r+1]))
    find_best_partition(gamma, min_size, max_size, min_pos, max_pos)
"""

import math


def _compute_weighted_median(y, w, left, right):
    """
    Compute weighted median and L1 distance for the range [left, right].

    Matches the C++ compute_weighted_median behavior exactly:
    - Sort (value, weight) pairs by value
    - Walk sorted sequence accumulating weight
    - Median is at first element where cumulative weight >= midpoint
    - If cumulative weight == midpoint exactly, average with next element
    """
    n = right - left + 1
    if n == 0:
        return 0.0, 0.0

    # Build and sort (value, weight) pairs
    pairs = sorted(
        ((y[left + i], w[left + i]) for i in range(n)),
        key=lambda p: p[0],
    )

    # Compute midpoint = sum(weights) / 2
    midpoint = sum(p[1] for p in pairs) / 2.0

    # Find weighted median
    wsum = 0.0
    mu = pairs[-1][0]  # fallback (matches C++ error path)
    for i, (val, wt) in enumerate(pairs):
        wsum += wt
        if wsum >= midpoint:
            mu = val
            if wsum == midpoint and i + 1 < n:
                mu = (mu + pairs[i + 1][0]) / 2.0
            break

    # Compute L1 distance using original (unsorted) data
    dist = sum(w[j] * abs(y[j] - mu) for j in range(left, right + 1))

    return mu, dist


class RangeMedian:
    """
    Pure Python range median computation.

    Drop-in replacement for _rangemedian.RangeMedian (C++ extension).
    Provides the same interface: mu(l, r), dist(l, r),
    find_best_partition(gamma, min_size, max_size, min_pos, max_pos).
    """

    def __init__(self, y, w):
        if len(y) != len(w):
            raise ValueError("y and w must have same length")
        self._y = [float(v) for v in y]
        self._w = [float(v) for v in w]
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

        y = self._y
        w = self._w
        i0 = min_pos
        i1 = max_pos
        n = i1 - i0

        B = [0.0] * (n + 1)
        B[0] = -gamma
        p = [0] * n

        for right in range(i0, i1):
            B[right + 1 - i0] = math.inf

            aa = max(right + 1 - max_size, i0)
            bb = max(right + 1 - min_size + 1, i0)

            for left in range(aa, bb):
                # Inline cache lookup + compute
                key = (left, right)
                cached = self._cache.get(key)
                if cached is None:
                    cached = _compute_weighted_median(y, w, left, right)
                    self._cache[key] = cached
                _, dist = cached

                b = B[left - i0] + gamma + dist
                if b <= B[right + 1 - i0]:
                    B[right + 1 - i0] = b
                    p[right - i0] = left - 1

        return p

    def cleanup_cache(self):
        if len(self._cache) < 500000:
            return
        self._cache.clear()
