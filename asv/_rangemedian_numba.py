# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Fast range median distance computations for dataset `y`:
#
#    mu(l, r) = median(y[l:r+1])
#    dist(l, r) = sum(abs(x - mu(l, r)) for x in y[l:r+1])
#
# and an implementation of the find-best-partition dynamic program.
#
# This is a numba-accelerated replacement for the C++ _rangemedian extension.

import numba as nb
import numpy as np

# Sentinel value for empty hash table slots
_EMPTY = np.int64(-1)


@nb.njit(cache=True)
def _compute_weighted_median(values, weights, n):
    """Compute weighted median and L1 distance for a slice of (value, weight) pairs."""
    if n == 0:
        return 0.0, 0.0

    # Sort by value using argsort (O(n log n))
    idx = np.argsort(values[:n])

    # Find midpoint
    midpoint = 0.0
    for i in range(n):
        midpoint += weights[idx[i]]
    midpoint /= 2.0

    # Find weighted median
    wsum = 0.0
    median_idx = n - 1  # fallback
    for i in range(n):
        wsum += weights[idx[i]]
        if wsum >= midpoint:
            median_idx = i
            break

    mu = values[idx[median_idx]]
    if wsum == midpoint and median_idx + 1 < n:
        mu = (mu + values[idx[median_idx + 1]]) / 2.0

    # Compute distance
    dist = 0.0
    for i in range(n):
        dist += weights[idx[i]] * abs(values[idx[i]] - mu)

    return mu, dist


@nb.njit(cache=True)
def _compute_mu_dist_single(y_vals, y_weights, left, right):
    """Compute mu and dist for a single (left, right) range. No caching."""
    n = right - left + 1
    return _compute_weighted_median(y_vals[left:right + 1], y_weights[left:right + 1], n)


@nb.njit(cache=True)
def _hash_key(left, right, capacity):
    """Compute hash table index for (left, right) pair using open addressing."""
    h = nb.uint64(left * 1000003) ^ nb.uint64(right * 999983)
    return nb.int64(h % nb.uint64(capacity))


@nb.njit(cache=True)
def _cache_lookup(cache_left, cache_right, cache_mu, cache_dist, left, right):
    """Look up (left, right) in hash table cache. Returns (found, mu, dist)."""
    capacity = cache_left.shape[0]
    idx = _hash_key(left, right, capacity)

    for _ in range(capacity):
        if cache_left[idx] == _EMPTY:
            return False, 0.0, 0.0
        if cache_left[idx] == left and cache_right[idx] == right:
            return True, cache_mu[idx], cache_dist[idx]
        idx += 1
        if idx >= capacity:
            idx = 0

    return False, 0.0, 0.0


@nb.njit(cache=True)
def _cache_store(cache_left, cache_right, cache_mu, cache_dist,
                 left, right, mu, dist):
    """Store (left, right) -> (mu, dist) in hash table cache.

    If the table is full, the entry is silently dropped (this is acceptable
    since the cache is sized generously at init time).
    """
    capacity = cache_left.shape[0]
    idx = _hash_key(left, right, capacity)

    for _ in range(capacity):
        if cache_left[idx] == _EMPTY:
            cache_left[idx] = left
            cache_right[idx] = right
            cache_mu[idx] = mu
            cache_dist[idx] = dist
            return
        if cache_left[idx] == left and cache_right[idx] == right:
            return
        idx += 1
        if idx >= capacity:
            idx = 0


@nb.njit(cache=True)
def _cached_mu_dist(y_vals, y_weights, left, right,
                    cache_left, cache_right, cache_mu, cache_dist):
    """Look up (left, right) in cache or compute and store."""
    found, mu, dist = _cache_lookup(
        cache_left, cache_right, cache_mu, cache_dist, left, right
    )
    if found:
        return mu, dist

    n = right - left + 1
    mu, dist = _compute_weighted_median(y_vals[left:right + 1], y_weights[left:right + 1], n)
    _cache_store(cache_left, cache_right, cache_mu, cache_dist, left, right, mu, dist)
    return mu, dist


@nb.njit(cache=True)
def _find_best_partition(y_vals, y_weights, gamma, min_size, max_size, min_pos, max_pos,
                         cache_left, cache_right, cache_mu, cache_dist):
    """Find best partition using dynamic programming (Bellman recursion)."""
    size = max_pos - min_pos
    B = np.empty(size + 1, dtype=np.float64)
    p = np.empty(size, dtype=np.int64)

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
            mu, dist = _cached_mu_dist(
                y_vals, y_weights, left, right,
                cache_left, cache_right, cache_mu, cache_dist
            )

            b = B[left - min_pos] + gamma + dist
            if b <= B[right + 1 - min_pos]:
                B[right + 1 - min_pos] = b
                p[right - min_pos] = left - 1

    return p


def _make_cache(capacity):
    """Create hash table cache arrays with given capacity."""
    cache_left = np.full(capacity, _EMPTY, dtype=np.int64)
    cache_right = np.full(capacity, _EMPTY, dtype=np.int64)
    cache_mu = np.empty(capacity, dtype=np.float64)
    cache_dist = np.empty(capacity, dtype=np.float64)
    return cache_left, cache_right, cache_mu, cache_dist


class RangeMedian:
    """Numba-accelerated range median computation.

    Drop-in replacement for the C++ _rangemedian.RangeMedian class.

    Uses two cache layers:
    - A numba-level hash table for ``find_best_partition`` (stays in compiled code)
    - A Python dict for individual ``mu``/``dist`` calls (avoids numba call overhead)

    After ``find_best_partition`` runs, its numba cache entries are drained into the
    Python dict so that subsequent ``mu``/``dist`` calls from Python (e.g. in
    ``merge_pieces``) get immediate cache hits.

    Parameters
    ----------
    y : list of float
        Data values.
    w : list of float
        Data weights.
    """

    def __init__(self, y, w):
        if len(y) != len(w):
            raise ValueError("y and w must have same length")

        self._y_vals = np.array(y, dtype=np.float64)
        self._y_weights = np.array(w, dtype=np.float64)
        self._size = len(y)

        # Numba-level hash table cache (used internally by find_best_partition)
        cache_capacity = max(1024, len(y) * 64) | 1
        self._nb_cache = _make_cache(cache_capacity)

        # Python-level dict cache for fast access from Python callers.
        self._py_cache = {}

    def _drain_nb_cache(self):
        """Copy all numba hash table entries into the Python dict cache."""
        cache_left, cache_right, cache_mu, cache_dist = self._nb_cache
        py_cache = self._py_cache
        for i in range(cache_left.shape[0]):
            left = cache_left[i]
            if left != _EMPTY:
                key = (int(left), int(cache_right[i]))
                if key not in py_cache:
                    py_cache[key] = (float(cache_mu[i]), float(cache_dist[i]))

    def _get_mu_dist(self, left, right):
        """Compute or retrieve cached (mu, dist) for interval [left, right]."""
        key = (left, right)
        result = self._py_cache.get(key)
        if result is not None:
            return result

        mu, dist = _compute_mu_dist_single(self._y_vals, self._y_weights, left, right)
        self._py_cache[key] = (mu, dist)
        return mu, dist

    def mu(self, left, right):
        """Return weighted median of y[left:right+1]."""
        if left < 0 or right < 0 or left >= self._size or right >= self._size:
            raise ValueError("argument out of range")
        return self._get_mu_dist(left, right)[0]

    def dist(self, left, right):
        """Return sum of weighted absolute deviations from median for y[left:right+1]."""
        if left < 0 or right < 0 or left >= self._size or right >= self._size:
            raise ValueError("argument out of range")
        return self._get_mu_dist(left, right)[1]

    def find_best_partition(self, gamma, min_size, max_size, min_pos, max_pos):
        """Find optimal partition using dynamic programming.

        Parameters
        ----------
        gamma : float
            Penalty parameter.
        min_size : int
            Minimum interval size.
        max_size : int
            Maximum interval size.
        min_pos : int
            Start position (inclusive).
        max_pos : int
            End position (exclusive).

        Returns
        -------
        p : list of int
            Partition boundary indices.
        """
        size = self._y_vals.shape[0]
        if not (0 < min_size <= max_size and
                0 <= min_pos <= max_pos <= size):
            raise ValueError("invalid input indices")

        p = _find_best_partition(
            self._y_vals, self._y_weights,
            gamma, min_size, max_size, min_pos, max_pos,
            *self._nb_cache
        )

        # Sync numba cache into Python dict so subsequent mu/dist calls
        # from Python (e.g. merge_pieces) get immediate cache hits.
        self._drain_nb_cache()

        return p.tolist()
