.. _conf-reference:

``asv.conf.json`` reference
===========================

The ``asv.conf.json`` file contains information about a particular
benchmarking project.  The following describes each of the keys in
this file and their expected values.

.. only:: not man

``version``
-----------
Version of the ``asv.conf.json`` spec. Currently only ``1``.

``project``
-----------
The name of the project being benchmarked.

``project_url``
---------------
The URL to the homepage of the project.  This can point to anywhere,
really, as it's only used for the link at the top of the benchmark
results page back to your project.

``repo``
--------
The URL or path to the repository for the project.

Currently, only ``git`` and ``hg`` repositories are supported.
A local path or mirror is required for analysis.

``branches``
------------
Branches to generate benchmark results for.

This controls how the benchmark results are displayed.

If not provided, "main" (Git) or "default" (Mercurial) is chosen.

``show_commit_url``
-------------------
The base URL to show information about a particular commit.  The
commit hash will be added to the end of this URL and then opened in a
new tab when a data point is clicked on in the web interface.

For example, if using Github to host your repository, the
``show_commit_url`` should be:

    https://github.com/owner/project/commit/

``benchmark_dir``
-----------------
The directory, relative to the current directory, that benchmarks are
stored in.  Should rarely need to be overridden.  If not provided,
defaults to ``"benchmarks"``.

``results_dir``
---------------
The directory, relative to the current directory, that the raw results
are stored in.  If not provided, defaults to ``"results"``.

``html_dir``
------------
The directory, relative to the current directory, to save the website
content in.  If not provided, defaults to ``"html"``.

``hash_length``
---------------
The number of characters to retain in the commit hashes when displayed
in the web interface.  The default value of 8 should be more than
enough for most projects, but projects with extremely large history
may need to increase this value.  This does not affect the storage of
results, where the full commit hash is always retained.

``plugins``
-----------
A list of modules to import containing asv plugins.

``regressions_first_commits``
-----------------------------

The commits after which the regression search in :ref:`cmd-asv-publish`
should start looking for regressions.

The value is a dictionary mapping benchmark identifier regexps to
commits after which to look for regressions. The benchmark identifiers
are of the form ``benchmark_name(parameters)@branch``, where
``(parameters)`` is present only for parameterized benchmarks. If the
commit identifier is *null*, regression detection for the matching
benchmark is skipped.  The default is to start from the first commit
with results.

Example::

    "regressions_first_commits": {
        ".*": "v0.1.0",
        "benchmark_1": "80fca08d",
        "benchmark_2@main": null,
    }

In this case, regressions are detected only for commits after tag
``v0.1.0`` for all benchmarks. For ``benchmark_1``, regression
detection is further limited to commits after the commit given, and
for ``benchmark_2``, regression detection is skipped completely in the
``main`` branch.

``regressions_thresholds``
--------------------------

The minimum relative change required before :ref:`cmd-asv-publish` reports a
regression.

The value is a dictionary, similar to ``regressions_first_commits``.
If multiple entries match, the largest threshold is taken.  If no
entry matches, the default threshold is ``0.05`` (iow. 5%).

Example::

    "regressions_thresholds": {
        ".*": 0.01,
        "benchmark_1": 0.2,
    }

In this case, the reporting threshold is 1% for all benchmarks, except
``benchmark_1`` which uses a threshold of 20%.
