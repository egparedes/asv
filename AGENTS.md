# Repository Overview

This repository contains a fork of ASV (Airspeed Velocity). ASV is a Python benchmarking framework that tracks performance of Python projects over their repository history. This repository contains a trimmed down fork which does not run the benchmarks and and so it only does statistical analysis and visualization of the benchmark results generating a static HTML dashboard with interactive graphs

## Tech Stack

- Language:	Python 3.9+ with an optional numba-accelerated extension (_rangemedian_numba.py)
- Build:	setuptools + setuptools_scm (version from git tags)
- Testing:	pytest (with xdist, timeout, rerunfailures plugins)
- Linting:	ruff
- Docs:	Sphinx + Furo theme, hosted on ReadTheDocs
- Web frontend:	Static HTML/JS — jQuery + Flot for charts
- Key deps:	json5, tabulate, packaging, numba (optional)

## Directory Layout
    
    asv/                   # Core package
    commands/            # CLI command implementations (one file per command)
    plugins/             # Pluggable backends (git, conda, virtualenv, uv, rattler, regressions…)
    www/                 # Static web frontend (HTML/JS/CSS)
    template/            # Quickstart project template
    test/                  # 20+ pytest modules
    docs/                  # Sphinx documentation
    benchmarks/            # ASV benchmarking itself (dogfooding)
    pyproject.toml         # Project metadata, deps, tool config
    asv.conf.json          # ASV's own benchmark config


## Main Entry Points
    CLI — asv/__main__.py:main (installed as the asv command)
    Key commands in typical workflow order:


## Architecture

The core workflow is: Config → JSON result storage → Statistical regression detection → Static HTML generation. 
The codebase is plugin-driven display components and benchmarks importers are all pluggable via asv/plugins/.
