# Licensed under a 3-clause BSD style license - see LICENSE.rst


import itertools
import os
import re

from . import util


class Benchmarks(dict):
    """
    Manages the set of benchmarks in the project.
    """

    api_version = 2

    def __init__(self, conf, benchmarks, regex=None):
        """
        Initialize a list of benchmarks.

        Parameters
        ----------
        conf : Config object
            The project's configuration

        benchmarks : list
            Benchmarks as loaded from a file.

        regex : str or list of str, optional
            `regex` is a list of regular expressions matching the
            benchmarks to run.  If none are provided, all benchmarks
            are run.
            For parameterized benchmarks, the regex match against
            `funcname(param0, param1, ...)` to include the parameter
            combination in regex filtering.
        """
        self._conf = conf
        self._benchmark_dir = conf.benchmark_dir

        if not regex:
            regex = []
        if isinstance(regex, str):
            regex = [regex]

        self._all_benchmarks = {}
        self._benchmark_selection = {}
        for benchmark in benchmarks:
            self._all_benchmarks[benchmark['name']] = benchmark
            if benchmark['params']:
                self._benchmark_selection[benchmark['name']] = []
                for idx, param_set in enumerate(itertools.product(*benchmark['params'])):
                    name = f"{benchmark['name']}({', '.join(param_set)})"
                    if not regex or any(re.search(reg, name) for reg in regex):
                        self[benchmark['name']] = benchmark
                        self._benchmark_selection[benchmark['name']].append(idx)
            else:
                self._benchmark_selection[benchmark['name']] = None
                if not regex or any(re.search(reg, benchmark['name']) for reg in regex):
                    self[benchmark['name']] = benchmark

    @property
    def benchmark_selection(self):
        """
        Active sets of parameterized benchmarks.
        """
        return self._benchmark_selection

    @property
    def benchmark_dir(self):
        """
        Benchmark directory.
        """
        return self._benchmark_dir

    def filter_out(self, skip):
        """
        Return a new Benchmarks object, with some benchmarks filtered out.
        """
        benchmarks = super().__new__(self.__class__)
        benchmarks._conf = self._conf
        benchmarks._benchmark_dir = self._benchmark_dir
        benchmarks._all_benchmarks = self._all_benchmarks

        selected_idx = {}

        for name, benchmark in self.items():
            if name not in skip:
                benchmarks[name] = benchmark
                if name in self._benchmark_selection:
                    selected_idx[name] = self._benchmark_selection[name]

        benchmarks._benchmark_selection = selected_idx

        return benchmarks

    @classmethod
    def get_benchmark_file_path(cls, results_dir):
        """
        Get the path to the benchmarks.json file in the results dir.
        """
        return os.path.join(results_dir, "benchmarks.json")

    def save(self):
        """
        Save the ``benchmarks.json`` file, which is a cached set of the
        metadata about the discovered benchmarks, in the results dir.
        """
        path = self.get_benchmark_file_path(self._conf.results_dir)
        util.write_json(path, self._all_benchmarks, self.api_version)

    @classmethod
    def load(cls, conf, regex=None):
        """
        Load the benchmark descriptions from the `benchmarks.json` file.

        Parameters
        ----------
        conf : Config object
            The project's configuration
        regex : str or list of str, optional
            `regex` is a list of regular expressions matching the
            benchmarks to load. See __init__ docstring.

        Returns
        -------
        benchmarks : Benchmarks object
        """
        try:
            path = cls.get_benchmark_file_path(conf.results_dir)
            if not os.path.isfile(path):
                raise util.UserError(f"Benchmark list file {path} missing!")
            d = util.load_json(path, api_version=cls.api_version)
            benchmarks = d.values()
            return cls(conf, benchmarks, regex=regex)
        except util.UserError as err:
            if "asv update" in str(err):
                # Don't give conflicting instructions
                raise
            raise util.UserError(
                f"{err}\nEnsure benchmarks.json exists in the results directory."
            )
