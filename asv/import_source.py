# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc


class ImportSource(abc.ABC):
    """
    Base class for benchmark result import sources.

    Plugins that want to support importing results from an external
    benchmarking tool should subclass this and implement the required
    methods.

    The import source is discovered via subclass iteration, matching
    on the ``name`` class attribute.
    """

    #: Short identifier used on the command line via ``--format``
    name = None

    @classmethod
    @abc.abstractmethod
    def add_arguments(cls, parser):
        """
        Add any format-specific arguments to the argument parser.
        Called during ``asv import --help`` setup.
        """

    @classmethod
    @abc.abstractmethod
    def import_results(cls, path, conf, args):
        """
        Import benchmark results from *path* and return a list of
        :class:`~asv.results.Results` objects ready to be saved.

        Parameters
        ----------
        path : str
            Path to the input file or directory.
        conf : Config
            The asv configuration.
        args : argparse.Namespace
            Parsed command-line arguments (includes format-specific args).

        Returns
        -------
        results : list of Results
            Each Results object represents one (machine, commit, env)
            combination and can be saved with ``result.save(conf.results_dir)``.
        benchmarks : dict
            Mapping of benchmark name to benchmark metadata dict.  These
            are merged into the existing ``benchmarks.json``.
        """
