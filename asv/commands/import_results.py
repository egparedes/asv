# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from asv import import_source, util
from asv.benchmarks import Benchmarks
from asv.commands import Command
from asv.console import log


class Import(Command):
    @classmethod
    def setup_arguments(cls, subparsers):
        parser = subparsers.add_parser(
            "import",
            help="Import benchmark results from other tools",
            description=(
                "Import benchmark results from other benchmarking tools "
                "into the asv results directory.  Use --format to select "
                "the import source."
            ),
        )

        sources = {
            src.name: src
            for src in util.iter_subclasses(import_source.ImportSource)
            if src.name is not None
        }
        available = ", ".join(sorted(sources)) if sources else "(none)"
        parser.add_argument(
            "path",
            help="Path to the file or directory to import",
        )
        parser.add_argument(
            "--format",
            "-f",
            dest="format",
            default=None,
            help=f"Import format.  Available: {available}",
        )

        # Let each registered source add its own arguments
        for src in sources.values():
            src.add_arguments(parser)

        parser.set_defaults(func=cls.run_from_args)
        return parser

    @classmethod
    def run_from_conf_args(cls, conf, args):
        return cls.run(conf=conf, path=args.path, fmt=args.format, args=args)

    @classmethod
    def run(cls, conf, path, fmt=None, args=None):
        # Discover available import sources
        sources = {
            src.name: src
            for src in util.iter_subclasses(import_source.ImportSource)
            if src.name is not None
        }

        if not sources:
            raise util.UserError(
                "No import sources available.  Install a plugin that "
                "provides an ImportSource (e.g. the built-in 'bencher' plugin)."
            )

        if fmt is None:
            if len(sources) == 1:
                fmt = next(iter(sources))
            else:
                raise util.UserError(
                    "Multiple import formats available; please specify "
                    f"one with --format.  Available: {', '.join(sorted(sources))}"
                )

        if fmt not in sources:
            raise util.UserError(
                f"Unknown import format {fmt!r}.  "
                f"Available: {', '.join(sorted(sources))}"
            )

        source_cls = sources[fmt]

        if not os.path.exists(path):
            raise util.UserError(f"Path does not exist: {path}")

        log.info(f"Importing from {path!r} (format: {fmt})")
        result_list, new_benchmarks = source_cls.import_results(path, conf, args)

        if not result_list:
            log.info("No results to import.")
            return

        # Ensure results directory exists
        os.makedirs(conf.results_dir, exist_ok=True)

        # Merge benchmark metadata into benchmarks.json
        bench_path = Benchmarks.get_benchmark_file_path(conf.results_dir)
        if os.path.isfile(bench_path):
            existing = util.load_json(bench_path, api_version=Benchmarks.api_version)
        else:
            existing = {}

        existing.update(new_benchmarks)
        util.write_json(bench_path, existing, Benchmarks.api_version)

        # Save each result file
        saved = 0
        for result in result_list:
            machine_dir = os.path.join(conf.results_dir, result.params["machine"])
            os.makedirs(machine_dir, exist_ok=True)
            result.save(conf.results_dir)
            saved += 1
            log.step()

        log.info(f"Imported {saved} result(s) into {conf.results_dir}")
        log.info(f"Updated benchmarks.json with {len(new_benchmarks)} benchmark(s)")
