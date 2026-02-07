# Licensed under a 3-clause BSD style license - see LICENSE.rst

from os.path import dirname, join

from asv import commands, config
from asv.commands import Command


def test_config():
    conf = config.Config.load(join(dirname(__file__), 'asv.conf.json'))

    assert conf.project == 'astropy'
    assert conf.benchmark_dir == 'benchmark'
    assert conf.branches == [None]


class CustomCommand(Command):
    @classmethod
    def setup_arguments(cls, subparsers):
        parser = subparsers.add_parser("custom", help="Custom command", description="Just a test.")

        parser.set_defaults(func=cls.run_from_args)

        return parser

    @classmethod
    def run_from_conf_args(cls, conf, args):
        pass


def test_custom_command():
    parser, subparsers = commands.make_argparser()
    args = parser.parse_args(['custom'])

    assert hasattr(args, 'func')

    args.func(args)
