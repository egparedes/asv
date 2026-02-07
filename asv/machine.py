# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from . import util


def iter_machine_files(results_dir):
    """
    Iterate over all of the machine.json files in the results_dir
    """
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if filename == 'machine.json':
                path = os.path.join(root, filename)
                yield path


class Machine:
    """
    Stores information about a particular machine.
    """

    api_version = 1

    @classmethod
    def update(cls, path):
        util.update_json(cls, path, cls.api_version)
