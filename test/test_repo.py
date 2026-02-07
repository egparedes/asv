# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from os.path import join

import pytest

from asv import config, repo, util

try:
    import hglib
except ImportError:
    hglib = None

from . import tools

pytestmark = pytest.mark.skipif(tools.HAS_PYPY, reason="Randomly times out on pypy")


def _test_branches(conf, branch_commits, require_describe=False):
    r = repo.get_repo(conf)

    assert len(conf.branches) == 2

    for branch in conf.branches:
        commits = r.get_branch_commits(branch)

        for commit in branch_commits[branch]:
            assert commit in commits

            name = r.get_name_from_hash(commit)
            if require_describe:
                assert name is not None
            if name is not None:
                assert r.get_hash_from_name(name) == commit
                assert name in r.get_decorated_hash(commit)


def test_repo_git(tmpdir):
    tmpdir = str(tmpdir)

    dvcs = tools.generate_test_repo(
        tmpdir,
        list(range(10)),
        dvcs_type='git',
        extra_branches=[(f'{util.git_default_branch()}~4', 'some-branch', [11, 12, 13])],
    )

    conf = config.Config()
    conf.project = join(tmpdir, "repo")
    conf.repo = dvcs.path

    r = repo.get_repo(conf)

    # Test hash range
    hashes = r.get_hashes_from_range(
        f'{util.git_default_branch()}~4..{util.git_default_branch()}'
    )
    assert len(hashes) == 4

    dates = [r.get_date(hash) for hash in hashes]
    assert dates == sorted(dates)[::-1]

    tags = r.get_tags()
    for tag in tags:
        r.get_date_from_name(tag)

    # Test branches
    conf.branches = [f'{util.git_default_branch()}', 'some-branch']
    branch_commits = {
        f'{util.git_default_branch()}': [
            dvcs.get_hash(f'{util.git_default_branch()}'),
            dvcs.get_hash(f'{util.git_default_branch()}~6'),
        ],
        'some-branch': [dvcs.get_hash('some-branch'), dvcs.get_hash('some-branch~6')],
    }
    _test_branches(conf, branch_commits, require_describe=True)


def test_repo_git_annotated_tag_date(tmpdir):
    tmpdir = str(tmpdir)

    dvcs = tools.generate_test_repo(tmpdir, list(range(5)), dvcs_type='git')

    conf = config.Config()
    conf.project = 'sometest'
    conf.repo = dvcs.path

    r = repo.get_repo(conf)
    d1 = r.get_date('tag1')
    d2 = r.get_date(r.get_hash_from_name('tag1'))
    assert d1 == d2


@pytest.mark.skipif(hglib is None, reason="needs hglib")
def test_repo_hg(tmpdir):
    tmpdir = str(tmpdir)

    dvcs = tools.generate_test_repo(
        tmpdir,
        list(range(10)),
        dvcs_type='hg',
        extra_branches=[('default~4', 'somebranch', [11, 12, 13])],
    )

    conf = config.Config()
    conf.project = join(tmpdir, "repo")
    conf.repo = dvcs.path

    r = repo.get_repo(conf)

    hashes = r.get_hashes_from_range("reverse(default~3::default)")
    assert len(hashes) == 4

    dates = [r.get_date(hash) for hash in hashes]
    assert dates == sorted(dates)[::-1]

    tags = r.get_tags()
    for tag in tags:
        r.get_date_from_name(tag)

    conf.branches = ['default', 'somebranch']
    branch_commits = {
        'default': [dvcs.get_hash('default'), dvcs.get_hash('default~6')],
        'somebranch': [dvcs.get_hash('somebranch'), dvcs.get_hash('somebranch~6')],
    }
    _test_branches(conf, branch_commits)


def test_get_branch_commits(two_branch_repo_case):
    # Test that get_branch_commits() return an ordered list of commits (last
    # first) and follow first parent in case of merge
    dvcs, main, r, conf = two_branch_repo_case
    expected = {
        main: [
            "Revision 6",
            "Revision 4",
            "Merge stable",
            "Revision 3",
            "Revision 1",
        ],
        "stable": [
            "Revision 5",
            "Merge master",
            "Revision 2",
            "Revision 1",
        ],
    }
    for branch in conf.branches:
        commits = [
            dvcs.get_commit_message(commit_hash) for commit_hash in r.get_branch_commits(branch)
        ]
        assert commits == expected[branch]


@pytest.mark.parametrize(
    'dvcs_type',
    ["git", pytest.param("hg", marks=pytest.mark.skipif(hglib is None, reason="needs hglib"))],
)
def test_no_such_name_error(dvcs_type, tmpdir):
    tmpdir = str(tmpdir)
    dvcs = tools.generate_test_repo(tmpdir, values=[0], dvcs_type=dvcs_type)

    conf = config.Config()
    conf.branches = []
    conf.dvcs = dvcs_type
    conf.project = "project"
    conf.repo = dvcs.path

    r = repo.get_repo(conf)

    # Check that NoSuchNameError error gets raised correctly
    assert r.get_hash_from_name(None) == dvcs.get_hash(r._default_branch)
    with pytest.raises(repo.NoSuchNameError):
        r.get_hash_from_name("badbranch")

    if dvcs_type == "git":
        # Corrupted repository/etc should not give NoSuchNameError
        util.long_path_rmtree(join(dvcs.path, ".git"))
        with pytest.raises(Exception) as excinfo:
            r.get_hash_from_name(None)
        assert excinfo.type not in (AssertionError, repo.NoSuchNameError)
    elif dvcs_type == "hg":
        # hglib seems to do some caching, so this doesn't work
        pass
