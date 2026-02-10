import os
import shutil
import textwrap
from os.path import abspath, dirname, join

import pytest
import selenium

from asv import config, repo, results, step_detect, util
from asv.repo import get_repo
from asv.step_detect import L1Dist

from . import tools
from .test_web import _rebuild_basic_html
from .tools import (
    WAIT_TIME,
    locked_cache_dir,
    run_asv_with_conf,
)

try:
    import hglib
except ImportError:
    hglib = None

try:
    from asv._rangemedian_numba import RangeMedian as _RangeMedian

    HAVE_RANGEMEDIAN = True
except ImportError:
    HAVE_RANGEMEDIAN = False

DUMMY_VALUES = (
    (6, 1),
    (6, 6),
    (6, 6),
)


def pytest_addoption(parser):
    parser.addoption(
        "--webdriver",
        action="store",
        default="None",
        help=(
            "Selenium WebDriver interface to use for running the test. "
            "Choices: None, PhantomJS, Chrome, Firefox, ChromeHeadless, "
            "FirefoxHeadless. Alternatively, it can be arbitrary Python code "
            "with a return statement with selenium.webdriver object, for "
            "example 'return Chrome()'"
        ),
    )
    parser.addoption("--runflaky", action="store_true", default=False, help="run flaky tests")


@pytest.fixture(
    params=[
        "git",
        pytest.param("hg", marks=pytest.mark.skipif(hglib is None, reason="needs hglib")),
    ]
)
def two_branch_repo_case(request, tmpdir):
    r"""
    This test ensure we follow the first parent in case of merges

    The revision graph looks like this:

        @  Revision 6 (default)
        |
        | o  Revision 5 (stable)
        | |
        | o  Merge master
        |/|
        o |  Revision 4
        | |
        o |  Merge stable
        |\|
        o |  Revision 3
        | |
        | o  Revision 2
        |/
        o  Revision 1

    """
    dvcs_type = request.param
    tmpdir = str(tmpdir)
    if dvcs_type == "git":
        master = f"{util.git_default_branch()}"
    elif dvcs_type == "hg":
        master = "default"
    dvcs = tools.generate_repo_from_ops(
        tmpdir,
        dvcs_type,
        [
            ("commit", 1),
            ("checkout", "stable", master),
            ("commit", 2),
            ("checkout", master),
            ("commit", 3),
            ("merge", "stable"),
            ("commit", 4),
            ("checkout", "stable"),
            ("merge", master, "Merge master"),
            ("commit", 5),
            ("checkout", master),
            ("commit", 6),
        ],
    )

    conf = config.Config()
    conf.branches = [master, "stable"]
    conf.repo = dvcs.path
    conf.project = join(tmpdir, "repo")
    r = repo.get_repo(conf)
    return dvcs, master, r, conf


@pytest.fixture(scope="session")
def example_results(request):
    with locked_cache_dir(request.config, "example-results") as cache_dir:
        src = abspath(join(dirname(__file__), 'example_results'))
        dst = abspath(join(cache_dir, 'results'))

        if os.path.isdir(dst):
            return dst

        shutil.copytree(src, dst)

        # Convert old-format result files to current format
        for root, dirs, files in os.walk(dst):
            for fn in files:
                if fn.endswith('.json') and fn != 'benchmarks.json' and fn != 'machine.json':
                    try:
                        results.Results.update(os.path.join(root, fn))
                    except util.UserError:
                        pass  # Skip malformed test files

        return dst


@pytest.fixture(scope="session")
def browser(request, pytestconfig):
    """
    Fixture for Selenium WebDriver browser interface
    """
    driver_str = pytestconfig.getoption('webdriver')

    if driver_str == "None":
        pytest.skip("No webdriver selected for tests (use --webdriver).")

    # Evaluate the options
    def FirefoxHeadless():
        options = selenium.webdriver.FirefoxOptions()
        options.add_argument("-headless")
        return selenium.webdriver.Firefox(options=options)

    def ChromeHeadless():
        options = selenium.webdriver.ChromeOptions()
        options.add_argument('headless')
        return selenium.webdriver.Chrome(options=options)

    ns = {}
    exec("import selenium.webdriver", ns)
    exec("from selenium.webdriver import *", ns)
    ns['FirefoxHeadless'] = FirefoxHeadless
    ns['ChromeHeadless'] = ChromeHeadless

    create_driver = ns.get(driver_str, None)
    if create_driver is None:
        src = "def create_driver():\n"
        src += textwrap.indent(driver_str, "    ")
        exec(src, ns)
        create_driver = ns['create_driver']

    # Create the browser
    browser = create_driver()

    # Set timeouts
    browser.set_page_load_timeout(WAIT_TIME)
    browser.set_script_timeout(WAIT_TIME)

    # Clean up on fixture finalization
    def fin():
        browser.quit()

    request.addfinalizer(fin)

    # Set default time to wait for AJAX requests to complete
    browser.implicitly_wait(WAIT_TIME)

    return browser


@pytest.fixture(scope="session")
def basic_html(request):
    with locked_cache_dir(request.config, "asv-test_web-basic_html", timeout=900) as cache_dir:
        tmpdir = join(str(cache_dir), 'cached')
        html_dir, dvcs = _rebuild_basic_html(tmpdir)
        return html_dir, dvcs


@pytest.fixture(
    params=[
        "git",
        pytest.param("hg", marks=pytest.mark.skipif(hglib is None, reason="needs hglib")),
    ]
)
def generate_result_dir(request, tmpdir):
    tmpdir = str(tmpdir)
    dvcs_type = request.param

    def _generate_result_dir(values, commits_without_result=None):
        dvcs = tools.generate_repo_from_ops(
            tmpdir, dvcs_type, [("commit", i) for i in range(len(values))]
        )
        commits = list(reversed(dvcs.get_branch_hashes()))
        commit_values = {}
        commits_without_result = [commits[i] for i in commits_without_result or []]
        for commit, value in zip(commits, values):
            if commit not in commits_without_result:
                commit_values[commit] = value
        conf = tools.generate_result_dir(tmpdir, dvcs, commit_values)
        repo = get_repo(conf)
        return conf, repo, commits

    return _generate_result_dir


@pytest.fixture
def show_fixture(tmpdir, example_results):
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)

    conf = config.Config.from_json(
        {
            'results_dir': example_results,
            'repo': tools.generate_test_repo(tmpdir).path,
            'project': 'asv',
        }
    )

    return conf


@pytest.fixture(
    params=[
        "python",
        pytest.param(
            "rangemedian",
            marks=pytest.mark.skipif(
                not HAVE_RANGEMEDIAN, reason="asv._rangemedian_numba required"
            ),
        ),
    ]
)
def use_rangemedian(request):
    if request.param == "rangemedian":
        assert isinstance(step_detect.get_mu_dist([0], [1]), _RangeMedian)
        return True
    else:
        step_detect._rangemedian_available = False

        def restore():
            if HAVE_RANGEMEDIAN:
                step_detect._rangemedian_available = True

        request.addfinalizer(restore)

        assert isinstance(step_detect.get_mu_dist([0], [1]), L1Dist)
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "flaky_pypy: Tests that are flaky on pypy.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runflaky"):
        # --runflaky given in cli: do not skip flaky tests
        return
    skip_flaky = pytest.mark.skip(reason="need --runflaky option to run")
    for item in items:
        if "flaky" in item.keywords:
            item.add_marker(skip_flaky)
