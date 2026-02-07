Using airspeed velocity
=======================

**airspeed velocity** analyzes stored benchmark results and generates
interactive static websites for exploring performance trends.

The benchmark results are stored as JSON files in the ``results``
directory, organized by machine name.  Results from multiple machines
can be combined by placing them alongside each other in the
``results`` directory.  **airspeed velocity** is designed from the
ground up to handle missing data where certain benchmarks have yet to
be performed.

You can interact with **airspeed velocity** through the ``asv``
command.  Like ``git``, the ``asv`` command has a number of
"subcommands" for performing various actions on your benchmarking
project.

Configuration
-------------

A minimal ``asv.conf.json`` configuration file is needed.  The key
settings for analysis are:

- ``version``: Version of the ``asv.conf.json`` spec.  Currently only ``1``.

- ``project``: The Python package name of the project being benchmarked.

- ``project_url``: The project's homepage.

- ``repo``: The URL or path to the DVCS repository for the project.

- ``show_commit_url``: The base of URLs used to display commits for
  the project.  This allows users to click on a commit in the web
  interface and have it display the contents of that commit.  For a
  github project, the URL is of the form
  ``https://github.com/$OWNER/$REPO/commit/``.

- ``results_dir``: The directory containing the benchmark results
  (defaults to ``"results"``).

- ``html_dir``: The directory to output the generated website
  (defaults to ``"html"``).

There is also a :ref:`conf-reference` with more details on all
available settings.

.. _viewing-results:

Viewing the results
-------------------

You can use the :ref:`asv show <cmd-asv-show>` command to display
results from previous runs on the command line::

    $ asv show main
    Commit: 4238c44d <main>

    benchmarks.MemSuite.mem_list [mymachine/virtualenv-py3.7]
      2.42k
      started: 2018-08-19 18:46:47, duration: 1.00s

    benchmarks.TimeSuite.time_iterkeys [mymachine/virtualenv-py3.7]
      11.1±0.06μs
      started: 2018-08-19 18:46:47, duration: 1.00s

    ...

To collate a set of results into a viewable website, run::

    asv publish

This will put a tree of files in the ``html`` directory.  This website
can not be viewed directly from the local filesystem, since web
browsers do not support AJAX requests to the local filesystem.
Instead, **airspeed velocity** provides a simple static webserver that
can be used to preview the website.  Just run::

    asv preview

and open the URL that is displayed at the console.  Press Ctrl+C to
stop serving.

|screenshot| |screenshot2|

.. |screenshot| image:: screenshot-grid.png
   :width: 45%

.. |screenshot2| image:: screenshot-bench.png
   :width: 45%

To share the website on the open internet, simply put the files in the
``html`` directory on any webserver that can serve static content.  Github Pages
works quite well, for example.  For using Github Pages, ``asv``
includes the convenience command ``asv gh-pages`` to put the results
to the ``gh-pages`` branch and push them to Github. See :ref:`asv gh-pages
--help <cmd-asv-gh-pages>` for details.

Managing the results database
-----------------------------

The ``asv rm`` command can be used to remove benchmarks from the
database.  The command takes an arbitrary number of ``key=value``
entries that are "and"ed together to determine which benchmarks to
remove.

The keys may be one of:

- ``benchmark``: A benchmark name

- ``python``: The version of python

- ``commit_hash``: The commit hash

- machine-related: ``machine``, ``arch``, ``cpu``, ``os``, ``ram``

- environment-related: a name of a dependency, e.g. ``numpy``

The values are glob patterns, as supported by the Python standard
library module ``fnmatch``.  So, for example, to remove all benchmarks
in the ``time_units`` module::

    asv rm "benchmark=time_units.*"

Note the double quotes around the entry to prevent the shell from
expanding the ``*`` itself.

The ``asv rm`` command will prompt before performing any operations.
Passing the ``-y`` option will skip the prompt.

Here is a more complex example, to remove all of the benchmarks on
Python 3.7 and the machine named ``giraffe``::

    asv rm python=3.7 machine=giraffe

Regression detection
--------------------

**airspeed velocity** detects statistically significant decreases of
performance automatically based on the available data when you run
``asv publish``. The results can be inspected via the web interface,
clicking the "Regressions" tab on the web site.  The results include
links to each benchmark graph deemed to contain a decrease in
performance, the commits where the regressions were estimated to
occur, and other potentially useful information.

.. image:: screenshot-regressions.png
   :width: 60%

.. _comparing:

Comparing the benchmarking results for two revisions
----------------------------------------------------

In some cases, you may want to directly compare the results for two specific
revisions of the project. You can do so with the ``compare`` command::

    $ asv compare v0.1 v0.2
    All benchmarks:

           before           after         ratio
         [3bfda9c6]       [bf719488]
         <v0.1>           <v0.2>
                40.4m            40.4m     1.00  benchmarks.MemSuite.mem_list [amulet.localdomain/virtualenv-py3.7-numpy]
               failed            35.2m      n/a  benchmarks.MemSuite.mem_list [amulet.localdomain/virtualenv-py3.12-numpy]
          11.5±0.08μs         11.0±0μs     0.96  benchmarks.TimeSuite.time_iterkeys [amulet.localdomain/virtualenv-py3.7-numpy]
               failed           failed      n/a  benchmarks.TimeSuite.time_iterkeys [amulet.localdomain/virtualenv-py3.12-numpy]
             11.5±1μs      11.2±0.02μs     0.97  benchmarks.TimeSuite.time_keys [amulet.localdomain/virtualenv-py3.7-numpy]
               failed      8.40±0.02μs      n/a  benchmarks.TimeSuite.time_keys [amulet.localdomain/virtualenv-py3.12-numpy]
          34.6±0.09μs      32.9±0.01μs     0.95  benchmarks.TimeSuite.time_range [amulet.localdomain/virtualenv-py3.7-numpy]
               failed      35.6±0.05μs      n/a  benchmarks.TimeSuite.time_range [amulet.localdomain/virtualenv-py3.12-numpy]
           31.6±0.1μs      30.2±0.02μs     0.95  benchmarks.TimeSuite.time_xrange [amulet.localdomain/virtualenv-py3.7-numpy]
               failed           failed      n/a  benchmarks.TimeSuite.time_xrange [amulet.localdomain/virtualenv-py3.12-numpy]

This will show the times for each benchmark for the first and second
revision, and the ratio of the second to the first. In addition, the
benchmarks will be color coded green and red if the benchmark improves
or worsens more than a certain threshold factor, which defaults to 1.1
(that is, benchmarks that improve by more than 10% or worsen by 10%
are color coded). The threshold can be set with the
``--factor=value`` option. Finally, the benchmarks can be split
into ones that have improved, stayed the same, and worsened, using the
same threshold using the ``--split`` option.
See :ref:`cmd-asv-compare` for more.

ASV also has a compare column which can be used to get a quick (and colorless)
visual summary of benchmark results. This consists of a single ``mark`` where
each of its symbolic states can be understood as:

.. list-table:: ASV Change column states, ``before`` is the first commit ID, ``after`` is the second commit ID
   :widths: 15 25 15 15 15
   :header-rows: 1

   * - Change
     - Color
     - Description
     - After
     - Before
   * - ``x``
     - Light Gray
     - Not comparable
     -
     -
   * - ``!``
     - Red
     - Introduced a failure
     - Failed
     - Succeeded
   * - ``*``
     - Green
     - Fixed failure
     - Succeeded
     - Failed
   * -
     -
     - Both failed or either was skipped or no significant change
     -
     -
   * - ``-``
     - Green
     - Relative improvement
     - Better
     - Worse
   * - ``+``
     - Red
     - Relatively worse
     - Worse
     - Better

Additionally, statistically insignificant results have ``~`` in the ratio column as well.
