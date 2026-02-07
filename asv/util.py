# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Various low-level utilities.
"""

import collections
import datetime
import errno
import functools
import json
import math
import multiprocessing
import operator
import os
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time

import json5
from asv_runner.util import _human_time_units, human_float, human_time

WIN = os.name == 'nt'

if not WIN:
    from select import PIPE_BUF


TIMEOUT_RETCODE = -256
terminal_width = shutil.get_terminal_size().columns


class UserError(Exception):
    pass


class ParallelFailure(Exception):
    """
    Custom exception to work around a multiprocessing bug
    https://bugs.python.org/issue9400
    """

    def __new__(cls, message, exc_cls, traceback_str):
        self = Exception.__new__(cls)
        self.message = message
        self.exc_cls = exc_cls
        self.traceback_str = traceback_str
        return self

    def __reduce__(self):
        return (ParallelFailure, (self.message, self.exc_cls, self.traceback_str))

    def __str__(self):
        return "{}: {}\n    {}".format(
            self.exc_cls.__name__, self.message, self.traceback_str.replace("\n", "\n    ")
        )

    def reraise(self):
        if self.exc_cls is UserError:
            raise UserError(self.message)
        else:
            raise self


def human_file_size(size, err=None):
    """
    Returns a human-friendly string representing a file size
    that is 2-4 characters long.

    For example, depending on the number of bytes given, can be one
    of::

        256b
        64k
        1.1G

    Parameters
    ----------
    size : int
        The size of the file (in bytes)

    Returns
    -------
    size : str
        A human-friendly representation of the size of the file
    """
    size = float(size)

    if size < 1:
        size = 0.0

    suffixes = ' kMGTPEH'
    if size == 0:
        num_scale = 0
    else:
        num_scale = math.floor(math.log(size) / math.log(1000))
    if num_scale > 7:
        suffix = '?'
    else:
        suffix = suffixes[num_scale].strip()
    scale = int(math.pow(1000, num_scale))
    value = size / scale

    str_value = human_float(value, 3)

    if err is None:
        return f"{str_value:s}{suffix}"
    else:
        str_err = human_float(err / scale, 1, truncate_small=2)
        return f"{str_value:s}±{str_err:s}{suffix}"


def human_value(value, unit, err=None):
    """
    Formats a value in a given unit in a human friendly way.

    Parameters
    ----------
    value : anything
        The value to format

    unit : str
        The unit the value is in.  Currently understands `seconds` and `bytes`.

    err : float, optional
        Std. error in the value
    """
    if isinstance(value, (int, float)):
        if value != value:
            # nan
            display = "n/a"
        elif unit == 'seconds':
            display = human_time(value, err=err)
        elif unit == 'bytes':
            display = human_file_size(value, err=err)
        else:
            display = json.dumps(value)
            if err is not None:
                display += f"±{err:.2g}"
    elif value is None:
        display = "failed"
    else:
        display = json.dumps(value)

    return display


def parse_human_time(string, base_period='d'):
    """
    Parse a human-specified time period to an integer number of seconds.
    The following format is accepted: <number><suffix>

    Raises a ValueError on parse error.
    """
    units = dict(_human_time_units)
    units[''] = units[base_period]

    suffixes = '|'.join(units.keys())

    try:
        m = re.match(rf'^\s*([0-9.]+)\s*({suffixes})\s*$', string)
        if m is None:
            raise ValueError()
        return float(m.group(1)) * units[m.group(2)]
    except ValueError:
        raise ValueError(f"{string!r} is not a valid time period (valid units: {suffixes})")


def which(filename, paths=None):
    """
    Emulates the UNIX `which` command in Python.

    Raises an OSError if no result is found.
    """
    # Hide traceback from expected exceptions in pytest reports
    __tracebackhide__ = operator.methodcaller('errisinstance', OSError)

    if os.path.sep in filename:
        locations = ['']
    elif paths is not None:
        locations = paths
    else:
        locations = os.environ.get("PATH", "").split(os.pathsep)
        if WIN:
            # On windows, an entry in %PATH% may be quoted
            locations = [
                path[1:-1] if len(path) > 2 and path[0] == path[-1] == '"' else path
                for path in locations
            ]

    if WIN:
        filenames = [filename + ext for ext in ('.exe', '.bat', '.com', '')]
    else:
        filenames = [filename]

    candidates = []
    for location in locations:
        for filename in filenames:
            candidate = os.path.join(location, filename)
            if os.path.isfile(candidate) or os.path.islink(candidate):
                candidates.append(candidate)

    if len(candidates) == 0:
        if paths is None:
            loc_info = 'PATH'
        else:
            loc_info = os.pathsep.join(locations)
        raise OSError(f"Could not find '{filename}' in {loc_info}")

    return candidates[0]


class ProcessError(subprocess.CalledProcessError):
    def __init__(self, args, retcode, stdout, stderr):
        self.args = args
        self.retcode = retcode
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self):
        if self.retcode == TIMEOUT_RETCODE:
            return f"Command '{' '.join(self.args)}' timed out"
        else:
            return "Command '{}' returned non-zero exit status {}".format(
                ' '.join(self.args), self.retcode
            )


def check_call(
    args,
    valid_return_codes=(0,),
    timeout=600,
    dots=True,
    display_error=True,
    shell=False,
    env=None,
    cwd=None,
):
    """
    Runs the given command in a subprocess, raising ProcessError if it
    fails.

    See `check_output` for parameters.
    """
    # Hide traceback from expected exceptions in pytest reports
    __tracebackhide__ = operator.methodcaller('errisinstance', ProcessError)

    check_output(
        args,
        valid_return_codes=valid_return_codes,
        timeout=timeout,
        dots=dots,
        display_error=display_error,
        shell=shell,
        env=env,
        cwd=cwd,
    )


class DebugLogBuffer:
    def __init__(self, log):
        self.buf = []
        self.first = True
        self.linebreak_re = re.compile(b'.*\n')
        self.log = log
        self.lock = threading.Lock()

    def __call__(self, c):
        with self.lock:
            self._process(c)

    def _process(self, c):
        if c is None:
            text = b"".join(self.buf)
            del self.buf[:]
        elif b'\n' in c:
            m = self.linebreak_re.match(c)
            j = m.end()
            self.buf.append(c[:j])
            text = b"".join(self.buf)
            self.buf[:] = [c[j:]]
        else:
            self.buf.append(c)
            return

        text = text.decode('utf-8', 'replace')
        text = text.removesuffix('\n')

        if text:
            if self.first:
                self.log.debug('OUTPUT -------->', continued=True)
                self.first = False
            self.log.debug(text, continued=True)


def check_output(
    args,
    valid_return_codes=(0,),
    timeout=600,
    dots=True,
    display_error=True,
    shell=False,
    return_stderr=False,
    env=None,
    cwd=None,
    redirect_stderr=False,
    return_popen=False,
):
    """
    Runs the given command in a subprocess, raising ProcessError if it
    fails.  Returns stdout as a string on success.

    Parameters
    ----------
    valid_return_codes : list, optional
        A list of return codes to ignore. Defaults to only ignoring zero.
        Setting to None ignores all return codes.

    timeout : number, optional
        Kill the process if it does not produce any output in `timeout`
        seconds. If `None`, there is no timeout.
        Default: 10 min

    dots : bool, optional
        If `True` (default) write a dot to the console to show
        progress as the subprocess outputs content.  May also be
        a callback function to call (with no arguments) to indicate
        progress.

    display_error : bool, optional
        If `True` (default) display the stdout and stderr of the
        subprocess when the subprocess returns an error code.

    shell : bool, optional
        If `True`, run the command through the shell.  Default is
        `False`.

    return_stderr : bool, optional
        If `True`, return both the (stdout, stderr, errcode) as a
        tuple.

    env : dict, optional
        Specify environment variables for the subprocess.

    cwd : str, optional
        Specify the current working directory to use when running the
        process.

    redirect_stderr : bool, optional
        Whether to redirect stderr to stdout. In this case the returned
        ``stderr`` (when return_stderr == True) is an empty string.

    return_popen : bool, optional
        Whether to return immediately after subprocess.Popen.

    Returns
    -------
    stdout, stderr, retcode : when return_stderr == True
    stdout : otherwise
    """
    from .console import log

    # Hide traceback from expected exceptions in pytest reports
    __tracebackhide__ = operator.methodcaller('errisinstance', ProcessError)

    def get_content(header=None):
        content = []
        if header is not None:
            content.append(header)
        if redirect_stderr:
            content.extend(['OUTPUT -------->', stdout[:-1]])
        else:
            content.extend(['STDOUT -------->', stdout[:-1], 'STDERR -------->', stderr[:-1]])
        return '\n'.join(content)

    if isinstance(args, str):
        args = [args]

    log.debug(f"Running '{' '.join(args)}'")

    kwargs = {
        'shell': shell,
        'env': env,
        'cwd': cwd,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
    }
    if redirect_stderr:
        kwargs['stderr'] = subprocess.STDOUT
    if WIN:
        kwargs['close_fds'] = False
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['close_fds'] = True
        posix = getattr(os, 'setpgid', None)
        if posix:
            # Run the subprocess in a separate process group, so that we
            # can kill it and all child processes it spawns e.g. on
            # timeouts. Note that subprocess.Popen will wait until exec()
            # before returning in parent process, so there is no race
            # condition in setting the process group vs. calls to os.killpg
            kwargs['preexec_fn'] = lambda: os.setpgid(0, 0)

    proc = subprocess.Popen(args, **kwargs)

    if return_popen:
        return proc

    last_dot_time = time.time()
    stdout_chunks = []
    stderr_chunks = []
    is_timeout = False

    if log.is_debug_enabled():
        debug_log = DebugLogBuffer(log)
        dots = False
    else:

        def debug_log(c):
            return None

    if WIN:
        start_time = [time.time()]
        dot_start_time = start_time[0]
        is_timeout = False

        def stream_reader(stream, buf):
            try:
                while not is_timeout:
                    c = stream.read(1)
                    if not c:
                        break
                    start_time[0] = time.time()
                    buf.append(c)
                    debug_log(c)
            finally:
                stream.close()

        stdout_reader = threading.Thread(target=stream_reader, args=(proc.stdout, stdout_chunks))
        stdout_reader.daemon = True
        stdout_reader.start()

        all_threads = [stdout_reader]

        if not redirect_stderr:
            stderr_reader = threading.Thread(
                target=stream_reader, args=(proc.stderr, stderr_chunks)
            )
            stderr_reader.daemon = True
            stderr_reader.start()
            all_threads.append(stderr_reader)

        # Wait for reader threads
        threads = list(all_threads)
        while threads:
            thread = threads[0]

            if timeout is None:
                remaining = None
            else:
                remaining = timeout - (time.time() - start_time[0])
                if remaining <= 0:
                    # Timeout; we won't wait for the thread to join here
                    if not is_timeout:
                        is_timeout = True
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    threads.pop(0)
                    continue

            if dots:
                dot_remaining = 0.5 - (time.time() - last_dot_time)
                if dot_remaining <= 0:
                    # Print a dot only if there has been output
                    if dot_start_time != start_time[0]:
                        if dots is True:
                            log.dot()
                        elif dots:
                            dots()
                        dot_start_time = start_time[0]
                        last_dot_time = time.time()
                    dot_remaining = 0.5

                if remaining is None:
                    remaining = dot_remaining
                else:
                    remaining = min(dot_remaining, remaining)

            thread.join(remaining)
            if not thread.is_alive():
                threads.pop(0)

        if is_timeout:
            proc.terminate()

            # Wait a bit for the reader threads, if they're alive
            for thread in all_threads:
                thread.join(0.1)

        # Wait for process to exit
        proc.wait()
    else:
        try:
            if posix and is_main_thread():
                # Forward signals related to Ctrl-Z handling; the child
                # process is in a separate process group so it won't receive
                # these automatically from the terminal
                def sig_forward(signum, frame):
                    _killpg_safe(proc.pid, signum)
                    if signum == signal.SIGTSTP:
                        os.kill(os.getpid(), signal.SIGSTOP)

                signal.signal(signal.SIGTSTP, sig_forward)
                signal.signal(signal.SIGCONT, sig_forward)

            fds = {proc.stdout.fileno(): stdout_chunks}
            if not redirect_stderr:
                fds[proc.stderr.fileno()] = stderr_chunks

            while proc.poll() is None:
                try:
                    if timeout is None:
                        rlist, wlist, xlist = select.select(list(fds.keys()), [], [])
                    else:
                        rlist, wlist, xlist = select.select(list(fds.keys()), [], [], timeout)
                except OSError as err:
                    if err.args[0] == errno.EINTR:
                        # interrupted by signal handler; try again
                        continue
                    raise

                if len(rlist) == 0:
                    # We got a timeout
                    is_timeout = True
                    break
                for f in rlist:
                    output = os.read(f, PIPE_BUF)
                    fds[f].append(output)
                    debug_log(output)
                if dots and time.time() - last_dot_time > 0.5:
                    if dots is True:
                        log.dot()
                    elif dots:
                        dots()
                    last_dot_time = time.time()
        finally:
            if posix and is_main_thread():
                # Restore signal handlers
                signal.signal(signal.SIGTSTP, signal.SIG_DFL)
                signal.signal(signal.SIGCONT, signal.SIG_DFL)

            if proc.returncode is None:
                # Timeout or another exceptional condition occurred, and
                # the program is still running.
                if posix:
                    # Terminate the whole process group
                    _killpg_safe(proc.pid, signal.SIGTERM)

                    for _ in range(10):
                        time.sleep(0.1)
                        if proc.poll() is not None:
                            break
                    else:
                        # Didn't terminate within 1 sec, so kill it
                        _killpg_safe(proc.pid, signal.SIGKILL)
                else:
                    proc.terminate()
                proc.wait()

        proc.stdout.flush()
        if not redirect_stderr:
            proc.stderr.flush()

        stdout_chunks.append(proc.stdout.read())
        if not redirect_stderr:
            stderr_chunks.append(proc.stderr.read())

        proc.stdout.close()
        if not redirect_stderr:
            proc.stderr.close()

    # Flush and disconnect debug log, if any
    debug_log(None)

    def debug_log(c):
        return None

    stdout = b''.join(stdout_chunks)
    stderr = b''.join(stderr_chunks)

    stdout = stdout.decode('utf-8', 'replace')
    stderr = stderr.decode('utf-8', 'replace')

    if is_timeout:
        retcode = TIMEOUT_RETCODE
    else:
        retcode = proc.returncode

    if valid_return_codes is not None and retcode not in valid_return_codes:
        header = f"Error running {' '.join(args)} (exit status {retcode})"
        if display_error:
            if log.is_debug_enabled():
                # Output was already printed
                log.error(header)
            else:
                log.error(get_content(header))
        raise ProcessError(args, retcode, stdout, stderr)

    if return_stderr:
        return (stdout, stderr, retcode)
    else:
        return stdout


def _killpg_safe(pgid, signo):
    """
    Same as os.killpg, but deal with OSX/BSD
    """
    try:
        os.killpg(pgid, signo)
    except OSError as exc:
        if exc.errno == errno.EPERM:
            # OSX/BSD may raise EPERM on killpg if the process group
            # already terminated
            pass
        else:
            raise


def is_main_thread():
    """
    Return True if the current thread is the main thread.
    """
    return threading.current_thread() == threading.main_thread()


def write_json(path, data, api_version=None, compact=False):
    """
    Writes JSON to the given path, including indentation and sorting.

    Parameters
    ----------
    path : str
        File name to write
    data : object
        Data to serialize as JSON
    api_version : int, optional
        API version number
    compact : bool, optional
        Whether to produce compact, non-human readable JSON.
        Disables sorting and indentation.
    """
    path = os.path.abspath(path)

    dirname = long_path(os.path.dirname(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if api_version is not None:
        data = dict(data)
        data['version'] = api_version

    open_kwargs = {}
    open_kwargs['encoding'] = 'utf-8'
    with long_path_open(path, 'w', **open_kwargs) as fd:
        if not compact:
            json.dump(data, fd, indent=4, sort_keys=True)
        else:
            json.dump(data, fd)


def load_json(path, api_version=None, js_comments=False):
    """
    Loads JSON from the given path.

    Parameters
    ----------
    path : str
        File name
    api_version : str or None
        API version identifier
    js_comments : bool, optional
        Whether to allow nonstandard javascript-style comments
        in the file. Note that this slows down the loading
        significantly.
    """
    # Hide traceback from expected exceptions in pytest reports
    __tracebackhide__ = operator.methodcaller('errisinstance', UserError)

    path = os.path.abspath(path)

    open_kwargs = {}
    open_kwargs['encoding'] = 'utf-8'
    with long_path_open(path, 'r', **open_kwargs) as fd:
        content = fd.read()

    if js_comments:
        # strips comments out
        data = json5.loads(content)
    else:
        try:
            data = json.loads(content)
        except ValueError as err:
            raise UserError(f"Error parsing JSON in file '{path}': {err}")

    if api_version is not None:
        if 'version' in data:
            if data['version'] < api_version:
                raise UserError(
                    f"{path} is stored in an old file format.  Run `asv update` to update it."
                )
            elif data['version'] > api_version:
                raise UserError(
                    f"{path} is stored in a format that is newer than "
                    "what this version of asv understands.  Update "
                    "asv to use this file."
                )

            del data['version']
        else:
            raise UserError(f"No version specified in {path}.")

    return data


def update_json(cls, path, api_version, compact=False):
    """
    Perform JSON file format updates.

    Parameters
    ----------
    cls : object
        Object containing methods update_to_X which updates
        the given JSON tree from version X-1 to X.

    path : str
        Path to JSON file

    api_version : int
        The current API version
    """
    # Hide traceback from expected exceptions in pytest reports
    __tracebackhide__ = operator.methodcaller('errisinstance', UserError)

    d = load_json(path)
    if 'version' not in d:
        raise UserError(f"No version specified in {path}.")

    if d['version'] < api_version:
        for x in range(d['version'] + 1, api_version + 1):
            d = getattr(cls, f'update_to_{x}', lambda x: x)(d)
        write_json(path, d, api_version, compact=compact)
    elif d['version'] > api_version:
        raise UserError(
            f"{path} is stored in a format that is newer than "
            "what this version of asv understands. "
            "Upgrade asv in order to use or add to "
            "these results."
        )


def iter_subclasses(cls):
    """
    Returns all subclasses of a class.
    """
    for x in cls.__subclasses__():
        yield x
        for y in iter_subclasses(x):
            yield y


def hash_equal(a, b):
    """
    Returns `True` if a and b represent the same commit hash.
    """
    min_len = min(len(a), len(b))
    return a.lower()[:min_len] == b.lower()[:min_len]


def format_text_table(rows, num_headers=0, top_header_span_start=0, top_header_text=None):
    """
    Format rows in as a reStructuredText table, in the vein of:

    .. code-block::

           ========== ========== ==========
           --         top header text, span start 1
           ---------- ---------------------
            row0col0     r0c1      r0c2
           ========== ========== ==========
            row1col0     r1c1      r1c2
            row2col0     r2c1      r2c2
           ========== ========== ==========

    """

    # Format content
    text_rows = [[f"{item}".replace("\n", " ") for item in row] for row in rows]

    # Ensure same number of items on all rows
    num_items = max(len(row) for row in text_rows)
    for row in text_rows:
        row.extend([''] * (num_items - len(row)))

    # Determine widths
    col_widths = [max(len(row[j]) for row in text_rows) + 2 for j in range(num_items)]

    # Pad content
    text_rows = [[item.center(w) for w, item in zip(col_widths, row)] for row in text_rows]

    # Generate result
    headers = [" ".join(row) for row in text_rows[:num_headers]]
    content = [" ".join(row) for row in text_rows[num_headers:]]
    separator = " ".join("-" * w for w in col_widths)

    result = []
    if top_header_text is not None:
        left_span = "-".join("-" * w for w in col_widths[:top_header_span_start])
        right_span = "-".join("-" * w for w in col_widths[top_header_span_start:])
        if left_span and right_span:
            result += ["--" + " " * (len(left_span) - 1) + top_header_text.center(len(right_span))]
            result += [" ".join([left_span, right_span])]
        else:
            result += [top_header_text.center(len(separator))]
            result += ["-".join([left_span, right_span])]
        result += headers
        result += [separator.replace("-", "=")]
    elif headers:
        result += headers
        result += [separator]
    result += content
    result = [separator.replace("-", "=")] + result
    result += [separator.replace("-", "=")]
    return "\n".join(result)


def _datetime_to_timestamp(dt, divisor):
    delta = dt - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    microseconds = (delta.days * 86400 + delta.seconds) * 10**6 + delta.microseconds
    value, remainder = divmod(microseconds, divisor)
    if remainder >= divisor // 2:
        value += 1
    return value


def datetime_to_js_timestamp(dt):
    """
    Convert a Python datetime object to a JavaScript timestamp.
    """
    return _datetime_to_timestamp(dt, 10**3)


def js_timestamp_to_datetime(ts):
    """
    Convert a JavaScript timestamp to a Python datetime object.
    """
    return datetime.datetime.fromtimestamp(ts / 1000)


def is_nan(x):
    """
    Returns `True` if x is a NaN value.
    """
    if isinstance(x, float):
        return x != x
    return False


def is_na(value):
    """
    Return True if value is None or NaN
    """
    return value is None or is_nan(value)


def mean_na(values):
    """
    Take a mean, with the understanding that None and NaN stand for
    missing data.
    """
    values = [x for x in values if not is_na(x)]
    if values:
        return sum(values) / len(values)
    else:
        return None


def geom_mean_na(values):
    """
    Compute geometric mean, with the understanding that None and NaN
    stand for missing data.
    """
    values = [x for x in values if not is_na(x)]
    if values:
        exponent = 1 / len(values)
        prod = 1.0
        acc = 0
        for x in values:
            prod *= abs(x) ** exponent
            acc += x
        return prod if acc >= 0 else -prod
    else:
        return None


if not WIN:
    long_path_open = open
    long_path_rmtree = shutil.rmtree

    def long_path(path):
        return path
else:

    def long_path(path):
        if path.startswith("\\\\"):
            return path
        return "\\\\?\\" + os.path.abspath(path)

    def _remove_readonly(func, path, exc_info):
        """Try harder to remove files on Windows"""

        if isinstance(exc_info[1], OSError) and exc_info[1].errno == errno.EACCES:
            # Clear read-only flag and try again
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                func(path)
                return
            except OSError:
                pass

        # Reraise original error
        raise exc_info[1]

    def long_path_open(filename, *a, **kw):
        return open(long_path(filename), *a, **kw)

    def long_path_rmtree(path, ignore_errors=False):
        if ignore_errors:
            onerror = None
        else:
            onerror = _remove_readonly
        shutil.rmtree(long_path(path), ignore_errors=ignore_errors, onerror=onerror)


def sanitize_filename(filename):
    """
    Replace characters to make a string safe to use in file names.

    This is not a 1-to-1 mapping.

    The implementation needs to match www/asv.js:escape_graph_parameter
    """
    if not isinstance(filename, str):
        filename = filename.decode(sys.getfilesystemencoding())

    # ntfs & ext3
    filename = re.sub('[<>:"/\\^|?*\x00-\x1f]', '_', filename)

    # ntfs
    forbidden = [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]
    if filename.upper() in forbidden:
        filename = filename + "_"

    return filename


def truncate_float_list(item, digits=5):
    """
    Truncate floating-point numbers (in a possibly nested list)
    to given significant digits, for a shorter base-10
    representation.
    """
    if isinstance(item, float):
        fmt = f'{{:.{digits - 1}e}}'
        return float(fmt.format(item))
    elif isinstance(item, list):
        return [truncate_float_list(x, digits) for x in item]
    else:
        return item


_global_locks = {}


def _init_global_locks(lock_dict, env):
    """Initialize global locks in a new multiprocessing process
    Also inherit the base environment even if using a forkserver"""
    _global_locks.update(lock_dict)
    os.environ.update(env)


def get_multiprocessing_pool(parallel=None):
    """Create a multiprocessing.Pool, managing global locks properly"""
    env = os.environ.copy()
    return multiprocessing.Pool(parallel, initializer=_init_global_locks, initargs=(_global_locks, env))


def git_default_branch():
    try:
        # Local name gets precedence
        default_branch = check_output(
            [which('git'), 'config', 'init.defaultBranch'], display_error=False
        ).strip()
    except ProcessError:
        # Check global
        try:
            default_branch = check_output(
                [which('git'), 'config', '--global', 'init.defaultBranch'], display_error=False
            ).strip()
        except ProcessError:
            # Check system
            try:
                default_branch = check_output(
                    [which('git'), 'config', '--system', 'init.defaultBranch'], display_error=False
                ).strip()
            except ProcessError:
                # Default to master when global and system are not set
                default_branch = 'master'
    return default_branch


