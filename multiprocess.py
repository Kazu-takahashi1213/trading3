__all__ = ['lin_parts', 'nested_parts', 'mp_pandas_obj', 'process_jobs_', 'report_progress', 'process_jobs',
           'expand_call', '#', '#', '#', '#']

# Cell

# Linear Partitions [20.4.1]
import pandas as pd
import numpy as np
import time
import sys
import os


def lin_parts(num_atoms, num_threads):
    # partition of atoms with a single loop
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, upper_triang=False):
    # partition of atoms with an inner loop
    parts, num_threads_ = [0], min(num_threads, num_atoms)
    for num in range(num_threads_):
        part = 1 + 4 * (
            parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_
        )
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def mp_pandas_obj(func, pd_obj, num_threads=32, mp_batches=1, lin_mols=True, **kargs):
    """
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pd_obj[0]: Name of argument used to pass the molecule
    + pd_obj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)
    """
    import pandas as pd

    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1] : parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0


# =======================================================
# single-thread execution for debugging [20.8]
def process_jobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)
    return out


# =======================================================
# Example of async call to multiprocessing lib [20.9]
import multiprocessing as mp
import datetime as dt

# ________________________________
def report_progress(job_num, num_jobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = (
        time_stamp
        + " "
        + str(round(msg[0] * 100, 2))
        + "% "
        + task
        + " done after "
        + str(round(msg[1], 2))
        + " minutes. Remaining "
        + str(round(msg[2], 2))
        + " minutes."
    )
    if job_num < num_jobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")
    return


# ________________________________
def process_jobs(jobs, task=None, num_threads=36):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]["func"].__name__
    pool = mp.Pool(processes=num_threads)
    outputs, out, time0 = pool.imap_unordered(expand_call, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


# =======================================================
# Unwrapping the Callback [20.10]
def expand_call(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func = kargs["func"]
    del kargs["func"]
    out = func(**kargs)
    return out


# =======================================================
# Pickle Unpickling Objects [20.11]
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


# ________________________________
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


# ________________________________