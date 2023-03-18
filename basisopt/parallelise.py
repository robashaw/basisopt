from typing import Any, Callable

from . import api
from .util import bo_logger

if api._PARALLEL:
    from distributed import Client, LocalCluster, wait
else:
    bo_logger.warning("Dask not installed, parallelisation not available")


def chunk(x: list[Any], n_chunks: int) -> list[list[Any]]:
    """Chunks an array into roughly equal-sized subarrays

    Args:
         x - array of values of length L
         n_chunks - number of chunks to split into

    Returns:
         a list of n_chunks arrays of length L//n_chunks
         or (L//n_chunks)+1
    """
    chunk_size = len(x) // n_chunks
    remainder = len(x) % n_chunks
    chunk_list = [chunk_size] * n_chunks
    for i in range(remainder):
        chunk_list[i] += 1
    new_x = []
    ctr = 0
    for i in range(n_chunks):
        new_x.append(x[ctr : ctr + chunk_list[i]])
        ctr += chunk_list[i]
    return new_x


def distribute(
    n_proc: int, func: Callable[[list[Any], dict], Any], x: list[Any], **kwargs
) -> list[Any]:
    """Distributes a function over a desired no. of procs
    using the distributed library.

    Args:
         n_proc - the number of processes to start
         func - the function to call, with signature (x, **kwargs)
         x  - the array of values to distribute over
         kwargs - the named arguments accepted by func
    Returns:
         a list of results ordered by process ID
    """
    n_chunks = len(x) // n_proc
    if len(x) % n_proc > 0:
        n_chunks += 1
    new_x = chunk(x, n_chunks)

    all_results = []
    for i in range(n_chunks):
        cluster = LocalCluster(n_workers=len(new_x[i]), processes=True)
        client = Client(cluster)
        ens = client.map(func, new_x[i], **kwargs)
        wait(ens)
        results = [e.result() for e in ens]
        all_results.append(results)
        client.close()
        cluster.close()

    return [item for sublist in all_results for item in sublist]
