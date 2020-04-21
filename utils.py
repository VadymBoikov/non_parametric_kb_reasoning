import multiprocessing as mp
from functools import partial


def run_function_on_list(func, list_data, single_core=False, **kwargs):
    fn = partial(func, **kwargs)
    if single_core:
        out = [fn(val) for val in list_data]
    else:
        num_workers = mp.cpu_count()
        pool = mp.Pool(num_workers)

        out = pool.map(fn, list_data)
        pool.close()

    return out