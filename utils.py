import multiprocessing as mp
from functools import partial
import time


def run_function_on_list(func, list_data, cores=1, **kwargs):
    fn = partial(func, **kwargs)
    if cores == 1:
        out = [fn(val) for val in list_data]
    else:
        pool = mp.Pool(cores)

        out = pool.map(fn, list_data)
        pool.close()

    return out


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__}, done in {round(te - ts, 2)}.')
        return result
    return timed