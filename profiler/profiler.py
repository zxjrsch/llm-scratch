import cProfile
import gc
import pstats
import time
import traceback
from contextlib import contextmanager


class Profiler:
    def __init__(self):
        self.prof = cProfile.Profile()

    def __enter__(self):
        self.prof.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()

        if exc_type is not None:
            print(f'Error encountered: {exc_val}')

        self.prof.disable()
        ps = pstats.Stats(self.prof, stream=None).sort_stats('cumulative')
        ps.print_stats()

@contextmanager
def timer():
    try:
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
    finally:
        print(t1-t0)


if __name__ == '__main__':
    
    with Profiler():
        print('hello profiler')