import time


class Timer:
    def __init__(self, label: str = "timer", print_elapsed=True):
        self.label = label
        self.print_elapsed = print_elapsed

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        if self.print_elapsed:
            print(f"{self.label}: {self.elapsed:.6f}s")
