import time
from typing import Callable
from torch import nn

def timeit(number: int = 1000) -> Callable:
    """
    A deorator with an argument `number` for how many times it should run.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> float:
            t0 = time.perf_counter()
            for _ in range(number):
                _ = func(*args, **kwargs)
            exec_t = time.perf_counter() - t0
            return exec_t
        return wrapper
    return decorator

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())