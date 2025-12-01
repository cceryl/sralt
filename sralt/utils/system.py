import os
import time
import logging
import hashlib
import functools
from typing import Any, Callable

import torch
import pickle


def get_device(allow_share: bool = False, allow_cpu: bool = False) -> torch.device:
    """
    Get the available device. Try to use the GPU with the most free memory.

    :param allow_share: Whether to allow sharing GPU with other running processes.
    :param allow_cpu: Whether to allow using CPU if no suitable GPU is found.
    :return: The selected torch device.
    """
    if torch.cuda.is_initialized():
        error_message = (
            'CUDA has already been initialized. '
            'Set CUDA_DEVICE_ORDER=PCI_BUS_ID before initializing CUDA, '
            'or avoid initializing CUDA before calling get_device().'
        )
        assert os.environ.get('CUDA_DEVICE_ORDER', '') == 'PCI_BUS_ID', error_message
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    if not allow_share:
        devices = [device for device in devices if 'no processes are running' in torch.cuda.list_gpu_processes(device)]

    if devices:
        device = max(devices, key=lambda device: torch.cuda.mem_get_info(device)[0])
        logging.info(f'Using GPU: {torch.cuda.get_device_name(device)} (index {device.index}).')
        return device

    if allow_cpu:
        logging.warning('No available GPU found. Using CPU.')
        return torch.device('cpu')

    raise RuntimeError('No available GPU found.')


def timer(log_format: str = 'Function {func.__name__} took {elapsed:.2f} seconds.') -> Callable:
    """
    A decorator to measure the execution time of a function.
    Available placeholders are 'func', 'args', 'kwargs', 'start', 'end', and 'elapsed'.

    :param log_format: The format string for logging.
    :return: The decorated function with timer.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.info(log_format.format(
                func=func, args=args, kwargs=kwargs,
                start=start, end=end, elapsed=end-start
            ))
            return result
        return wrapper
    return decorator


def disk_cache(cache_dir: str = 'cache') -> Callable:
    """
    A decorator to cache the result of a function to disk.

    :param cache_dir: Directory where cached results will be stored.
    :return: The decorated function with caching capability.
    """
    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key_raw = (func.__name__, repr(args), repr(sorted(kwargs.items())))
            key_bytes = pickle.dumps(key_raw)
            key_hash = hashlib.md5(key_bytes).hexdigest()
            cache_path = os.path.join(cache_dir, f'{key_hash}.pkl')

            if os.path.exists(cache_path):
                logging.info(f'Loading cached result of function {func.__name__} from {cache_path}.')
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            logging.info(f'Computing result of function {func.__name__} and caching to {cache_path}.')
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            return result
        return wrapper
    return decorator
