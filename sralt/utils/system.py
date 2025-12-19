import os
import time
import types
import typing
import logging
import hashlib
import inspect
import functools
from uuid import UUID
from typing import Any, Callable

import torch
import pynvml
import pickle
from cuda import cuda
from torch.utils.tensorboard import SummaryWriter


from .model import calculate_tensor_fingerprint


def get_object_repr(obj: Any, visited: set[int] | None = None, max_depth: int = 10) -> str:
    """
    Get a string representation of an object for caching purposes.

    :param obj: The object to represent.
    :param visited: Set of visited object ids to avoid infinite recursion.
    :param max_depth: Maximum recursion depth.
    :return: The string representation of the object.
    """
    if max_depth <= 0:
        raise ValueError('Maximum recursion depth exceeded in get_object_repr.')
    max_depth -= 1

    if visited is None:
        visited = set()
    if id(obj) in visited:
        return '...'
    if isinstance(obj, (list, tuple, set, dict, frozenset, types.FunctionType)):
        visited.add(id(obj))

    if isinstance(obj, types.FunctionType):
        return inspect.getsource(obj)

    if isinstance(obj, SummaryWriter):
        return 'SummaryWriter'

    if isinstance(obj, torch.Tensor):
        return f'torch.Tensor(fingerprint={calculate_tensor_fingerprint(obj)})'

    if isinstance(obj, (list, tuple, set, frozenset)):
        elements = [get_object_repr(e, visited, max_depth) for e in obj]
        if isinstance(obj, list):
            return f'[{", ".join(elements)}]'
        elif isinstance(obj, tuple):
            return f'({", ".join(elements)})'
        elif isinstance(obj, set):
            return f'{{{", ".join(elements)}}}'
        else:
            return f'frozenset({{{", ".join(elements)}}})'

    if isinstance(obj, dict):
        items = [
            f'{get_object_repr(k, visited, max_depth)}: {get_object_repr(v, visited, max_depth)}' for k, v in obj.items()
        ]
        return f'{{{", ".join(items)}}}'

    repr_str = repr(obj)
    if ('at 0x' in repr_str):
        raise NotImplementedError(f'Cannot cache function with unrepresentable arguments: {repr_str}')

    return repr_str


def disk_cache(
    cache_dir: str,
    hit_log_format: str = 'Loading cached result of function "{__cache__.func.__name__}" from "{__cache__.path}".',
    miss_log_format: str = 'Computing result of function "{__cache__.func.__name__}" and caching to "{__cache__.path}".'
) -> Callable:
    """
    A decorator to cache the result of a function to disk.
    Available placeholders are arguments of the decorated function and 'func', 'hash', 'path' in '__cache__' namespace.
    Note that the decorated function should not have a parameter named '__cache__'.

    :param cache_dir: Directory where cached results will be stored.
    :param hit_log_format: Log format string for cache hits.
    :param miss_log_format: Log format string for cache misses.
    :return: The decorated function with caching capability.
    """
    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            cache_dir_formatted = cache_dir.format(**bound_args.arguments)
            os.makedirs(cache_dir_formatted, exist_ok=True)

            key = (func.__name__, {k: get_object_repr(v) for k, v in bound_args.arguments.items()})
            key_bytes = pickle.dumps(key)
            key_hash = hashlib.md5(key_bytes).hexdigest()
            cache_path = os.path.join(cache_dir_formatted, f'{key_hash}.pkl')
            logging.debug(f'Key for function "{func.__name__}": {key}')

            cache_namespace = types.SimpleNamespace()
            cache_namespace.func = func
            cache_namespace.hash = key_hash
            cache_namespace.path = cache_path

            if os.path.exists(cache_path):
                logging.info(hit_log_format.format(**bound_args.arguments, __cache__=cache_namespace))
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)

                if isinstance(cache_data, dict) and cache_data.get('key') == key:
                    return cache_data['result']

                logging.warning(f'Hash collision detected for cache file "{cache_path}". Recomputing the result.')

            logging.info(miss_log_format.format(**bound_args.arguments, __cache__=cache_namespace))
            result = func(*args, **kwargs)

            with open(cache_path, 'wb') as f:
                pickle.dump({'key': key, 'result': result}, f)
            return result

        return wrapper

    return decorator


def get_device(allow_share: bool = False, allow_cpu: bool = False) -> torch.device:
    """
    Get the available device. Try to use the GPU with the most free memory.

    :param allow_share: Whether to allow sharing GPU with other running processes.
    :param allow_cpu: Whether to allow using CPU if no suitable GPU is found.
    :return: The selected torch device.
    """
    @typing.no_type_check
    def safe_cuda_call(func: Callable, *args, **kwargs):
        code, result = func(*args, **kwargs)
        if code != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'CUDA error code: {code}.')
        return result

    @typing.no_type_check
    def get_uuid_to_cuda_id() -> dict[str, int]:
        cuda.cuInit(0)
        uuid_to_cuda_id: dict[str, int] = {}

        for cuda_id in range(safe_cuda_call(cuda.cuDeviceGetCount)):
            device = safe_cuda_call(cuda.cuDeviceGet, cuda_id)
            uuid_bytes = safe_cuda_call(cuda.cuDeviceGetUuid, device)
            uuid_str = 'GPU-' + str(UUID(bytes=uuid_bytes.bytes))
            uuid_to_cuda_id[uuid_str] = cuda_id

        return uuid_to_cuda_id

    try:
        pynvml.nvmlInit()

        handles = [pynvml.nvmlDeviceGetHandleByIndex(physical_id) for physical_id in range(pynvml.nvmlDeviceGetCount())]
        uuid_to_cuda_id = get_uuid_to_cuda_id()
        uuid_to_handle = {pynvml.nvmlDeviceGetUUID(handle): handle for handle in handles}

        uuids = list(uuid_to_handle.keys())
        uuids.sort(key=lambda u: pynvml.nvmlDeviceGetMemoryInfo(uuid_to_handle[u]).free, reverse=True)

        if not allow_share:
            uuids = [uuid for uuid in uuids if pynvml.nvmlDeviceGetComputeRunningProcesses(uuid_to_handle[uuid]) == []]

        for uuid in uuids:
            if uuid not in uuid_to_cuda_id:
                continue

            cuda_id = uuid_to_cuda_id[uuid]
            device = torch.device(f'cuda:{cuda_id}')
            logging.info(f'Using GPU: {torch.cuda.get_device_name(device)} (index {device.index}).')
            return device

        if allow_cpu:
            logging.warning('No available GPU found. Using CPU.')
            return torch.device('cpu')

        raise RuntimeError('No available GPU found.')

    finally:
        pynvml.nvmlShutdown()


def timer(log_format: str = 'Function "{__timer__.func.__name__}" took {__timer__.elapsed:.2f} seconds.') -> Callable:
    """
    A decorator to measure execution time of a function.
    Available placeholders are arguments of the decorated function and 'func', 'start', 'end', 'elapsed' in '__timer__' namespace.
    Note that the decorated function should not have a parameter named '__timer__'.

    :param log_format: The format string for logging.
    :return: The decorated function with timer.
    """
    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            timer_namespace = types.SimpleNamespace()
            timer_namespace.func = func
            timer_namespace.start = start
            timer_namespace.end = end
            timer_namespace.elapsed = end - start

            logging.info(log_format.format(**bound_args.arguments, __timer__=timer_namespace))
            return result

        return wrapper

    return decorator
