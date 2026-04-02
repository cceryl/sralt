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
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .model import tensor_fingerprint


def get_object_repr(obj: Any, visited: set[int] | None = None, max_depth: int = 10) -> str:
    """
    Get a string representation of an object for caching purposes.
    All torch.device objects are treated as identical.

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
    visited.add(id(obj))

    if isinstance(obj, torch.device):
        return 'device'

    if isinstance(obj, SummaryWriter):
        return 'SummaryWriter'

    if isinstance(obj, torch.Tensor):
        return f'torch.Tensor(fingerprint={tensor_fingerprint(obj)})'

    if isinstance(obj, (list, tuple, set, frozenset)):
        elements = [get_object_repr(e, visited, max_depth) for e in obj]
        match type(obj).__name__:
            case 'list': return f'[{", ".join(elements)}]'
            case 'tuple': return f'({", ".join(elements)})'
            case 'set': return f'{{{", ".join(elements)}}}'
            case 'frozenset': return f'frozenset({{{", ".join(elements)}}})'

    if isinstance(obj, dict):
        items = [
            f'{get_object_repr(k, visited, max_depth)}: {get_object_repr(v, visited, max_depth)}'
            for k, v in obj.items()
        ]
        return f'{{{", ".join(items)}}}'

    repr_str = repr(obj)
    if ('at 0x' in repr_str):
        raise NotImplementedError(f'Object {repr_str} is not representable.')

    return repr_str


def disk_cache(
    cache_dir: str,
    hit: str = 'Loading cached result of function "{__cache__.func.__name__}" from "{__cache__.path}".',
    miss: str = 'Computing result of function "{__cache__.func.__name__}" and caching to "{__cache__.path}".'
) -> Callable:
    """
    A decorator to cache the result of a function to disk.
    Available placeholders are arguments of the decorated function and 'func', 'hash', 'path' in '__cache__' namespace.
    Note that the decorated function should not have a parameter named '__cache__'.
    All torch.device objects are treated as identical in cache key generation.

    :param cache_dir: Directory where cached results will be stored.
    :param hit: Log format string for cache hits.
    :param miss: Log format string for cache misses.
    :return: The decorated function with caching capability.
    """
    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            cache_dir_path = Path(cache_dir.format(**bound_args.arguments))
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            try:
                key = (func.__name__, {k: get_object_repr(v) for k, v in bound_args.arguments.items()})
            except NotImplementedError as e:
                raise RuntimeError(
                    f'Failed to generate cache key for function "{func.__name__}", arguments: {bound_args.arguments}.'
                ) from e

            key_bytes = pickle.dumps(key)
            key_hash = hashlib.md5(key_bytes).hexdigest()
            cache_path = cache_dir_path / f'{key_hash}.pkl'
            logging.debug(f'Key for function "{func.__name__}": {key}')

            cache_ns = types.SimpleNamespace(func=func, hash=key_hash, path=cache_path)

            if cache_path.is_file():
                logging.info(hit.format(**bound_args.arguments, __cache__=cache_ns))
                with cache_path.open('rb') as f:
                    cache_data = pickle.load(f)

                if isinstance(cache_data, dict) and cache_data.get('key') == key:
                    return cache_data['result']

                logging.warning(f'Hash collision detected for cache file "{cache_path}". Recomputing the result.')

            logging.info(miss.format(**bound_args.arguments, __cache__=cache_ns))
            result = func(*args, **kwargs)

            with cache_path.open('wb') as f:
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


def timer(format: str = 'Function "{__timer__.func.__name__}" took {__timer__.elapsed:.2f} seconds.') -> Callable:
    """
    A decorator to measure execution time of a synchronous function.
    Available placeholders are arguments of the decorated function and 'func', 'start', 'end', 'elapsed' in '__timer__' namespace.
    Note that the decorated function should not have a parameter named '__timer__'.

    :param format: The format string for logging.
    :return: The decorated function with timer.
    """
    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            start_time = time.time()
            start_counter = time.perf_counter()

            try:
                result = func(*args, **kwargs)
            except:
                logging.error(f'Timer failed for function "{func.__name__}" due to an exception.')
                raise

            end_time = time.time()
            end_counter = time.perf_counter()

            timer_ns = types.SimpleNamespace(
                func=func,
                start=start_time,
                end=end_time,
                elapsed=end_counter - start_counter
            )

            logging.info(format.format(**bound_args.arguments, __timer__=timer_ns))
            return result

        return wrapper

    return decorator
