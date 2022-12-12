import importlib
from types import ModuleType
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

import sift._types as st
from sift._constants import Backend

__all__ = [
    "is_torch_available",
    "is_torch_tensor",
    "is_numpy_tensor",
    "get_device",
    "get_dtype",
    "get_backend",
    "assert_backend_available",
]


def _safe_import(modname: str, *, raise_exc: bool = False) -> Optional[ModuleType]:
    try:
        return importlib.import_module(modname)
    except ImportError:
        if not raise_exc:
            return None
        raise ImportError("TODO") from None


def is_torch_available() -> bool:
    return _safe_import("torch", raise_exc=False) is not None


def is_numpy_tensor(x: st.ArrayLike, allow_sparse: bool = False) -> bool:
    return isinstance(x, np.ndarray) or (allow_sparse and sp.issparse(x))


def is_torch_tensor(x: st.ArrayLike) -> bool:
    torch = _safe_import("torch")
    return torch.is_tensor(x) if torch else False


def get_device(backend: Backend) -> Optional[st.Device_t]:
    if backend == Backend.NUMPY:
        return None
    if backend == Backend.TORCH:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise NotImplementedError(backend)


def get_dtype(backend: Backend) -> st.DtypeLike:
    if backend == Backend.NUMPY:
        return np.float32
    if backend == Backend.TORCH:
        import torch

        return torch.float32
    raise NotImplementedError(backend)


def get_backend(backend: Optional[Union[str, Backend]]) -> Backend:
    if backend is None:
        return Backend.TORCH if is_torch_available() else Backend.NUMPY
    backend = Backend(backend)
    assert_backend_available(backend)
    return backend


def assert_backend_available(backend: Backend) -> None:
    if backend == Backend.NUMPY:
        return
    if backend == Backend.TORCH:
        assert is_torch_available(), f"Install `{backend}` first."
