from typing import Literal, Union

import numpy as np
import scipy.sparse as sp

__all__ = ["ArrayLike", "DtypeLike", "Device_t", "Kernel_t"]


ArrayLike = Union[np.ndarray, sp.spmatrix]
KernelLike = np.ndarray
DtypeLike = np.dtype

Device_t = Literal["cpu", "cuda"]
try:
    import torch

    Device_t = Union[Device_t, torch.device]
    ArrayLike = Union[ArrayLike, torch.Tensor]
    DtypeLike = Union[DtypeLike, torch.dtype]
    KernelLike = Union[KernelLike, torch.Tensor]
except ImportError:
    pass

Kernel_t = Literal["precomputed", "knn", "mapping", "rbf", "laplacian"]
