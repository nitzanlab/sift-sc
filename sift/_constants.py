from enum import Enum

__all__ = ["Backend", "KernelType", "Stage", "MAX_SRC_SIZE", "BATCH_SIZE", "UNS_KEY"]


class Backend(str, Enum):
    TORCH = "torch"
    NUMPY = "numpy"


class KernelType(str, Enum):
    PRECOMPUTED = "precomputed"
    KNN = "knn"
    MAPPING = "mapping"
    RBF = "rbf"
    LAPLACIAN = "laplacian"


class Stage(str, Enum):
    INITIALIZED = "initialized"
    SIFTED = "sifted"


MAX_SRC_SIZE = 400_000
BATCH_SIZE = 100_000
UNS_KEY = "kernel"
