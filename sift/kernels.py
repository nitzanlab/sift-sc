"""Kernels for SiFT filtering."""

# Note: this module is strongly inspired by the kernel module of sklearn
# <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/gaussian_process/kernels.py>
#

import numbers
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pykeops
from pykeops.common.lazy_tensor import GenericLazyTensor
from scipy.sparse import issparse

import sift._backend_utils as bu
import sift._types as st
from sift._constants import Backend, KernelType

__all__ = [
    "MappingKernel",
    "PrecomputedKernel",
    "KnnKernel",
    "RBFKernel",
    "LaplacianKernel",
]
torch = bu._safe_import("torch", raise_exc=False)


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


class Kernel(ABC):
    """Base kernel class."""

    def __init__(
        self,
        kernel: Optional[st.ArrayLike] = None,
        dtype: Optional[st.DtypeLike] = None,
        device: Optional[st.Device_t] = None,
        backend: Optional[Backend] = None,
    ):
        self._k = kernel
        self._dtype = dtype
        self._backend = bu.get_backend(backend)
        self._device = bu.get_device(self._backend) if device is None else device

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Evaluate the kernel on a pair of inputs.

        Parameters
        ----------
        x
            Left argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
        y
            Right argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
            If `None`, :math:`k(X, X)` is evaluated instead.
        materialize
            Whether to materialize the kernel.

        Returns
        -------
        LazyTensor resembling the kernel.
        """

    @classmethod
    def _create(cls, kernel_type: st.Kernel_t, **kwargs: Any) -> "Kernel":
        kernel_type = KernelType(kernel_type)
        if kernel_type == KernelType.PRECOMPUTED:
            return PrecomputedKernel(**kwargs)
        if kernel_type == KernelType.KNN:
            return KnnKernel(**kwargs)
        if kernel_type == KernelType.MAPPING:
            return MappingKernel(**kwargs)
        if kernel_type == KernelType.RBF:
            return RBFKernel(**kwargs)
        if kernel_type == KernelType.LAPLACIAN:
            return LaplacianKernel(**kwargs)
        raise NotImplementedError(kernel_type)

    @property
    def dtype(self) -> st.DtypeLike:
        """The kernel's data type."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: st.DtypeLike) -> None:
        self._dtype = dtype

    @property
    def device(self) -> Optional[st.Device_t]:
        """The kernel's device."""
        return self._device

    @device.setter
    def device(self, device: Optional[st.Device_t]) -> None:
        self._device = device

    @property
    def backend(self) -> Backend:
        """The kernel's :mod:`pykeops` backend.

        Can be either :mod:`torch` or :mod:`numpy`."""
        return self._backend

    @backend.setter
    def backend(self, backend: Union[str, Backend]) -> None:
        self._backend = Backend(backend)

    @property
    def k(self) -> st.KernelLike:
        """The instantiated kernel object."""
        if self._k is None:
            raise RuntimeError(
                "Must instantiate a kernel object or supply a valid basis"
            )
        return self._k

    @k.setter
    def k(self, kernel: st.KernelLike) -> None:
        self._k = kernel

    def _materialize_kernel(self):
        if isinstance(self.k, GenericLazyTensor):
            if self.backend == Backend.TORCH:
                return self.k @ torch.diag(
                    torch.ones(self.k.shape[1], dtype=self.dtype, device=self.device)
                )
            elif self.backend == Backend.NUMPY:
                return self.k @ np.diag(np.ones(self.k.shape[1]))
        else:
            return self.k

    def _LazyTensor(self, x, axis=0):
        if self._backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            if axis == 0:
                return pykeops.torch.LazyTensor(x[:, None, None])
            return pykeops.torch.LazyTensor(x[None, :, None])
        elif self._backend == Backend.NUMPY:
            return pykeops.numpy.LazyTensor(x)

    def _Vi(self, x):
        if self._backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            return pykeops.torch.Vi(x)
        elif self._backend == Backend.NUMPY:
            return pykeops.numpy.Vi(x)

    def _Vj(self, x):
        if self._backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            return pykeops.torch.Vj(x)
        elif self._backend == Backend.NUMPY:
            return pykeops.numpy.Vj(x)

    def _Pm(self, x):
        if self._backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            return pykeops.torch.Pm(x)
        elif self._backend == Backend.NUMPY:
            return pykeops.numpy.Pm(x)

    def _normalize(self, K) -> GenericLazyTensor:
        """Row normalize an array.

        Parameters
        ----------
        K
            The array to normalize.

        Returns
        -------
        The row-normalized kernel.
        """
        norm = K.sum(1)
        # norm = np.where(norm == 0)
        return self._Vi(1 / norm) * K


class ConstantKernel(Kernel):
    r"""Constant kernel.

    Apply row-normalization to the provided constant.

    .. math::
        k(x_1, x_2) = constant\_value \;\forall\; x_1, x_2

    Parameters
    ----------
    constant_value
        The constant value which defines the covariance k(x_1, x_2) = constant_value
    kwargs
        Keyword arguments for the parent class.
    """

    def __init__(self, constant_value: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.constant_value = constant_value

    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Compute the constant kernel.

        Parameters
        ----------
        x
            Left argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
        y
            Right argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
            If `None`, :math:`k(X, X)` is evaluated instead.
        materialize
            Whether to materialize the kernel.

        Returns
        -------
        The kernel.
        """

        if y is None:
            y = x
        if self.backend == Backend.TORCH:
            self.k = torch.full(
                (_num_samples(x), _num_samples(y)),
                self.constant_value,
                dtype=self.dtype,
                device=self.device,
            )
        elif self.backend == Backend.NUMPY:
            self.k = np.full(
                (_num_samples(x), _num_samples(y)),
                self.constant_value,
            )
        if materialize:
            return self._materialize_kernel()
        return self.k

    def __repr__(self) -> str:
        return f"{self.constant_value:.3g}"


class PrecomputedKernel(Kernel):
    """Precomputed kernel.

    Defines a kernel based on a given pre-computed similarity and
    applies row-normalization to the provided kernel.
    """

    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Return the row-normalized kernel.

        Parameters
        ----------
        x
            Array of shape `[n_samples, n_samples]` to normalize.
        y
            Ignored, used for API compatibility.
        materialize
            Ignored, the kernel is already pre-computed.

        Returns
        -------
        The kernel.
        """

        if issparse(x):
            x = x.copy()
            row_sums = np.array(x.sum(axis=1))[:, 0]
            row_indices, col_indices = x.nonzero()
            x.data[:] = x.data / row_sums[row_indices]
            k = x
        elif self.backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            k = x / torch.sum(x, 1).unsqueeze(-1)
        elif self.backend == Backend.NUMPY:
            k = x / np.sum(x, 1)[:, np.newaxis]
        else:
            raise ValueError(
                f"received {self.backend} which is not a valid SiFT backend. See sift.Backends."
            )
        self.k = k
        return k

    def __repr__(self) -> str:
        return "K"


class KnnKernel(PrecomputedKernel):
    """k-NN kernel.

    Defines a kernel based on a given pre-computed k-nn similarity and
    applies row-normalization to the provided kernel.
    """

    def __repr__(self) -> str:
        return "knn"


class MappingKernel(Kernel):
    r"""Mapping kernel.

    Define a kernel based on a given mapping :math:`T`.

    .. math::
        k(x_i, x_j) =  \sum_k p_{l}(k) p_{c}(k)

    where :math:`p_{l} = T / T.sum(0), p_{c} = T / T.sum(1)`

    Parameters
    ----------
    ignore_self
        Whether to ignore self transitions.
    kwargs
        Keyword arguments for the parent class.
    """

    def __init__(self, ignore_self: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self._ignore_self = ignore_self

    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Compute the mapping kernel.

        Parameters
        ----------
        x
            Left argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
        y
            Right argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
            If `None`, :math:`k(X, X)` is evaluated instead.
        materialize
            Whether to materialize the kernel.

        Returns
        -------
        The kernel.
        """
        if y is None:
            y = x
        if x is y:
            y = y.copy()

        if issparse(x) and issparse(y):
            row_sums = np.array(x.sum(axis=1))[:, 0]
            row_indices, col_indices = x.nonzero()
            x.data[:] = x.data / row_sums[row_indices]
            pl = x

            y = y.T
            col_sums = np.array(y.sum(axis=1))[:, 0]
            row_indices, col_indices = y.nonzero()
            y.data[:] = y.data / col_sums[row_indices]
            pc = y.T

            K = pl @ pc.T
            if self._ignore_self:
                K.setdiag(0)
                K = self._normalize(K)
            self.k = K

        elif self.backend == Backend.TORCH:
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype, device=self.device).contiguous()
            pl = x / torch.sum(x, 1).unsqueeze(-1)

            if not torch.is_tensor(y):
                y = torch.tensor(y, dtype=self.dtype, device=self.device).contiguous()
            pc = y / torch.sum(y, 0)
            K = self._Vi(pl) | self._Vj(pc)
            if self._ignore_self:
                i = self._LazyTensor(np.arange(x.shape[0]), axis=0)
                j = self._LazyTensor(np.arange(y.shape[0]), axis=1)
                K = K - K * (0.5 - (i - j) ** 2).step()
                K = self._normalize(K)
            self.k = K
        elif self.backend == Backend.NUMPY:
            pl = x / np.sum(x, 1)[:, np.newaxis]
            pc = y / np.sum(y, 1)
            K = pl @ pc.T

            if self._ignore_self:
                np.fill_diagonal(K, 0)
                K = self._normalize(K)
            self.k = K
        else:
            raise ValueError(
                f"received {self.backend} which is not a valid SiFT backend."
            )

        if materialize:
            return self._materialize_kernel()

        return self.k

    def __repr__(self) -> str:
        return "Mapping kernel, T"


class RBFKernel(Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBFKernel kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and :math:`d(x_i,x_j)`
    is the Euclidean distance.

    For advice on how to set the length scale parameter, see e.g. :cite:`duvenaud:14`, Chapter 2.
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth. See :cite:`rasmussen:05`, Chapter 4, Section 4.2, for further details.

    Parameters
    ----------
    length_scale
        The length scale.
    ignore_self
        Whether to ignore self transitions.
    kwargs
        Keyword arguments for the parent class.
    """

    def __init__(
        self, length_scale: float = 1.0, ignore_self: bool = False, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self._ignore_self = ignore_self

    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Compute the RBF kernel.

        Parameters
        ----------
        x
            Left argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
        y
            Right argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
            If `None`, :math:`k(X, X)` is evaluated instead.
        materialize
            Whether to materialize the kernel.

        Returns
        -------
        The kernel.
        """

        if y is None:
            y = x

        K = (
            -self._Pm(1 / (2 * self.length_scale**2))
            * self._Vi(x).sqdist(self._Vj(y))
        ).exp()
        if self._ignore_self:
            i = self._LazyTensor(np.arange(x.shape[0]), axis=0)
            j = self._LazyTensor(np.arange(y.shape[0]), axis=1)
            K = K - K * (0.5 - (i - j) ** 2).step()
        self.k = self._normalize(K)

        if materialize:
            return self._materialize_kernel()

        return self.k

    def __repr__(self) -> str:
        ls = np.ravel(self.length_scale)[0]
        return f"{self.__class__.__name__}(length_scale={ls:.3g})"


class LaplacianKernel(Kernel):
    r"""L1-exponential kernel.

    It is parameterized by a length scale parameter :math:`l>0`,
    The kernel is given by:

    .. math::
        k(x_i, x_j) = \exp\left(- \frac{| x_i, x_j|_{1}}{2l^2} \right)

    where :math:`l` is the length scale of the kernel.

    Parameters
    ----------
    length_scale
        The length scale.
    ignore_self
        Whether to ignore self transitions.
    kwargs
        Keyword arguments for the parent class.
    """

    def __init__(
        self, length_scale: float = 1.0, ignore_self: bool = False, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self._ignore_self = ignore_self

    def __call__(
        self,
        x: st.ArrayLike,
        y: Optional[st.ArrayLike] = None,
        materialize: bool = False,
    ) -> st.KernelLike:
        """Compute the Laplacian kernel.

        Note that this compound kernel returns the results of all simple kernel
        stacked along an additional axis.

        Parameters
        ----------
        x
            Left argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
        y
            Right argument of the returned kernel :math:`k(X, Y)`, array of shape ``[n_samples, n_features]``.
            If `None`, :math:`k(X, X)` is evaluated instead.
        materialize
            Whether to materialize the kernel.

        Returns
        -------
        The kernel.
        """
        if y is None:
            y = x

        K = (
            -(
                self._Pm(1 / (2 * self.length_scale**2))
                * self._Vi(x).sqdist(self._Vj(y))
            ).sqrt()
        ).exp()

        if self._ignore_self:
            i = self._LazyTensor(np.arange(x.shape[0]), axis=0)
            j = self._LazyTensor(np.arange(y.shape[0]), axis=1)
            K = K - K * (0.5 - (i - j) ** 2).step()
        self.k = self._normalize(K)

        if materialize:
            return self._materialize_kernel()
        return self.k

    def __repr__(self) -> str:
        ls = np.ravel(self.length_scale)[0]
        return f"{self.__class__.__name__}(length_scale={ls:.3g})"
