import logging
import os
from types import MappingProxyType
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

import sift._backend_utils as bu
import sift._types as st
from sift._constants import BATCH_SIZE, MAX_SRC_SIZE, Backend, KernelType, Stage
from sift._utils import (
    _get_embedding,
    _get_mask_idx,
    _prepare_kernel_container,
    _save_res,
)
from sift._utils import plot_kernel as _plot_kernel
from sift.kernels import Kernel

logger = logging.getLogger(__name__)


__all__ = ["SiFT", "sifter"]


class SiFT:
    """SiFT object.

    Parameters
    ----------
    adata
        Annotated data object.
    kernel_key
        Key in :class:`anndata.AnnData` where information for distance is stored.
    metric
        The kernel type.
    src_key
        Mask for the source space of the kernel.
    tgt_key
        Mask for the target space of the kernel.
    knn_key
        wey in :attr:`anndata.AnnData.obs` for k-NN masking.
    save: bool
        Whether to save the kernel.
    kernel_params
        Kernel object parameters [length_scale].
    kwargs
        Keyword arguments for kernel computations  [precomputed_kernel, n_neighbors, knn_batch_key].

    Examples
    --------
    >>> import scanpy as sc
    >>> import sift
    >>> adata = sc.read(...)
    >>> sft = sift.SiFT(
            adata=adata,
            kernel_key="cell_cycle_genes",
            metric="RBF",
        )
    >>> sft.filter()
    """

    def __init__(
        self,
        adata: AnnData,
        kernel_key: Optional[str] = None,
        metric: st.Kernel_t = KernelType.RBF,
        mask_key: Optional[str] = None,
        src_key: Optional[str] = None,
        tgt_key: Optional[str] = None,
        copy: bool = False,
        kernel_params: Optional[Dict[str, Any]] = MappingProxyType({}),
        **kwargs: Any,
    ):
        metric = KernelType(metric)
        self._adata = adata.copy() if copy else adata
        self._kernel_key = kernel_key
        self._kernel_metric = metric
        self._src_key = src_key
        self._tgt_key = tgt_key
        self._stage = Stage.INITIALIZED
        self._kernel = None
        self._kernel_mask = None
        self._mask_key = None
        self._kernel_containers = {}
        self._copy = copy

        backend = bu.get_backend(None)
        bu.assert_backend_available(backend)
        device = bu.get_device(backend)
        dtype = bu.get_dtype(backend)

        self._device = kwargs.pop("device", device)
        self._dtype = kwargs.pop("dtype", dtype)
        self._backend = kwargs.pop("backend", backend)
        self._batch_size = kwargs.pop("batch_size", BATCH_SIZE)
        self._max_src_size = kwargs.pop("max_src_size", MAX_SRC_SIZE)

        self._init_kernels(
            kernel_key=kernel_key,
            mask_key=mask_key,
            metric=metric,
            kernel_params=kernel_params,
            **kwargs,
        )

        self._model_summary_string = f"SiFT object with {metric} kernel"
        logger.info(f"initialized a SiFTer with {metric} kernel.")
        if self._mask_key is not None:
            logger.info(f"SiFTer has a mask_kernel of type {self._mask_key}")

    @property
    def adata(self) -> AnnData:
        """The SiFT annotated data object."""
        return self._adata

    @adata.setter
    def adata(self, adata: AnnData) -> None:
        self._adata = adata

    @property
    def kernel(self) -> st.KernelLike:
        """The :mod:`pykeops` kernel object."""
        return self._kernel(**self._kernel_containers)

    @kernel.setter
    def kernel(self, k: st.KernelLike) -> None:
        self._kernel = k

    def _init_kernels(
        self,
        kernel_key: Optional[str] = None,
        mask_key: Optional[str] = None,
        metric: st.Kernel_t = KernelType.RBF,
        precomputed_kernel: Optional[st.ArrayLike] = None,
        kernel_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Sets the cell-cell similarity kernel.

        Parameters
        ----------
        kernel_key
            Key in :class:`anndata.AnnData` where information for distance is stored.
        mask_key
            Key in :attr:`anndata.AnnData.obs` for k-NN masking.
        metric
          The kernel type.
        precomputed_kernel
            The user can provide a pre-computed kernel, it is assumed cells are ordered as in :attr:`adata`.
        kernel_params
            Kernel object parameters [length_scale].
        kwargs
            Keyword arguments for kernel computations  [precomputed_kernel, n_neighbors, knn_batch_key].

        Returns
        -------
        Nothing, just initializes the kernels.
        """
        (
            self._kernel_containers["x"],
            self._kernel_containers["y"],
        ) = _prepare_kernel_container(
            adata=self.adata,
            kernel_key=kernel_key,
            metric=metric,
            precomputed_kernel=precomputed_kernel,
            src_key=self._src_key,
            tgt_key=self._tgt_key,
            **kwargs,
        )

        self._kernel = Kernel._create(
            metric,
            device=self._device,
            dtype=self._dtype,
            backend=self._backend,
            **kernel_params,
        )

        if mask_key is not None:
            x, y = _prepare_kernel_container(
                adata=self.adata,
                kernel_key=mask_key,
                metric=KernelType.KNN,
                src_key=self._src_key,
                tgt_key=self._tgt_key,
                uns_key="mask_kernel",
                **kwargs,
            )

            _kernel_mask = Kernel._create(
                KernelType.MAPPING,
                device=self._device,
                dtype=self._dtype,
                backend=self._backend,
            )
            _kernel_mask(x=x, y=y)

            self._kernel_mask = (
                _kernel_mask.k.A if issparse(_kernel_mask.k) else _kernel_mask.k
            )
            self._mask_key = mask_key

    def _materialize_kernel(
        self,
        k: Optional[st.ArrayLike] = None,
        to_cpu: Optional[bool] = True,
    ) -> st.KernelLike:
        """Materialize the initialized kernel.

        Parameters
        ----------
        k
            Kernel to materialize, if `None` call `self._kernel.kernel()`.
        to_cpu: bool
            Whether to move kernel to CPU.

        Returns
        -------
        The materialized kernel object.

        Notes
        -----
        This operation can result in out-of-memory (OOM) error.
        """
        k = (
            k
            if k is not None
            else self._kernel(materialize=True, **self._kernel_containers)
        )
        k = k.cpu().numpy() if to_cpu and bu.is_torch_tensor(k) else k
        k = k.A if issparse(k) else k
        return k

    def materialize_kernel(
        self, kernel_containers: Optional[Dict] = None, to_cpu: bool = True
    ) -> st.KernelLike:
        """Materialize the initialized kernel.

        Parameters
        ----------
        to_cpu
            Whether to move kernel to CPU.
        kernel_containers
            Keyword arguments to the kernel call.

        Returns
        -------
        The materialized kernel after masking, if :attr:`._kernel_mask` is present.

        Notes
        -----
        This operation can result in out-of-memory (OOM) error.
        """
        kernel_containers = (
            kernel_containers
            if kernel_containers is not None
            else self._kernel_containers
        )
        if self._mask_key is not None:
            return self._materialize_kernel(self._mask_kernel(), to_cpu=to_cpu)
        else:
            return self._materialize_kernel(
                self._kernel(materialize=True, **kernel_containers), to_cpu=to_cpu
            )

    def _mask_kernel(
        self, kernel_input_idx: Optional[Dict[str, Any]] = None
    ) -> st.ArrayLike:
        """Mask the kernel.

        Parameters
        ----------
        kernel_input_idx
            Dictionary with index for kernel input.

        Returns
        -------
        The materialized kernel after masking, if :attr:`_kernel_mask` is present.

        Notes
        -----
        This operation can result in out-of-memory (OOM) error.
        """
        if kernel_input_idx is not None:
            x = self._kernel_containers["x"][kernel_input_idx["x"], :]
            y = self._kernel_containers["y"][kernel_input_idx["y"], :]
            kernel_mask = (
                self._kernel_mask[kernel_input_idx["x"], kernel_input_idx["y"]]
                if self._mask_key
                else None
            )
        else:
            x = self._kernel_containers["x"]
            y = self._kernel_containers["y"]
            kernel_mask = self._kernel_mask if self._mask_key else None

        if self._mask_key is not None:
            if self._backend == Backend.TORCH and not issparse(self.kernel):
                k = (
                    self._materialize_kernel(
                        k=self._kernel(x=x, y=y, materialize=True), to_cpu=False
                    )
                    * kernel_mask
                )
            elif issparse(self.kernel) and self._mask_key is not None:
                k = self.kernel.multiply(kernel_mask)
            else:
                k = (
                    self._materialize_kernel(
                        k=self._kernel(materialize=True, **self._kernel_containers)
                    )
                    * kernel_mask
                )
        else:
            k = self._kernel(x=x, y=y)
        return k

    def filter(
        self,
        embedding_key: Optional[str] = None,
        use_raw: Optional[bool] = False,
        pseudocount: bool = False,
        key_added: Optional[str] = None,
    ) -> Optional[AnnData]:
        """Perform filtering, subtract the projection of the kernel from the expression.

        Parameters
        ----------
        embedding_key
            Key in :attr:`anndata.AnnData.obsm` for which to compute the projection on.
            If `None`, use :attr:`anndata.AnnData.X`.
        use_raw
            Whether to use :attr:`anndata.AnnData.raw`, if present.
        pseudocount
            If `True`, add a pseudocount to filtered expression to avoid negative values
        key_added
            Key used when saving the result. If `None`, use ``'{embedding_key}_sift'``.
            The result is saved to :attr:`anndata.AnnData.X`, :attr:`anndata.AnnData.layers` or
            :attr:`anndata.AnnData.obsm`, depending on the ``embedding_key``.

        Returns
        -------
        The filtered data matrix.
        """
        if use_raw is True and self.adata.raw is None:
            raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")
        if use_raw and self.adata.raw is not None:
            logger.info("Using `adata.raw`")
            self.adata = self.adata.raw.to_adata()

        emb, type_emb, embedding_key = _get_embedding(
            adata=self.adata, embedding_key=embedding_key
        )
        tgt_idx = (
            _get_mask_idx(adata=self.adata, key=self._tgt_key)
            if self._tgt_key is not None
            else np.arange(self.adata.n_obs)
        )

        src_idx = (
            _get_mask_idx(adata=self.adata, key=self._src_key)
            if self._src_key is not None
            else np.arange(self.adata.n_obs)
        )

        logger.info(
            f"Filtering cell-cell similarity kernel using projection on `{embedding_key}`."
        )
        if self._backend == Backend.TORCH:
            import torch

            if issparse(self.kernel):
                k = self._mask_kernel()
                emb_sifted = emb[src_idx, :] - k @ emb[tgt_idx, :]
            else:
                emb_src = emb[src_idx, :]
                if src_idx.shape[0] > self._max_src_size:
                    batches = np.array_split(
                        np.arange(src_idx.shape[0]),
                        np.ceil(src_idx.shape[0] / self._batch_size),
                    )
                    logger.info(f"Evaluating with {len(batches)} batches.")
                    y_idx = np.arange(tgt_idx.shape[0])
                    emb_sifted_batches = []
                    emb_tgt = torch.tensor(
                        emb[tgt_idx, :].A if issparse(emb) else emb[tgt_idx, :],
                        dtype=self._dtype,
                        device=self._device,
                    )
                    for batch in batches:
                        k = self._mask_kernel(kernel_input_idx={"x": batch, "y": y_idx})
                        emb_src_batch = torch.tensor(
                            emb_src[batch, :].A if issparse(emb) else emb_src[batch, :],
                            dtype=self._dtype,
                            device=self._device,
                        )
                        emb_sifted_batch = emb_src_batch - k @ emb_tgt
                        emb_sifted_batches.append(emb_sifted_batch.cpu().numpy())
                    emb_sifted = np.concatenate(emb_sifted_batches)
                else:
                    k = self._mask_kernel()
                    emb = torch.tensor(
                        emb.A if issparse(emb) else emb,
                        dtype=self._dtype,
                        device=self._device,
                    )
                    emb_sifted = emb[src_idx, :] - k @ emb[tgt_idx, :]

        else:  # numpy backend
            k = self._mask_kernel()
            emb = (
                emb.astype(dtype=self.kernel.dtype).A
                if issparse(emb)
                else emb.astype(dtype=self.kernel.dtype)
            )
            if issparse(self.kernel):
                emb_sifted = emb[src_idx, :] - k @ emb[tgt_idx, :]
            else:
                emb_sifted = emb[src_idx, :] - k.__matmul__(
                    emb[tgt_idx, :], backend="CPU"
                )

        if pseudocount:
            emb_sifted -= emb_sifted.min()

        if bu.is_torch_tensor(emb_sifted):
            emb_sifted = emb_sifted.cpu().numpy()

        emb_sifted_full = np.zeros((self.adata.n_obs, emb_sifted.shape[-1]))
        emb_sifted_full[src_idx, :] = (
            emb_sifted.A if issparse(emb_sifted) else emb_sifted
        )

        key_added, attr = _save_res(
            adata=self.adata,
            embedding=emb_sifted_full,
            type_emb=type_emb,
            type_res="sift",
            embedding_key=embedding_key,
            key_added=key_added,
        )
        self._stage = Stage.SIFTED

        if type_emb == "X":
            logger.info(
                "The data is `SiFTed`!\n"
                "The filtered embedding is stored in `adata.X`\n"
                "    Finish"
            )
        else:
            logger.info(
                f"The data is `SiFTed`!\n"
                f"The filtered embedding is stored in `adata.{attr}[{key_added!r}]`\n"
                f"    Finish"
            )
        if self._copy:
            return self.adata
        return

    def plot_kernel(
        self, save_path: Optional[os.PathLike] = None, show: bool = True, **kwargs: Any
    ) -> Optional[plt.Axes]:
        """Visualize the cell-cell similarity kernel.

        Parameters
        ----------
        save_path
            Path where to save the figure.
        show
            If `False`, return :class:`matplotlib.axes.Axes`.
        kwargs
            Additional keyword arguments to :func:`sift._utils.plot_kernel()`.

        Returns
        -------
        The axes object, if ``show = False``, otherwise `None`.
        """
        kernel_containers = kwargs.pop("kernel_containers", self._kernel_containers)
        groupby = kwargs.pop("groupby", None)
        if self._kernel_metric == KernelType.MAPPING:
            groupby = self._kernel_key
        return _plot_kernel(
            kernel=self.materialize_kernel(
                kernel_containers=kernel_containers, to_cpu=True
            ),
            adata=self.adata,
            src_key=self._src_key,
            tgt_key=self._tgt_key,
            groupby=groupby,
            save_path=save_path,
            show=show,
            **kwargs,
        )


def sifter(
    adata: AnnData,
    kernel_key: Optional[str] = None,
    metric: st.Kernel_t = KernelType.RBF,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    knn_key: Optional[str] = None,
    batch_key: Optional[str] = None,
    embedding_key: Optional[str] = None,
    use_raw: Optional[bool] = False,
    pseudocount: bool = False,
    key_added: Optional[str] = None,
    copy: bool = False,
    **kwargs: Any,
) -> AnnData:
    """Perform filtering, subtract the projection of the kernel from the expression.

    Parameters
    ----------
    adata
        Annotated data object.
    kernel_key
        Key in :class:`anndata.AnnData` where information for distance is stored.
    metric
        Metric to use.
    src_key
        Mask for the source space of the kernel.
    tgt_key
        Mask for the target space of the kernel.
    knn_key
        Key in :attr:`anndata.AnnData.obs` for k-NN masking.
    batch_key
        Key in :attr:`anndata.AnnData.obs` for batch masking.
    embedding_key
        Key in :attr:`anndata.AnnData.obsm` for which to compute the projection on.
        If `None`, use :attr:`anndata.AnnData.X`.
    use_raw
        Whether to use :attr:`anndata.AnnData.raw`, if present.
    pseudocount
        If `True`, add a pseudocount to filtered expression to avoid negative values.
    key_added
        Key used when saving the result. If `None`, use ``'{embedding_key}_sift'``.
        The result is saved to :attr:`anndata.AnnData.X`, :attr:`anndata.AnnData.layers` or
        :attr:`anndata.AnnData.obsm`, depending on the ``embedding_key``.
    copy
        Whether to modify ``adata`` inplace.
    kwargs
        Additional keyword arguments [n_neighbors, precomputed_kernel].

    Returns
    -------
    The filtered data matrix.
    """
    return SiFT(
        adata=adata,
        kernel_key=kernel_key,
        metric=metric,
        knn_key=knn_key,
        batch_key=batch_key,
        src_key=src_key,
        tgt_key=tgt_key,
        copy=copy,
        **kwargs,
    ).filter(
        embedding_key=embedding_key,
        use_raw=use_raw,
        pseudocount=pseudocount,
        key_added=key_added,
    )
