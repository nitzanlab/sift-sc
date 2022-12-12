import os
import random
import string
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from scipy import ndimage
from scipy.sparse import csr_matrix, spmatrix

from sift._constants import UNS_KEY, KernelType

__all__ = ["plot_kernel"]


def _knn_kernel_base(
    adata: Optional[AnnData] = None,
    kernel_key: Optional[str] = None,
    uns_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    **kwargs,
) -> str:

    n_neighbors = kwargs.pop("n_neighbors", 15)
    batch_key = kwargs.pop("knn_batch_key", None)
    key_added = kwargs.pop("knn_key_added", "sift")
    use_tgt = kwargs.pop("use_tgt", False)
    use_bbknn = kwargs.pop("use_bbknn", True)
    epsilon = 1e-6

    if (tgt_key is not None) and use_tgt:
        from sklearn.neighbors import NearestNeighbors

        tgt_idx = _get_mask_key(adata=adata, key=tgt_key)
        tgt_idx_idx = np.argwhere(tgt_idx.values).flatten()
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        container = _get_key(adata, kernel_key)
        neigh.fit(container[tgt_idx, :])
        distances_, neighbors_ = neigh.kneighbors(container)
        rows = (
            np.zeros((adata.n_obs, n_neighbors)) + np.arange(adata.n_obs)[:, np.newaxis]
        ).flatten()
        cols = tgt_idx_idx[neighbors_.flatten()]
        distances_ = (
            1 - (distances_ / distances_.max(1)[:, np.newaxis])
        ).flatten() + epsilon
        adata.obsm[f"{key_added}_connectivities"] = csr_matrix(
            (distances_, (rows, cols)), shape=(adata.n_obs, adata.n_obs)
        )

    else:
        try:
            from bbknn import bbknn

            BBKNN = True
        except ImportError:
            BBKNN = False
        BBKNN = BBKNN and use_bbknn
        if BBKNN:
            if batch_key is None:  # add dummy batch
                batch_key = "".join(random.sample(string.ascii_lowercase, 10))
                adata.obs[batch_key] = 1
                sc.external.pp.bbknn(
                    adata,
                    batch_key=batch_key,
                    use_rep=kernel_key,
                    neighbors_within_batch=n_neighbors,
                )

                del adata.obs[batch_key]
            else:
                neighbors_within_batch = int(
                    n_neighbors / len(adata.obs[batch_key].unique())
                )
                sc.external.pp.bbknn(
                    adata,
                    batch_key=batch_key,
                    use_rep=kernel_key,
                    neighbors_within_batch=neighbors_within_batch,
                )
                adata.uns[uns_key]["batch_key"] = batch_key

            adata.obsp[f"{key_added}_connectivities"] = adata.obsp[
                "connectivities"
            ].copy()

            del adata.obsp["connectivities"]
            del adata.obsp["distances"]
            del adata.uns["neighbors"]
        else:
            sc.pp.neighbors(
                adata, n_neighbors=n_neighbors, use_rep=kernel_key, key_added=key_added
            )
    kernel_key = f"{key_added}_connectivities"
    return kernel_key


def _kernel_base(
    adata: Optional[AnnData] = None,
    kernel_key: Optional[str] = None,
    metric: Optional[str] = KernelType.RBF,
    precomputed_kernel: Optional[np.ndarray] = None,
    uns_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    adata.uns[uns_key] = {}
    if precomputed_kernel is not None:
        adata.uns[uns_key]["metric"] = KernelType.PRECOMPUTED
        container = precomputed_kernel

    else:
        if metric == KernelType.KNN:
            kernel_key = _knn_kernel_base(
                adata=adata,
                kernel_key=kernel_key,
                uns_key=uns_key,
                tgt_key=tgt_key,
                **kwargs,
            )

        adata.uns[uns_key]["kernel_key"] = kernel_key
        adata.uns[uns_key]["metric"] = metric.name
        container = _get_key(adata, kernel_key)
    if container.ndim == 1:
        if metric == KernelType.MAPPING:
            if container.dtype.name != "category":
                n_bins = kwargs.pop("n_bins", 15)
                container = pd.cut(container, bins=n_bins)
                adata.uns[uns_key]["n_bins"] = n_bins
                adata.uns[uns_key]["bins"] = container
            container = pd.get_dummies(container).values
        else:
            container = container[:, np.newaxis]
    return container


def _prepare_kernel_base(
    adata: AnnData,
    container: np.ndarray,
    metric: Optional[str] = KernelType.RBF,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    uns_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    src_idx = np.arange(container.shape[0])
    tgt_idx = np.arange(container.shape[0])
    if src_key is not None:
        src_idx = _get_mask_idx(adata=adata, key=src_key)
        adata.uns[uns_key]["src_key"] = src_key
    if tgt_key is not None:
        tgt_idx = _get_mask_idx(adata=adata, key=tgt_key)
        adata.uns[uns_key]["tgt_key"] = tgt_key

    x = container[src_idx, :]
    y = container[tgt_idx, :]
    if metric == KernelType.KNN or metric == KernelType.PRECOMPUTED:
        x = x[:, tgt_idx]
        y = None

    return x, y


def _prepare_kernel_container(
    adata: Optional[AnnData] = None,
    kernel_key: Optional[str] = None,
    metric: Optional[str] = KernelType.RBF,
    precomputed_kernel: Optional[np.ndarray] = None,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    uns_key = kwargs.pop("uns_key", UNS_KEY)
    container = _kernel_base(
        adata=adata,
        kernel_key=kernel_key,
        metric=metric,
        precomputed_kernel=precomputed_kernel,
        uns_key=uns_key,
        tgt_key=tgt_key,
        **kwargs,
    )

    return _prepare_kernel_base(
        adata=adata,
        container=container,
        metric=metric,
        src_key=src_key,
        tgt_key=tgt_key,
        uns_key=uns_key,
    )


def _get_embedding(
    adata: AnnData,
    embedding_key: Optional[str] = None,
) -> Tuple[Union[np.ndarray, spmatrix], str, str]:
    if embedding_key is not None:
        emb, type_emb = _get_basis(adata, embedding_key)
    else:
        embedding_key = "X"
        type_emb = "X"
        emb = adata.X

    return emb, type_emb, embedding_key


def _save_res(
    adata: AnnData,
    embedding: np.ndarray,
    type_emb: str,
    type_res: Optional[str] = "sift",
    embedding_key: Optional[str] = None,
    key_added: Optional[str] = None,
) -> Tuple[str, str]:
    # save projection\filtered
    key_added = key_added if key_added is not None else f"{embedding_key}_{type_res}"

    if type_emb == "X" and type_res == "sift":
        adata.X = embedding.copy()
        attr = "x"
    elif type_emb == "layer" or type_emb == "X":
        adata.layers[key_added] = embedding.copy()
        attr = "layers"
    elif type_emb == "obsm" or type_emb == "X_obsm":
        adata.obsm[key_added] = embedding.copy()
        attr = "obsm"

    return key_added, attr


def _get_basis(adata: AnnData, basis: str) -> Tuple[Union[np.ndarray, spmatrix], str]:
    if basis == "X":
        return adata.X, "X"
    if f"X_{basis}" in adata.obsm:
        return adata.obsm[f"X_{basis}"], "X_obsm"
    elif basis in adata.obsm:
        return adata.obsm[basis], "obsm"
    elif basis in adata.layers:
        return adata.layers[basis], "layer"
    else:
        raise KeyError(
            f"Unable to find a basis in `adata.obsm['X_{basis}']`,"
            f" `adata.obsm[{basis!r}]` or adata.layer[{basis!r}."
        ) from None


def _get_key(adata: AnnData, key: str) -> np.ndarray:
    if key in adata.obsm:
        container = adata.obsm[key]
    elif f"X_{key}" in adata.obsm:
        container = adata.obsm[f"X_{key}"]
    elif key in adata.obs:
        container = adata.obs[key]
    elif key in adata.obsp:
        container = adata.obsp[key]
    elif key in adata.var:  # if key defines a subset of marker genes
        container = adata.X[:, adata.var[key]]
    else:
        raise KeyError(
            f"Unable to find a basis in:"
            f" `adata.obsm['X_{key}']` or `adata.obsm[{key!r}]`"
            f" or `adata.obs[{key!r}]` "
            f"or `adata.obsm['X_{key}']` or `adata.obsp[{key!r}]`"
            f" or `adata.var[{key!r}]`."
        ) from None

    return container


def _get_mask_key(adata: AnnData, key: str) -> np.ndarray:
    if key in adata.obs:
        container = adata.obs[key]
    elif key in adata.obsm:
        container = adata.obsm[key]
    elif key in adata.var:  # if key defines a subset of marker genes
        container = adata.var[key]
    else:
        raise KeyError(
            f"Unable to find a basis in:"
            f" `adata.obsm[{key!r}]` or `adata.obs[{key!r}]` "
            f"or `adata.var[{key!r}]`."
        ) from None
    if container.dtype != "bool":
        raise KeyError(
            f"Masking key must be of 'bool' type," f" got {container.dtype} instead."
        ) from None
    return container


def _get_mask_idx(adata: AnnData, key: str) -> np.ndarray:
    return np.where(_get_mask_key(adata=adata, key=key))[0]


def _maybe_create_dir(dirpath: Union[str, os.PathLike]) -> None:
    """
    Try creating a directory if it does not already exist.
    Parameters
    ----------
    dirpath
        Path of the directory to create.
    Returns
    -------
    None
        Nothing, just creates a directory if it doesn't exist.
    """

    if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
        try:
            os.makedirs(dirpath, exist_ok=True)
        except OSError:
            pass


def plot_kernel(
    kernel: np.ndarray,
    adata: Optional[AnnData] = None,
    src_key: Optional[str] = None,
    tgt_key: Optional[str] = None,
    groupby: Optional[Union[str, Sequence[str]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Visualize the cell-cell similarity kernel.
    Parameters
    ----------
    kernel
        kernel to plot
    adata
         Optional, reference adata
    src_key
        Optional, mask for source space of kernel
    tgt_key
        Optional, mask for target space of kernel
    groupby
        the key of the observation grouping to consider.
    save_path
        path and fig_name to save the fig
    show
        If `False`, return :class:`matplotlib.pyplot.Axes`.
    **kwargs
        additional plotting arguments
    Returns
    -------
    The axes object, if ``show = False``.
    """
    interpolation = kwargs.pop("interpolation", True)
    if interpolation:
        kernel = ndimage.gaussian_filter(kernel, sigma=3)
    vmax = kwargs.pop("vmax", np.quantile(kernel, 0.9))
    vmax = vmax if vmax > 0 else np.max(kernel)
    color_palette = kwargs.pop("color_palette", "deep")
    cmap = kwargs.pop("cmap", "Blues")
    ncol = kwargs.pop("ncol", 2)
    figsize = kwargs.pop("figsize", None)
    fontsize = kwargs.pop("fontsize", 20)
    if groupby is not None:
        src_idx = (
            _get_mask_idx(adata=adata, key=src_key)
            if src_key is not None
            else np.arange(adata.n_obs)
        )
        tgt_idx = (
            _get_mask_idx(adata=adata, key=tgt_key)
            if tgt_key is not None
            else np.arange(adata.n_obs)
        )
        if adata is None:
            raise ValueError("cannot use groupby without anndata reference.")

        if groupby in adata.obs:
            groups = adata.obs[groupby]
        elif groupby in adata.uns[UNS_KEY]:
            groups = adata.uns[UNS_KEY][groupby]
        else:
            raise ValueError(
                "cannot find groupby in `anndata.obs` " 'or `anndata.uns["kernel"]`.'
            )

        if groups.dtype.name == "category":
            labels = groups.cat.categories.astype(str)
            df = pd.DataFrame(
                groups.astype(str), index=adata.obs_names, columns=[groupby]
            )
        else:
            labels = groups.unique()
            df = pd.DataFrame(
                groups.astype(str), index=adata.obs_names, columns=[groupby]
            )

        label_pal = sns.color_palette(color_palette, labels.size)
        label_lut = dict(zip(map(str, labels), label_pal))
        row_colors = pd.Series(
            df[groupby].iloc[src_idx], index=adata.obs_names[src_idx], name=groupby
        ).map(label_lut)
        col_colors = pd.Series(
            df[groupby].iloc[tgt_idx], index=adata.obs_names[tgt_idx], name=groupby
        ).map(label_lut)

        g = sns.clustermap(
            kernel,
            row_cluster=False,
            col_cluster=False,
            standard_scale=False,
            col_colors=col_colors.to_numpy(),
            row_colors=row_colors.to_numpy(),
            linewidths=0,
            xticklabels=[],
            yticklabels=[],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            figsize=figsize,
            **kwargs,
        )
        if adata.obs[groupby].dtype.name == "category":
            for label in labels:
                g.ax_col_dendrogram.bar(
                    0, 0, color=label_lut[label], label=label, linewidth=0
                )

            g.ax_col_dendrogram.legend(
                title=groupby,
                loc="center",
                ncol=ncol,
                bbox_to_anchor=(0.47, 0.9),
                bbox_transform=plt.gcf().transFigure,
            )
    else:
        g = sns.clustermap(
            kernel,
            row_cluster=False,
            col_cluster=False,
            linewidths=0,
            xticklabels=[],
            yticklabels=[],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            figsize=figsize,
        )

    g.cax.set_position([0.05, 0.2, 0.03, 0.45])
    ax = g.ax_heatmap
    ax.set_xlabel("cells", fontsize=fontsize)
    ax.set_ylabel("cells", fontsize=fontsize)

    if save_path is not None:
        g.fig.savefig(save_path, bbox_inches="tight", transparent=True)

    if not show:
        return g.cax
