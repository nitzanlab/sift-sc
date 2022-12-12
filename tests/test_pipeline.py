import numpy as np
import pytest
from anndata import AnnData

from sift import SiFT


@pytest.mark.parametrize("copy", [False, True])
def test_copy(adata: AnnData, copy: bool):
    X_orig = adata.X.copy()

    sft = SiFT(adata, metric="rbf", kernel_key="X_umap", copy=copy)
    sft.filter()
    if copy:
        assert adata is not sft.adata
    else:
        assert adata is sft.adata

    assert np.sum(sft.adata.X == X_orig) == 0
