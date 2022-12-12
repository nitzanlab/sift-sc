import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def adata() -> AnnData:
    return sc.datasets.pbmc3k_processed()
