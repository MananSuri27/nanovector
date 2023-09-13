import numpy as np

from index.index import Index
from index.pca_index import PCAIndex
from utils.config import IndexConfig


def initialise_index(config: IndexConfig, embeddings: np.array):
    """
    Initialize an index for vectors based on the provided configuration.

    Args:
        config (IndexConfig): The configuration for the index.
        embeddings (np.array): The input vectors to be indexed.

    Returns:
        Index or PCAIndex: An instance of the index based on the configuration.

    Raises:
        AssertionError: If the dimensions specified in the configuration are not compatible.

    Example:
        config = IndexConfig(dim_input=256, dim_final=64, pca=True, normalise=False)
        embeddings = np.random.rand(100, 256)
        index = initialise_index(config, embeddings)
    """
    if config.pca:
        return PCAIndex(
            embeddings=embeddings,
            dimension_input=config.dim_input,
            dimension_final=config.dim_final,
            normalise=config.normalise,
        )
    else:
        assert (
            config.dim_input == config.dim_final
        ), "Input and final dimensions must be the same when PCA is not used."
        return Index(
            embeddings=embeddings,
            dimension=config.dim_final,
            normalise=config.normalise,
        )
