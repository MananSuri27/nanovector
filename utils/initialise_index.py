import numpy as np
from config import IndexConfig

from index.index import Index
from index.pca_index import PCAIndex


def initialise_index(config: IndexConfig, embeddings: np.array):
    if config.pca:
        return PCAIndex(
            embeddings=embeddings,
            dimension_input=config.dim_input,
            dimension_final=config._dim_final,
            normalise=config.normalise,
        )
    else:
        assert config.dim_input == config.dim_final
        return Index(
            embeddings=embeddings,
            dimension=config.dim_final,
            normalise=config.normalise,
        )
