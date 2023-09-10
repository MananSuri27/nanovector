class IndexConfig:
    def __init__(
        self, dim_input: int, dim_final: int, pca: bool = False, normalise: bool = True
    ):
        self._dim_input = dim_input
        self._dim_final = dim_final
        self._pca = pca
        self._normalise = normalise

    @property
    def dim_input(self):
        return self._dim_input

    @property
    def dim_final(self):
        return self._dim_final

    @property
    def pca(self):
        return self._pca

    @property
    def normalise(self):
        return self._normalise

    def __repr__(self) -> str:
        return f"IndexConfig(dim_input={self._dim_input}, dim_final={self._dim_final}, pca={self._pca}, normalise={self._normalise})"
