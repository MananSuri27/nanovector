class IndexConfig:
    """
    A configuration class for indexing vectors.

    Attributes:
        dim_input (int): The dimensionality of input vectors.
        dim_final (int): The desired dimensionality after processing.
        pca (bool): Whether to perform PCA dimension reduction (default is False).
        normalise (bool): Whether to normalize input vectors (default is True).

    Methods:
        dim_input: Get the dimensionality of input vectors.
        dim_final: Get the desired dimensionality after processing.
        pca: Check if PCA dimension reduction is enabled.
        normalise: Check if input vector normalization is enabled.
        __repr__(): Get a string representation of the configuration.

    Example:
        config = IndexConfig(dim_input=256, dim_final=64, pca=True, normalise=False)
        print(config.dim_input)  # Prints the dimensionality of input vectors
    """

    def __init__(
        self, dim_input: int, dim_final: int, pca: bool = False, normalise: bool = True
    ):
        """
        Initialize an IndexConfig instance.

        Args:
            dim_input (int): The dimensionality of input vectors.
            dim_final (int): The desired dimensionality after processing.
            pca (bool, optional): Whether to perform PCA dimension reduction (default is False).
            normalise (bool, optional): Whether to normalize input vectors (default is True).
        """
        self._dim_input = dim_input
        self._dim_final = dim_final
        self._pca = pca
        self._normalise = normalise

    @property
    def dim_input(self) -> int:
        """Get the dimensionality of input vectors."""
        return self._dim_input

    @property
    def dim_final(self) -> int:
        """Get the desired dimensionality after processing."""
        return self._dim_final

    @property
    def pca(self) -> bool:
        """Check if PCA dimension reduction is enabled."""
        return self._pca

    @property
    def normalise(self) -> bool:
        """Check if input vector normalization is enabled."""
        return self._normalise

    def __repr__(self) -> str:
        """
        Get a string representation of the configuration.

        Returns:
            str: A string representation of the configuration.
        """
        return f"IndexConfig(dim_input={self._dim_input}, dim_final={self._dim_final}, pca={self._pca}, normalise={self._normalise})"
