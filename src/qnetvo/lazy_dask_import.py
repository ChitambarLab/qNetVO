"""
Lazily imports dask when needed.
Throws an error if the user does not have dask[delayed] installed.
"""
try:
    import dask

    from dask import delayed as dask_delayed
    from dask import compute as dask_compute

except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Dask must be installed for parallel evaluation. "
        "\nDask can be installed using pip:"
        "\n\npip install dask[delayed]"
    ) from e
