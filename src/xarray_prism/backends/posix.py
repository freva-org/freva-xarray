"""Module to open local files using xarray
with a specified engine."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def open_posix(
    uri: str,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """Open local file with detected engine."""
    import xarray as xr

    # the following posix backends don't accept storage_options
    _NO_STORAGE_OPTIONS = frozenset({"cfgrib", "scipy", "netcdf4", "rasterio"})
    if engine in _NO_STORAGE_OPTIONS:
        kwargs.pop("storage_options", None)

    if engine == "cfgrib":
        from .._cache import get_cache_dir

        bk = dict(backend_kwargs or {})
        if "indexpath" not in bk:
            basename = os.path.basename(uri)
            bk["indexpath"] = str(get_cache_dir() / f"{basename}.{{short_hash}}.idx")
        backend_kwargs = bk

    if engine == "rasterio":
        from ..utils import sanitize_rasterio_kwargs

        kwargs = sanitize_rasterio_kwargs(kwargs)
    ds = xr.open_dataset(
        uri,
        engine=engine,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs or None,
        **kwargs,
    )
    if engine == "h5netcdf":
        from ..utils import sanitize_dataset_attrs

        return sanitize_dataset_attrs(ds)
    return ds
