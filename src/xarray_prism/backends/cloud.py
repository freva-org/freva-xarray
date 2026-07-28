"""Cloud backend for xarray datasets."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .._cache import cache_remote_file

logger = logging.getLogger(__name__)


def open_cloud(
    uri: str,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    **kwargs,
) -> Any:
    """Open remote file with detected engine."""
    from .._delegate import open_with_engine
    from ..utils import gdal_env

    storage_options = kwargs.pop("storage_options", None)

    bk = backend_kwargs or None

    # GRIB / NetCDF3: must download the full file first
    if engine in ("cfgrib", "scipy"):
        local_path = cache_remote_file(uri, engine, storage_options, show_progress)
        return open_with_engine(
            local_path,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=bk,
            **kwargs,
        )

    # NetCDF4 (OPeNDAP)
    if engine == "netcdf4":
        return open_with_engine(
            uri,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=bk,
            **kwargs,
        )

    # Rasterio: translate storage_options -> GDAL env vars
    if engine == "rasterio":
        from ..utils import sanitize_rasterio_kwargs

        with gdal_env(storage_options):
            return open_with_engine(
                uri,
                engine=engine,
                drop_variables=drop_variables,
                backend_kwargs=bk,
                **sanitize_rasterio_kwargs(kwargs),
            )

    # Zarr, h5netcdf
    ds = open_with_engine(
        uri,
        engine=engine,
        drop_variables=drop_variables,
        backend_kwargs=bk,
        storage_options=storage_options,
        **kwargs,
    )
    if engine == "h5netcdf":
        from ..utils import sanitize_dataset_attrs

        return sanitize_dataset_attrs(ds)
    return ds
