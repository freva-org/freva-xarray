# ---------------------------------------------------------------#
# Data format   | Remote backend         | Local FS  | Cache
# ---------------------------------------------------------------#
# GRIB          | cfgrib + fsspec        | cfgrib    | fsspec simplecache (full-file)
# Zarr          | zarr + fsspec          | zarr      | chunked key/value store
# NetCDF3       | scipy + fsspec         | scipy     | fsspec byte cache (full-file)
# NetCDF4/HDF5  | h5netcdf + fsspec      | h5netcdf  | fsspec byte cache (5 MB blocks)
# GeoTIFF       | rasterio + fsspec      | rasterio  | GDAL/rasterio block cache
# OPeNDAP/DODS  | netCDF4                | n/a       | n/a
# ---------------------------------------------------------------#

# Important: GRIB and NetCDF3 files are not chunk-addressable.
# cfgrib and scipy typically must read the entire file (and build
# its index) even when only a small subset is requested.

import logging
import os

from ._cache import cache_info, clear_cache
from ._detection import (
    detect_engine,
    detect_uri_type,
    register_detector,
    register_uri_type,
)
from ._registry import registry
from ._version import __version__  # noqa

_level = getattr(
    logging,
    os.environ.get("XARRAY_PRISM_LOG_LEVEL", "WARNING").upper(),
    logging.WARNING,
)
_logger = logging.getLogger("xarray_prism")
_logger.setLevel(_level)
if _level < logging.WARNING and not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    _logger.addHandler(_handler)

__all__ = [
    "PrismBackendEntrypoint",
    "detect_engine",
    "detect_uri_type",
    "register_detector",
    "register_uri_type",
    "registry",
    "cache_info",
    "clear_cache",
]


def __getattr__(name):
    if name == "PrismBackendEntrypoint":
        from .entrypoint import PrismBackendEntrypoint

        return PrismBackendEntrypoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
