from __future__ import annotations

import importlib.util
import inspect
import os
import tempfile
import warnings
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import xarray as xr

from xarray_prism._delegate import API_ONLY_KWARGS, DECODER_PARAMETERS

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
THREDDS_ENDPOINT = os.environ.get("THREDDS_ENDPOINT", "http://localhost:8088")

S3_ENDPOINT_URL = MINIO_ENDPOINT
S3_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
S3_BUCKET = "testdata"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def s3_storage_options() -> dict:
    """Return S3 storage options for MinIO."""
    return {
        "key": S3_ACCESS_KEY,
        "secret": S3_SECRET_KEY,
        "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL},
    }


@pytest.fixture
def s3_env(s3_storage_options: dict) -> Generator[dict, None, None]:
    """
    Set AWS environment variables for S3 access.

    This is needed because detect_engine uses fsspec without storage_options,
    so credentials must come from environment variables.

    Also sets GDAL-specific variables for rasterio/rioxarray.
    """
    endpoint_url = s3_storage_options["client_kwargs"]["endpoint_url"]
    # Extract host:port from endpoint URL for GDAL
    endpoint_host = endpoint_url.replace("http://", "").replace("https://", "")

    env_vars = {
        # Standard AWS env vars
        "AWS_ACCESS_KEY_ID": s3_storage_options["key"],
        "AWS_SECRET_ACCESS_KEY": s3_storage_options["secret"],
        "AWS_ENDPOINT_URL": endpoint_url,
        # GDAL-specific env vars
        "AWS_S3_ENDPOINT": endpoint_host,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "NO",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    try:
        yield s3_storage_options
    finally:
        for key in env_vars:
            os.environ.pop(key, None)


@pytest.fixture(scope="session")
def s3_endpoint() -> str:
    """Return the S3 endpoint URL."""
    return S3_ENDPOINT_URL


@pytest.fixture(scope="session")
def thredds_endpoint() -> str:
    """Return the THREDDS server endpoint."""
    return THREDDS_ENDPOINT


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for GRIB cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_netcdf_path(data_dir: Path) -> Path:
    """Return path to a sample NetCDF4 file."""
    nc_path = (
        data_dir
        / "model/global/cmip6/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/amip/r1i1p1f1"
        / "Amon/ua/gn/v20191108/ua_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197001-201512.nc"
    )
    return nc_path


@pytest.fixture
def sample_grib_path(data_dir: Path) -> Path:
    """Return path to a sample GRIB file."""
    return data_dir / "grib_data/gfs/2025/11/25/test.grib2"


@pytest.fixture
def sample_geotiff_path(data_dir: Path) -> Path:
    """Return path to a sample GeoTIFF file."""
    return data_dir / "geodata/TCD/2021/10m/districts/DE111/TCD_S2021_R10m_DE111.tif"


@pytest.fixture
def sample_cordex_path(data_dir: Path) -> Path:
    """Return path to a sample CORDEX NetCDF file."""
    return (
        data_dir
        / "model/regional/cordex/output/EUR-11/GERICS/NCC-NorESM1-M/rcp85/r1i1p1"
        / "GERICS-REMO2015/v1/3hr/pr/v20181212"
        / "pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"
    )


@pytest.fixture
def sample_incoherent_encoding_path(data_dir: Path) -> Path:
    """Return path to a incoherent NetCDF file."""
    return data_dir / "misc/diff_missing_fill.nc"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_minio: mark test as requiring MinIO service"
    )
    config.addinivalue_line(
        "markers", "requires_thredds: mark test as requiring THREDDS service"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring local test data"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available services and data."""
    import socket

    def is_service_available(host: str, port: int, timeout: float = 1.0) -> bool:
        """Check if a service is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except (socket.error, OSError):
            return False

    minio_available = is_service_available("localhost", 9000)
    thredds_available = is_service_available("localhost", 8088)
    data_available = DATA_DIR.exists()

    skip_minio = pytest.mark.skip(reason="MinIO service not available")
    skip_thredds = pytest.mark.skip(reason="THREDDS service not available")
    skip_data = pytest.mark.skip(reason="Test data directory not found")

    for item in items:
        if "requires_minio" in item.keywords and not minio_available:
            item.add_marker(skip_minio)
        if "requires_thredds" in item.keywords and not thredds_available:
            item.add_marker(skip_thredds)
        if "requires_data" in item.keywords and not data_available:
            item.add_marker(skip_data)


# first-class open_dataset arguments fixtures, parametrization and probes

XR_OPEN_DATASET_PARAMS = frozenset(inspect.signature(xr.open_dataset).parameters)

_PASSTHROUGH_KWARGS = frozenset(
    {"filename_or_obj", "drop_variables", *DECODER_PARAMETERS}
)

_CONSUMED_KWARGS = frozenset({"decode_cf", "backend_kwargs", "kwargs"})


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _needs_dask(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not _module_available("dask"), reason=reason)


def _needs_open_dataset_arg(name: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        name not in XR_OPEN_DATASET_PARAMS,
        reason=f"xarray {xr.__version__} has no open_dataset({name}=...)",
    )


_DECODER_CASES = {
    "defaults": {},
    "decode_cf-False": {"decode_cf": False},
    "decode_cf-True": {"decode_cf": True},
    "mask_and_scale-False": {"mask_and_scale": False},
    "decode_times-False": {"decode_times": False},
    "decode_timedelta-True": {"decode_timedelta": True},
    "concat_characters-False": {"concat_characters": False},
    "decode_coords-all": {"decode_coords": "all"},
    "decode_coords-False": {"decode_coords": False},
    "drop_variables": {"drop_variables": "junk"},
    "cache-False": {"cache": False},
    "cache-True": {"cache": True},
    "create_default_indexes-False": {"create_default_indexes": False},
}

_RASTERIO_CASES = {
    "defaults": {},
    "mask_and_scale-False": {"mask_and_scale": False},
    "decode_times-False": {"decode_times": False},
    "decode_coords-all": {"decode_coords": "all"},
}

_API_LAYER_CASES = {
    "cache-False": {"cache": False},
    "cache-True": {"cache": True},
    "chunks": {"chunks": {}},
    "create_default_indexes": {"create_default_indexes": False},
    "inline_array": {"chunks": {}, "inline_array": True},
    "chunked_array_type": {"chunks": {}, "chunked_array_type": "dask"},
    "from_array_kwargs": {"chunks": {}, "from_array_kwargs": {}},
}

_CASE_MARKS = {
    "create_default_indexes": (_needs_open_dataset_arg("create_default_indexes"),),
    "create_default_indexes-False": (
        _needs_open_dataset_arg("create_default_indexes"),
    ),
    "chunks": (_needs_dask("chunks needs a chunk manager"),),
    "inline_array": (_needs_dask("inline_array needs a chunk manager"),),
    "chunked_array_type": (_needs_dask("chunked_array_type needs a chunk manager"),),
    "from_array_kwargs": (_needs_dask("from_array_kwargs needs a chunk manager"),),
}


def _params(cases: dict) -> list:
    return [
        pytest.param(kwargs, id=name, marks=list(_CASE_MARKS.get(name, ())))
        for name, kwargs in cases.items()
    ]


def pytest_generate_tests(metafunc):
    """Supply the argument matrices by fixture name."""
    if "decoder_kwargs" in metafunc.fixturenames:
        metafunc.parametrize("decoder_kwargs", _params(_DECODER_CASES))
    if "rasterio_kwargs" in metafunc.fixturenames:
        metafunc.parametrize("rasterio_kwargs", _params(_RASTERIO_CASES))
    if "api_layer_kwargs" in metafunc.fixturenames:
        metafunc.parametrize("api_layer_kwargs", _params(_API_LAYER_CASES))
    if "api_only_kwarg" in metafunc.fixturenames:
        metafunc.parametrize("api_only_kwarg", sorted(API_ONLY_KWARGS))
    if "cache_enabled" in metafunc.fixturenames:
        metafunc.parametrize("cache_enabled", [True, False], ids=["cache", "nocache"])
    if "any_dataset" in metafunc.fixturenames:
        metafunc.parametrize(
            "any_dataset",
            ["cf_file", "netcdf3_file", "zarr_store", "geotiff_file"],
            indirect=True,
        )


@pytest.fixture(scope="session")
def cf_decoding_supported() -> None:
    """
    Skip when the installed xarray/pandas pair cannot decode timedeltas.
    """
    try:
        var = xr.Variable(("t",), np.array([0, 6, 12]), {"units": "hours"})
        xr.conventions.decode_cf_variables({"t": var}, {})[0]["t"].values
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"xarray {xr.__version__} cannot decode timedeltas with the "
            f"installed pandas ({exc}); parity is untestable here"
        )


@pytest.fixture(scope="session")
def cf_file(tmp_path_factory, cf_decoding_supported) -> str:
    """
    NetCDF4 file exercising every CF decoder, incl. conflicting fill values.
    """
    nc = pytest.importorskip("netCDF4")
    path = tmp_path_factory.mktemp("prism") / "cf.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 4)
        ds.createDimension("x", 3)
        ds.createDimension("strlen", 4)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.calendar = "standard"
        time[:] = [0, 1, 2, 3]

        lead = ds.createVariable("lead", "f8", ("time",))
        lead.units = "hours"
        lead[:] = [0, 6, 12, 18]

        lon = ds.createVariable("lon", "f8", ("x",))
        lon[:] = [1.0, 2.0, 3.0]

        tas = ds.createVariable("tas", "i2", ("time", "x"), fill_value=np.int16(-32767))
        tas.scale_factor = 0.01
        tas.add_offset = 273.15
        tas.coordinates = "lon"
        tas[:] = np.arange(12, dtype="i2").reshape(4, 3)

        temperature = ds.createVariable(
            "temperature", "f4", ("time",), fill_value=np.float32(-9999.0)
        )
        temperature.missing_value = np.float64(-8888.0)
        temperature[:] = [280.0, 281.0, 282.0, 283.0]

        label = ds.createVariable("label", "S1", ("time", "strlen"))
        label[:] = np.array(
            [list(b"ab  "), list(b"cd  "), list(b"ef  "), list(b"gh  ")], dtype="S1"
        )

        junk = ds.createVariable("junk", "f8", ("x",))
        junk[:] = [9.0, 9.0, 9.0]
    return str(path)


@pytest.fixture(scope="session")
def netcdf3_file(tmp_path_factory, cf_decoding_supported) -> str:
    """NetCDF3 classic file -> scipy engine."""
    nc = pytest.importorskip("netCDF4")
    pytest.importorskip("scipy")
    path = tmp_path_factory.mktemp("prism") / "n3.nc"
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", 3)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time[:] = [0, 1, 2]
        tas = ds.createVariable("tas", "i2", ("time",))
        tas.scale_factor = 0.5
        tas.add_offset = 100.0
        tas[:] = [1, 2, 3]
    return str(path)


@pytest.fixture(scope="session")
def zarr_store(cf_file, tmp_path_factory) -> str:
    """Zarr store -> zarr engine."""
    pytest.importorskip("zarr")
    path = tmp_path_factory.mktemp("prism") / "z.zarr"
    try:
        with xr.open_dataset(cf_file, drop_variables="temperature") as ds:
            ds.to_zarr(path, mode="w")
    except Exception as exc:
        pytest.skip(f"cannot write zarr with this xarray/zarr combination: {exc}")
    return str(path)


@pytest.fixture(scope="session")
def geotiff_file(tmp_path_factory) -> str:
    """Single-band uint8 GeoTIFF -> rasterio engine."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    from rasterio.transform import from_origin

    path = tmp_path_factory.mktemp("prism") / "u8.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        nodata=0,
        crs="EPSG:4326",
        transform=from_origin(0, 3, 1, 1),
    ) as dst:
        dst.write(np.arange(9, dtype="uint8").reshape(1, 3, 3))
    return str(path)


@pytest.fixture
def any_dataset(request) -> str:
    """Indirect fixture: resolves a fixture *name* to its path."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def fingerprint():
    """Comparable summary of an opened dataset."""

    def _fingerprint(ds: xr.Dataset) -> dict:
        return {
            "dtypes": {n: str(v.dtype) for n, v in ds.variables.items()},
            "coords": sorted(ds.coords),
            "indexes": sorted(ds.xindexes),
            "values": {
                n: repr(np.asarray(v.values).ravel()[:3])
                for n, v in sorted(ds.variables.items())
            },
        }

    return _fingerprint


@pytest.fixture(scope="session")
def wrapper_chain():
    """
    Lazy-array wrappers around a variable, outermost first.
    """

    def _wrapper_chain(ds: xr.Dataset, name: str) -> list:
        node = ds[name].variable._data
        chain = []
        for _ in range(10):
            chain.append(type(node).__name__)
            node = getattr(node, "array", None)
            if node is None:
                break
        return chain

    return _wrapper_chain


@pytest.fixture(scope="session")
def open_probe(fingerprint):
    """
    Open a dataset and report (fingerprint or exception, warning categories).
    """

    def _open_probe(path: str, engine: str, kwargs: dict) -> tuple:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with xr.open_dataset(path, engine=engine, **kwargs) as ds:
                    result = fingerprint(ds)
            except Exception as exc:
                result = f"raised {type(exc).__name__}"
        return result, sorted({w.category.__name__ for w in caught})

    return _open_probe


@pytest.fixture(scope="session")
def assert_parity(open_probe):
    """
    Assert prism matches the engine it delegates to, warnings included.
    """

    def _assert_parity(path: str, delegate: str, kwargs: dict) -> None:
        expected = open_probe(path, delegate, kwargs)
        actual = open_probe(path, "prism", kwargs)
        assert actual == expected, (
            f"\n  kwargs   : {kwargs}"
            f"\n  {delegate:<9}: {expected}"
            f"\n  prism    : {actual}"
        )

    return _assert_parity


class _RecordingBackend:
    """
    Stands in for the delegate; records what prism hands it, then defers.
    """

    def __init__(self, real):
        self._real = real
        self.captured: dict | None = None

    def __getattr__(self, name):
        return getattr(self._real, name)

    def open_dataset(self, filename_or_obj, **kwargs):
        self.captured = dict(kwargs)
        return self._real.open_dataset(filename_or_obj, **kwargs)


@pytest.fixture
def recorder(monkeypatch) -> _RecordingBackend:
    """
    Intercept prism's delegate lookup, not xarray's lookup of prism.
    """
    pytest.importorskip("h5netcdf")
    from xarray.backends import plugins

    real_get_backend = plugins.get_backend
    rec = _RecordingBackend(real_get_backend("h5netcdf"))

    def scoped_get_backend(engine):
        if engine == "prism" or not isinstance(engine, str):
            return real_get_backend(engine)
        return rec

    monkeypatch.setattr(plugins, "get_backend", scoped_get_backend)
    return rec


@pytest.fixture(scope="session")
def open_dataset_arguments() -> frozenset:
    """Every first-class argument of this xarray's open_dataset."""
    return XR_OPEN_DATASET_PARAMS


@pytest.fixture(scope="session")
def classified_kwargs() -> dict:
    """How xarray-prism accounts for
    each open_dataset argument."""
    return {
        "api_only": API_ONLY_KWARGS,
        "passthrough": _PASSTHROUGH_KWARGS,
        "consumed": _CONSUMED_KWARGS,
    }


@pytest.fixture
def quiet():
    """Suppress warnings that are irrelevant
    to the assertion under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
