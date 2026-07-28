"""
test xarray's first-class ``open_dataset`` arguments
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from xarray_prism import PrismBackendEntrypoint
from xarray_prism._delegate import API_ONLY_KWARGS, DECODER_PARAMETERS


class TestParity:
    """prism must produce what the delegate
    engine produces, warnings included."""

    def test_netcdf4(self, cf_file, decoder_kwargs, assert_parity):
        pytest.importorskip("h5netcdf")
        assert_parity(cf_file, "h5netcdf", decoder_kwargs)

    def test_netcdf3(self, netcdf3_file, decoder_kwargs, assert_parity):
        assert_parity(netcdf3_file, "scipy", decoder_kwargs)

    def test_zarr(self, zarr_store, decoder_kwargs, assert_parity):
        assert_parity(zarr_store, "zarr", decoder_kwargs)

    def test_geotiff(self, geotiff_file, rasterio_kwargs, assert_parity):
        assert_parity(geotiff_file, "rasterio", rasterio_kwargs)

    def test_open_dataarray(self, cf_file, quiet):
        pytest.importorskip("h5netcdf")
        drop = ["tas", "junk", "lead", "label", "lon"]
        for kwargs in ({}, {"decode_cf": False}):
            with xr.open_dataarray(
                cf_file, engine="h5netcdf", drop_variables=drop, **kwargs
            ) as expected:
                with xr.open_dataarray(
                    cf_file, engine="prism", drop_variables=drop, **kwargs
                ) as actual:
                    assert actual.dtype == expected.dtype

    def test_open_mfdataset(self, cf_file, quiet):
        pytest.importorskip("dask")
        pytest.importorskip("h5netcdf")
        for kwargs in ({}, {"decode_cf": False}):
            with xr.open_mfdataset([cf_file], engine="h5netcdf", **kwargs) as expected:
                with xr.open_mfdataset([cf_file], engine="prism", **kwargs) as actual:
                    assert actual["tas"].dtype == expected["tas"].dtype


class TestOriginalReport:
    """https://github.com/freva-org/xarray-prism -- decode_cf=False ignored."""

    def test_no_serialization_warning(self, cf_file, recwarn):
        from xarray.coding.variables import SerializationWarning

        with xr.open_dataset(cf_file, engine="prism", decode_cf=False) as ds:
            assert ds["tas"].dtype == np.int16
            assert ds["time"].dtype == np.float64

        offenders = [w for w in recwarn if issubclass(w.category, SerializationWarning)]
        assert not offenders, f"unexpected SerializationWarning(s): {offenders}"

    def test_decoders_are_advertised(self):
        """xarray only expands decode_cf for the names a backend advertises."""
        declared = set(PrismBackendEntrypoint.open_dataset_parameters)
        assert set(DECODER_PARAMETERS) <= declared
        assert {"filename_or_obj", "drop_variables"} <= declared


class TestCacheWrapping:
    """cache is applied by the API layer
    and must not be overridden early."""

    def test_wrapper_chain_matches_delegate(
        self, cf_file, cache_enabled, wrapper_chain, quiet
    ):
        pytest.importorskip("h5netcdf")
        with xr.open_dataset(
            cf_file, engine="h5netcdf", cache=cache_enabled
        ) as expected:
            reference = wrapper_chain(expected, "tas")
        with xr.open_dataset(cf_file, engine="prism", cache=cache_enabled) as actual:
            observed = wrapper_chain(actual, "tas")

        assert observed == reference
        assert ("MemoryCachedArray" in observed) is cache_enabled


class TestDelegationBoundary:
    """What prism hands the real backend,
    asserted at the handover itself."""

    def test_api_layer_args_never_reach_delegate(
        self, cf_file, recorder, api_layer_kwargs, quiet
    ):
        xr.open_dataset(cf_file, engine="prism", **api_layer_kwargs).close()

        assert recorder.captured is not None, "delegate was never called"
        leaked = API_ONLY_KWARGS & recorder.captured.keys()
        assert not leaked, f"API-layer args leaked into the delegate: {sorted(leaked)}"
        assert "decode_cf" not in recorder.captured, "decode_cf is not a backend param"
        assert "backend_kwargs" not in recorder.captured

    def test_decoders_do_reach_delegate(self, cf_file, recorder, quiet):
        """The flip side: decode_cf must
        arrive resolved, not dropped."""
        xr.open_dataset(cf_file, engine="prism", decode_cf=False).close()

        assert recorder.captured is not None
        unresolved = {
            name: recorder.captured.get(name)
            for name in DECODER_PARAMETERS
            if recorder.captured.get(name) is not False
        }
        assert not unresolved, f"decode_cf=False did not resolve these: {unresolved}"

    def test_direct_helpers_return_raw_backend_dataset(
        self, cf_file, fingerprint, wrapper_chain, quiet
    ):
        """
        open_posix / open_cloud / PrismBackendEntrypoint.open_dataset
        return an undecorated backend dataset, like every other
        BackendEntrypoint
        """
        pytest.importorskip("h5netcdf")
        from xarray.backends.plugins import get_backend

        from xarray_prism.backends import open_posix

        actual = open_posix(cf_file, engine="h5netcdf")
        reference = get_backend("h5netcdf").open_dataset(cf_file)
        try:
            assert list(actual.xindexes) == list(reference.xindexes)
            assert fingerprint(actual) == fingerprint(reference)
            assert "MemoryCachedArray" not in wrapper_chain(actual, "tas")
        finally:
            actual.close()
            reference.close()

        with xr.open_dataset(cf_file, engine="prism") as full:
            assert list(full.xindexes) == ["time"]


class TestBackendKwargs:
    """backend_kwargs is flattened by xarray;
    prism must cope predictably."""

    def test_decode_cf_matches_first_class_form(self, any_dataset, open_probe):
        """
        the rasterio decode_cf
        """
        first_class = open_probe(any_dataset, "prism", {"decode_cf": False})
        nested = open_probe(
            any_dataset, "prism", {"backend_kwargs": {"decode_cf": False}}
        )
        assert nested == first_class

    def test_decode_cf_is_allowed(self, cf_file, quiet):
        """The one deliberate exception
        must survive the API-only check."""
        pytest.importorskip("h5netcdf")
        with xr.open_dataset(
            cf_file, engine="prism", backend_kwargs={"decode_cf": False}
        ) as ds:
            assert ds["tas"].dtype == np.int16

    def test_api_only_args_raise_clearly(self, cf_file, api_only_kwarg):
        with pytest.raises(TypeError, match="must be passed directly"):
            xr.open_dataset(
                cf_file, engine="prism", backend_kwargs={api_only_kwarg: None}
            )


class TestRasterioDivergence:
    """prism deliberately adapts rather
    than propagating rioxarray's error."""

    def test_decode_cf_succeeds_where_rioxarray_raises(self, geotiff_file, caplog):
        from rioxarray.exceptions import RioXarrayError

        with pytest.raises(RioXarrayError, match="decode_coords"):
            xr.open_dataset(geotiff_file, engine="rasterio", decode_cf=False)

        with caplog.at_level(logging.WARNING, logger="xarray_prism"):
            with xr.open_dataset(geotiff_file, engine="prism", decode_cf=False) as ds:
                assert ds["band_data"].dtype == np.uint8
        assert "decode_coords" in caplog.text

    def test_dropped_kwargs_are_logged(self, geotiff_file, caplog):
        """Discarding a user argument must at least be reported."""
        with caplog.at_level(logging.WARNING, logger="xarray_prism"):
            xr.open_dataset(
                geotiff_file, engine="prism", concat_characters=False
            ).close()
        assert "concat_characters" in caplog.text


class TestSignatureDrift:
    """A new xarray argument must be classified, not silently ignored."""

    def test_no_unclassified_open_dataset_arguments(
        self, open_dataset_arguments, classified_kwargs
    ):
        """Each argument is either handled by xarray's API layer, forwarded to
        the delegate, or consumed en route.
        """
        known = frozenset().union(*classified_kwargs.values())
        unclassified = open_dataset_arguments - known
        assert not unclassified, (
            f"xarray {xr.__version__} has open_dataset argument(s) "
            f"{sorted(unclassified)} that xarray-prism has not classified. "
            f"Decide whether each is API-layer-only, forwarded to the delegate, "
            f"or consumed by prism, then update the corresponding set."
        )

    def test_advertised_parameters_are_real_backend_parameters(self):
        advertised = set(PrismBackendEntrypoint.open_dataset_parameters)
        assert not advertised & API_ONLY_KWARGS, (
            "open_dataset_parameters advertises an API-layer-only argument: "
            f"{sorted(advertised & API_ONLY_KWARGS)}"
        )
        assert "decode_cf" not in advertised, "decode_cf is never a backend parameter"
