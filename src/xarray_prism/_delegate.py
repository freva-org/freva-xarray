"""
Delegation to the concrete xarray backend entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Decoder flags that xarray.open_dataset resolves from decode_cf
DECODER_PARAMETERS = (
    "mask_and_scale",
    "decode_times",
    "decode_timedelta",
    "concat_characters",
    "use_cftime",
    "decode_coords",
)


# Arguments xarray.open_dataset handles itself, after the backend
# has returned
API_ONLY_KWARGS = frozenset(
    {
        "cache",
        "chunks",
        "create_default_indexes",
        "inline_array",
        "chunked_array_type",
        "from_array_kwargs",
        "engine",
    }
)


def _reject_api_only_kwargs(kwargs: Dict[str, Any]) -> None:
    misplaced = sorted(API_ONLY_KWARGS & kwargs.keys())
    if not misplaced:
        return
    names = ", ".join(repr(m) for m in misplaced)
    example = misplaced[0]
    raise TypeError(
        f"xarray-prism: {names} must be passed directly to xarray.open_dataset, "
        f"not inside backend_kwargs. xarray applies these itself after the "
        f"backend returns, so nesting them cannot work with any engine.\n"
        f"  use:    xr.open_dataset(uri, engine='prism', {example}=...)\n"
        f"  not:    xr.open_dataset(uri, engine='prism', "
        f"backend_kwargs={{{example!r}: ...}})"
    )


def _expand_decode_cf(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a decode_cf that reached us through backend_kwargs
    """
    if "decode_cf" not in kwargs:
        return kwargs

    decode_cf = kwargs.pop("decode_cf")
    if decode_cf is False:
        for name in DECODER_PARAMETERS:
            kwargs[name] = False
    return kwargs


def normalize_open_kwargs(
    kwargs: Dict[str, Any],
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Flatten backend_kwargs and expand decode_cf into decoder flags
    """
    merged = dict(kwargs)
    if backend_kwargs:
        merged.update(backend_kwargs)
    _reject_api_only_kwargs(merged)
    return _expand_decode_cf(merged)


def open_with_engine(
    target: Any,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Open target with engine's backend entrypoint
    """
    from xarray.backends.plugins import get_backend

    kwargs = normalize_open_kwargs(kwargs, backend_kwargs)

    kwargs = {
        k: v for k, v in kwargs.items() if not (k in DECODER_PARAMETERS and v is None)
    }

    backend = get_backend(engine)
    return backend.open_dataset(target, drop_variables=drop_variables, **kwargs)
