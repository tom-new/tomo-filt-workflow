#!/usr/bin/env python3
"""Stack <foo>.<bar>.<model>.dvs.<type>.<depth>.dat slabs into a single NetCDF.

Input Format
------------
Each input text file must have 3 columns:
    lat  lon  dVs_percent
with longitude changing fastest and latitude slowest. Latitude spans [-90, 90],
longitude spans [-180, 180]. The <type> is one of:
    - repar  -> variable 'dVs_reparam_percent'
    - filt   -> variable 'dVs_tofi_percent'

Depth values are parsed from filenames (<depth> in km).

Output Dataset
--------------
Dims are (r, lat, lon) where r is the Earth radius in meters at each depth:
    r = earth_radius_km*1000 - depth_km*1000
and a coordinate 'depth' (km) that shares the 'r' dimension.

Longitude Handling
------------------
Although the source files include both -180° and +180° longitudes, the output
longitude coordinate runs from -180° to +179° (inclusive). For each depth,
the values at -180° and +180° are averaged; that mean is written back to the
-180° column, and the +180° column is removed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr

# regex to parse filenames like '<foo>.<bar>.<model>.dvs.<type>.<depth>.dat'
FNAME_RE = re.compile(
    r"""^(?P<foo>[^.]+)\.(?P<bar>[^.]+)\.(?P<model>[^.]+)\.dvs\.
        (?P<typ>repar|filt)\.(?P<depth>\d+(?:\.\d+)?)\.dat$""",
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Combine <foo>.<bar>.<model>.dvs.<type>.<depth>.dat slabs into a single NetCDF."
    )
    p.add_argument("indir", type=Path, help="directory containing the .dat files")
    p.add_argument("modelname", type=str, help="model name to match (the third token)")
    p.add_argument("outfile", type=Path, help="path to write the NetCDF file")
    p.add_argument(
        "--earth-radius-km",
        type=float,
        default=6371.0,
        help="earth mean radius in km (default 6371.0)",
    )
    return p.parse_args()


def parse_fname(path: Path) -> Optional[Tuple[str, str, float]]:
    """Return (model, typ, depth_km) parsed from a filename; None if it doesn't match."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    model = m.group("model")
    typ = m.group("typ")
    depth_km = float(m.group("depth"))
    return model, typ, depth_km


def load_grid_from_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    load one file and return (lat_vals, lon_vals, grid2d).

    grid2d has shape (nlat, nlon) with lat ascending (slow axis) and lon ascending
    (fast axis). input is assumed lon-fastest in the raw rows; we re-sort to enforce
    monotonic lat/lon and reshape accordingly.
    """
    arr = np.loadtxt(path, comments="#", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns 'lat lon dVs_percent'")

    lat = arr[:, 0]
    lon = arr[:, 1]
    val = arr[:, 2]

    lat_vals = np.unique(lat)
    lon_vals = np.unique(lon)

    nlat = lat_vals.size
    nlon = lon_vals.size
    if nlat * nlon != arr.shape[0]:
        raise ValueError(
            f"{path}: grid is not rectangular (unique lats={nlat}, lons={nlon}, rows={arr.shape[0]})"
        )

    # sort by (lat, lon) so that reshape to (nlat, nlon) yields lon as fastest axis
    idx = np.lexsort((lon, lat))  # last key (lat) is primary
    grid = val[idx].reshape(nlat, nlon)

    # fold the 180° longitude into -180° by averaging, then drop the 180° column
    # this makes longitude run from -180 to 179 (inclusive)
    # note: uses nanmean so if one side is nan and the other is finite, we keep the finite value
    neg180_idx = np.where(np.isclose(lon_vals, -180.0, atol=1e-12))[0]
    pos180_idx = np.where(np.isclose(lon_vals, 180.0, atol=1e-12))[0]
    if neg180_idx.size and pos180_idx.size:
        i0 = int(neg180_idx[0])
        i1 = int(pos180_idx[0])
        merged = np.nanmean(np.stack([grid[:, i0], grid[:, i1]], axis=1), axis=1)
        grid[:, i0] = merged
        grid = np.delete(grid, i1, axis=1)
        lon_vals = np.delete(lon_vals, i1)


    # basic range checks
    if not (np.isfinite(lat_vals).all() and np.isfinite(lon_vals).all()):
        raise ValueError(f"{path}: non-finite lat/lon encountered")
    if lat_vals.min() < -90 - 1e-6 or lat_vals.max() > 90 + 1e-6:
        raise ValueError(f"{path}: latitude range outside [-90, 90]")
    if lon_vals.min() < -180 - 1e-6 or lon_vals.max() > 180 + 1e-6:
        raise ValueError(f"{path}: longitude range outside [-180, 180]")

    return lat_vals, lon_vals, grid


def collect_files(indir: Path, modelname: str) -> Dict[str, Dict[float, Path]]:
    """
    scan directory and collect files of interest for the given model:
        { 'repar': {depth_km: Path, ...}, 'filt': {depth_km: Path, ...} }
    """
    out: Dict[str, Dict[float, Path]] = {"repar": {}, "filt": {}}
    for p in indir.iterdir():
        if not p.is_file() or not p.name.endswith(".dat"):
            continue
        parsed = parse_fname(p)
        if parsed is None:
            continue
        model, typ, depth_km = parsed
        if model != modelname:
            continue
        out.setdefault(typ, {})[depth_km] = p
    return out


def build_dataset(
    files_by_type: Dict[str, Dict[float, Path]], earth_radius_km: float
) -> xr.Dataset:
    """
    read all available depths and assemble an xarray Dataset with dims (r, lat, lon),
    coordinates:
        r [m], depth [km], lat [deg_north], lon [deg_east]
    data variables (subset depending on availability):
        dVs_reparam_percent (r, lat, lon)
        dVs_tofi_percent    (r, lat, lon)
    """
    # union of depths across both types, sorted ascending by depth (km)
    all_depths = sorted(
        set(files_by_type.get("repar", {})) | set(files_by_type.get("filt", {}))
    )
    if not all_depths:
        raise FileNotFoundError(
            "no matching repar/filt files found for the requested model"
        )

    # establish reference lat/lon from the first available file
    first_path = None
    for typ in ("repar", "filt"):
        if files_by_type.get(typ):
            first_path = next(iter(files_by_type[typ].values()))
            break
    assert first_path is not None
    ref_lat, ref_lon, _ = load_grid_from_file(first_path)

    # verify all files share identical lat/lon; while reading, stash 2d grids per type
    grids_by_type: Dict[str, List[Optional[np.ndarray]]] = {"repar": [], "filt": []}
    for depth in all_depths:
        for typ in ("repar", "filt"):
            p = files_by_type.get(typ, {}).get(depth)
            if p is None:
                grids_by_type[typ].append(None)  # placeholder for this depth
                continue
            lat, lon, g = load_grid_from_file(p)
            if not (np.array_equal(lat, ref_lat) and np.array_equal(lon, ref_lon)):
                raise ValueError(
                    f"{p}: lat/lon grid differs from reference; mixing grids is not supported"
                )
            grids_by_type[typ].append(g)

    nlat = ref_lat.size
    nlon = ref_lon.size
    ndep = len(all_depths)

    # allocate arrays (ndep, nlat, nlon) and fill with nan where missing
    def stack_values(grids: List[Optional[np.ndarray]]) -> np.ndarray:
        arr = np.full((ndep, nlat, nlon), np.nan, dtype=np.float32)
        for i, g in enumerate(grids):
            if g is not None:
                if g.shape != (nlat, nlon):
                    raise ValueError("inconsistent grid shape encountered")
                arr[i, :, :] = g.astype(np.float32, copy=False)
        return arr

    data_vars = {}
    if any(g is not None for g in grids_by_type["repar"]):
        data_vars["dVs_reparam_percent"] = (
            ("r", "lat", "lon"),
            stack_values(grids_by_type["repar"]),
        )
    if any(g is not None for g in grids_by_type["filt"]):
        data_vars["dVs_tofi_percent"] = (
            ("r", "lat", "lon"),
            stack_values(grids_by_type["filt"]),
        )
    if not data_vars:
        raise FileNotFoundError("no data arrays could be constructed (all missing?)")

    depth_km = np.asarray(all_depths, dtype=np.float64)
    r_m = earth_radius_km * 1000.0 - depth_km * 1000.0

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "r": ("r", r_m.astype(np.float64)),
            "depth": ("r", depth_km),  # attached along the same dimension as r
            "lat": ("lat", ref_lat.astype(np.float64)),
            "lon": ("lon", ref_lon.astype(np.float64)),
        },
        attrs={
            "id": "Firedrake model filtered by S40RTS",
        },
    )

    # add CF-like attributes
    ds["r"].attrs.update({"units": r"\m", "long_name": "radius", "positive": "up"})
    ds["depth"].attrs.update(
        {"units": r"\km", "long_name": "depth", "positive": "down"}
    )
    ds["lat"].attrs.update({"units": r"\degrees", "long_name": "latitude"})
    ds["lon"].attrs.update({"units": r"\degrees_east", "long_name": "longitude"})
    if "dVs_reparam_percent" in ds:
        ds["dVs_reparam_percent"].attrs.update(
            {"units": r"\percent", "long_name": "Shear-wave velocity perturbation"}
        )
    if "dVs_tofi_percent" in ds:
        ds["dVs_tofi_percent"].attrs.update(
            {"units": r"\percent", "long_name": "Shear-wave velocity perturbation"}
        )

    return ds


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not args.indir.is_dir():
        raise NotADirectoryError(f"input directory not found: {args.indir}")

    files = collect_files(args.indir, args.modelname)
    ds = build_dataset(files, earth_radius_km=args.earth_radius_km)

    # set compression/chunking; keep dims in the required order
    encoding = {}
    for v in ds.data_vars:
        encoding[v] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "chunksizes": (
                min(32, ds.sizes["r"]),
                min(256, ds.sizes["lat"]),
                min(256, ds.sizes["lon"]),
            ),
        }

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.outfile, format="NETCDF4", encoding=encoding)
    print(f"wrote {args.outfile} with variables: {', '.join(ds.data_vars)}")


if __name__ == "__main__":
    main()
