import argparse

import numpy as np
import pyvista as pv
import xarray as xr
import gdrift


def interpolate(
    filename: str,
    *,
    radii: tuple[float, float],
    cylindrical: bool,
    dims: tuple[int, ...],
    QN: str,
):
    if cylindrical:
        nx, nz = dims
        ny = 1
        grid_shape = nz, nx
    else:
        nx, ny, nz = dims
        grid_shape = nz, nx, ny

    r_min, r_max = radii

    theta = np.linspace(-180, 180, nx + 1)[:-1]
    phi = np.linspace(0, 180, ny)
    r = np.linspace(r_min, r_max, nz)

    x, y, z = np.meshgrid(np.radians(theta), 0 if cylindrical else np.radians(phi), r)
    phi_factor = 1 if cylindrical else np.sin(y)

    x_cart = z * np.cos(x) * phi_factor
    y_cart = z * np.sin(x) * phi_factor

    if cylindrical:
        z_cart = y
    else:
        z_cart = z * np.cos(y)

    cylinder_grid = pv.StructuredGrid(x_cart, y_cart, z_cart)

    g = pv.read(filename).clean()
    # either interpolate for weighted, or sample for direct
    sampled = cylinder_grid.sample(g)

    coords = {"lon": theta, "r": r}
    if cylindrical:
        coord_list = ("r", "lon")
    else:
        coord_list = ("r", "lon", "lat")
        coords["lat"] = 90 - phi

    ds = xr.Dataset(coords=coords)

    for name, data in sampled.point_data.items():
        # skip info arrays
        if name.startswith("vtk") and name != "vtkValidPointMask":
            continue

        if len(data.shape) > 1:
            # vector data
            for suffix, component in zip("xyz", data.T):
                ds[f"{name}_{suffix}"] = (coord_list, component.reshape(grid_shape))

        else:
            ds[name] = (coord_list, data.reshape(grid_shape))

    # assign attributes to coordinates
    ds["lon"].assign_attrs(
        {"long_name": "Longitude", "units": r"\degree", "convention": "bipolar"}
    )

    ds = ds.assign_coords(r=ds["r"] * 6371e3 / 2.22)
    ds["r"].assign_attrs({"long_name": "Radius", "units": r"\metre", "positive": "up"})

    if not cylindrical:
        ds["lat"].assign_attrs({"long_name": "Latitude", "units": r"\degree"})

    # add depth as a secondary coordinate for radial dimension and assign attributes
    depths = 6371 - ds["r"].data * 1e-3
    ds = ds.assign_coords(depth=("r", depths))
    ds["depth"].assign_attrs(
        {"long_name": "Depth", "units": r"\kilo\metre", "positive": "down"}
    )

    # calculate dimensional temperature, perturbation
    ds["T"] = ds["FullTemperature_CG"] * 3700 + 300
    ds["dT"] = ds["Temperature_Deviation_CG"] * (ds["T"].max() - ds["T"].min())
    ds["T_av"] = ds["T"] - ds["dT"]

    # build elastic seismic model and an anelastic corrections
    # Load PREM
    prem = gdrift.PreliminaryRefEarthModel()

    # initialise thermodynamic model
    slb_pyrolite = gdrift.ThermodynamicModel(
        "SLB_16",
        "pyrolite",
        temps=np.linspace(300, 4000),
        depths=np.linspace(0, 2890e3),
    )

    # A temperautre profile representing the mantle average temperature
    # This is used to anchor the regularised thermodynamic table (we make sure the seismic speeds are the same at those temperature for the regularised and unregularised table)
    temperature_spline = gdrift.SplineProfile(
        depth=np.asarray([0.0, 500e3, 2700e3, 3000e3]),
        value=np.asarray([300, 1000, 3000, 4000]),
    )

    # regularising the table
    # regularisation works by saturating the minimum and maximum of variable gradients with respect to temperature.
    # default values are between -inf and 0.0; which essentialy prohibits phase jumps that would otherwise render
    # v_s/v_p/rho versus temperature non-unique.
    linear_slb_pyrolite = gdrift.mineralogy.regularise_thermodynamic_table(
        slb_pyrolite,
        temperature_spline,
        regular_range={"v_s": [-0.5, 0], "v_p": [-0.5, 0.0], "rho": [-0.5, 0.0]},
    )

    # initialise anelasticity model
    anelasticity = gdrift.CammaranoAnelasticityModel.from_q_profile(
        QN
    )  # instantiate the anelasticity model
    # apply anelastic correction to the unlinearised thermodynamic model
    anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        slb_pyrolite, anelasticity
    )

    # apply anelastic correction to the linearised model
    linear_anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        linear_slb_pyrolite, anelasticity
    )

    # calculate body and shear wave velocities, perturbations
    ds["Vp"] = (
        ds["T"].dims,
        slb_pyrolite.temperature_to_vp(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vp_av"] = (
        ds["T_av"].dims,
        slb_pyrolite.temperature_to_vp(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVp"] = ds["Vp"] - ds["Vp_av"]
    ds["dVp_percent"] = 100 * ds["dVp"] / ds["Vp_av"]
    ds["Vs"] = (
        ds["T"].dims,
        slb_pyrolite.temperature_to_vs(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vs_av"] = (
        ds["T_av"].dims,
        slb_pyrolite.temperature_to_vs(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVs"] = ds["Vs"] - ds["Vs_av"]
    ds["dVs_percent"] = 100 * ds["dVs"] / ds["Vs_av"]

    # calculate anelastic corrected compression and shear wave velocities, perturbations
    ds["Vp_an"] = (
        ds["T"].dims,
        anelastic_slb_pyrolite.temperature_to_vp(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vp_an_av"] = (
        ds["T_av"].dims,
        anelastic_slb_pyrolite.temperature_to_vp(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVp_an"] = ds["Vp_an"] - ds["Vp_an_av"]
    ds["dVp_an_percent"] = 100 * ds["dVp_an"] / ds["Vp_an_av"]
    ds["Vs_an"] = (
        ds["T"].dims,
        anelastic_slb_pyrolite.temperature_to_vs(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vs_an_av"] = (
        ds["T_av"].dims,
        anelastic_slb_pyrolite.temperature_to_vs(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVs_an"] = ds["Vs_an"] - ds["Vs_an_av"]
    ds["dVs_an_percent"] = 100 * ds["dVs_an"] / ds["Vs_an_av"]

    # calculate linearised shear wave velocities, perturbations
    ds["Vp_lin"] = (
        ds["T"].dims,
        linear_slb_pyrolite.temperature_to_vp(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vp_lin_av"] = (
        ds["T_av"].dims,
        linear_slb_pyrolite.temperature_to_vp(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVp_lin"] = ds["Vp_lin"] - ds["Vp_lin_av"]
    ds["dVp_lin_percent"] = 100 * ds["dVp_lin"] / ds["Vp_lin_av"]
    ds["Vs_lin"] = (
        ds["T"].dims,
        linear_slb_pyrolite.temperature_to_vs(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vs_lin_av"] = (
        ds["T_av"].dims,
        linear_slb_pyrolite.temperature_to_vs(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVs_lin"] = ds["Vs_lin"] - ds["Vs_lin_av"]
    ds["dVs_lin_percent"] = 100 * ds["dVs_lin"] / ds["Vs_lin_av"]

    # linearise anelastic compression and shear wave velocities, perturbations
    ds["Vp_linan"] = (
        ds["T"].dims,
        linear_anelastic_slb_pyrolite.temperature_to_vp(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vp_linan_av"] = (
        ds["T"].dims,
        linear_anelastic_slb_pyrolite.temperature_to_vp(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVp_linan"] = ds["Vp_linan"] - ds["Vp_linan_av"]
    ds["dVp_linan_percent"] = 100 * ds["dVp_linan"] / ds["Vp_linan_av"]
    ds["Vs_linan"] = (
        ds["T"].dims,
        linear_anelastic_slb_pyrolite.temperature_to_vs(
            ds["T"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["Vs_linan_av"] = (
        ds["T"].dims,
        linear_anelastic_slb_pyrolite.temperature_to_vs(
            ds["T_av"].to_numpy(), ds["depth"].broadcast_like(ds["T"]).to_numpy() * 1e3
        ),
    )
    ds["dVs_linan"] = ds["Vs_linan"] - ds["Vs_linan_av"]
    ds["dVs_linan_percent"] = 100 * ds["dVs_linan"] / ds["Vs_linan_av"]

    # drop unneeded `DataArray`s from the dataset
    vars_to_keep = [
        "T",
        "dT",
        "Vp",
        "dVp",
        "dVp_percent",
        "Vp_lin",
        "dVp_lin",
        "dVp_lin_percent",
        "Vp_an",
        "dVp_an",
        "dVp_an_percent",
        "Vp_linan",
        "dVp_linan",
        "dVp_linan_percent",
        "Vs",
        "dVs",
        "dVs_percent",
        "Vs_lin",
        "dVs_lin",
        "dVs_lin_percent",
        "Vs_an",
        "dVs_an",
        "dVs_an_percent",
        "Vs_linan",
        "dVs_linan",
        "dVs_linan_percent",
    ]
    ds = ds.drop_vars([var for var in ds.data_vars if var not in vars_to_keep])

    # assign attributes to remaining `DataArray`s
    # temperature
    ds["T"] = ds["T"].assign_attrs({"long_name": "Temperature", "units": r"\kelvin"})
    ds["dT"] = ds["dT"].assign_attrs(
        {"long_name": "Temperature perturbation", "units": r"\kelvin"}
    )
    # compression wave speeds
    ds["Vp"] = ds["Vp"].assign_attrs(
        {"long_name": "Compression wave velocity", "units": r"\metre\per\second"}
    )
    ds["dVp"] = ds["dVp"].assign_attrs(
        {
            "long_name": "Compression wave velocity perturbation",
            "units": r"\metre\per\second",
        }
    )
    ds["dVp_percent"] = ds["dVp_percent"].assign_attrs(
        {"long_name": "Compression wave velocity perturbation", "units": r"\percent"}
    )
    # linearised compression wave speeds
    ds["Vp_lin"] = ds["Vp_lin"].assign_attrs(
        {
            "long_name": "Linearised compression wave velocity",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_lin"] = ds["dVp_lin"].assign_attrs(
        {
            "long_name": "Linearised compression wave velocity perturbation",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_lin_percent"] = ds["dVp_lin_percent"].assign_attrs(
        {
            "long_name": "Linearised compression wave velocity perturbation",
            "units": r"\percent",
        }
    )
    # anelastic compression wave speeds
    ds["Vp_an"] = ds["Vp_an"].assign_attrs(
        {
            "long_name": "Anelastic compression wave velocity",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_an"] = ds["dVp_an"].assign_attrs(
        {
            "long_name": "Anelastic compression wave velocity perturbation",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_an_percent"] = ds["dVp_an_percent"].assign_attrs(
        {
            "long_name": "Anelastic compression wave velocity perturbation",
            "units": r"\percent",
        }
    )
    # linear anelastic compression wave speeds
    ds["Vp_linan"] = ds["Vp_linan"].assign_attrs(
        {
            "long_name": "Linearised anelastic compression wave velocity",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_linan"] = ds["dVp_linan"].assign_attrs(
        {
            "long_name": "Linearised anelastic compression wave velocity perturbation",
            "units": r"\meter\per\second",
        }
    )
    ds["dVp_linan_percent"] = ds["dVp_linan_percent"].assign_attrs(
        {
            "long_name": "Linearised anelastic compression wave velocity perturbation",
            "units": r"\percent",
        }
    )
    # shear wave speeds
    ds["Vs"] = ds["Vs"].assign_attrs(
        {"long_name": "Shear wave velocity", "units": r"\metre\per\second"}
    )
    ds["dVs"] = ds["dVs"].assign_attrs(
        {"long_name": "Shear wave velocity perturbation", "units": r"\metre\per\second"}
    )
    ds["dVs_percent"] = ds["dVs_percent"].assign_attrs(
        {"long_name": "Shear wave velocity perturbation", "units": r"\percent"}
    )
    # Linearised shear wave speeds
    ds["Vs_lin"] = ds["Vs_lin"].assign_attrs(
        {"long_name": "Linearised shear wave velocity", "units": r"\metre\per\second"}
    )
    ds["dVs_lin"] = ds["dVs_lin"].assign_attrs(
        {
            "long_name": "Linearised shear wave velocity perturbation",
            "units": r"\metre\per\second",
        }
    )
    ds["dVs_lin_percent"] = ds["dVs_lin_percent"].assign_attrs(
        {
            "long_name": "Linearised shear wave velocity perturbation",
            "units": r"\percent",
        }
    )
    # anelastic shear wave speeds
    ds["Vs_an"] = ds["Vs_an"].assign_attrs(
        {"long_name": "Anelastic shear wave velocity", "units": r"\metre\per\second"}
    )
    ds["dVs_an"] = ds["dVs_an"].assign_attrs(
        {
            "long_name": "Anelastic shear wave velocity perturbation",
            "units": r"\metre\per\second",
        }
    )
    ds["dVs_an_percent"] = ds["dVs_an_percent"].assign_attrs(
        {
            "long_name": "Anelastic shear wave velocity perturbation",
            "units": r"\percent",
        }
    )
    # linearised anelastic shear wave speeds
    ds["Vs_linan"] = ds["Vs_linan"].assign_attrs(
        {
            "long_name": "Linearised anelastic shear wave velocity",
            "units": r"\metre\per\second",
        }
    )
    ds["dVs_linan"] = ds["dVs_linan"].assign_attrs(
        {
            "long_name": "Linearised anelastic shear wave velocity perturbation",
            "units": r"\metre\per\second",
        }
    )
    ds["dVs_linan_percent"] = ds["dVs_linan_percent"].assign_attrs(
        {
            "long_name": "Linearised anelastic shear wave velocity perturbation",
            "units": r"\percent",
        }
    )

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="interp",
        description="Interpolate from unstructured VTK grids to netCDF",
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-r",
        "--radii",
        required=True,
        help="Comma-separated minimum and maximum radii for the grid",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="interp.nc",
        help="Output filename, defaults to interp.nc",
    )
    parser.add_argument(
        "-d",
        "--dims",
        required=True,
        help="Comma-separated list of dimensions; cylindrical: (nx,nz) or spherical: (nx,ny,nz)",
    )
    parser.add_argument(
        "-a",
        "--anelasticity",
        required=True,
        help="String for Cammarano model between Q1 and Q6 to choose anelasticity model parameters",
    )

    grid_group = parser.add_mutually_exclusive_group(required=True)
    grid_group.add_argument(
        "-c",
        "--cylindrical",
        action="store_true",
        help="Grid is a cylindrical annulus (2D)",
    )
    grid_group.add_argument(
        "-s",
        "--spherical",
        action="store_false",
        help="Grid is 3D spherical",
    )

    args = parser.parse_args()

    radii = [float(s) for s in args.radii.split(",")]
    dims = [int(s) for s in args.dims.split(",")]

    ds = interpolate(
        args.filename,
        radii=radii,
        cylindrical=args.cylindrical,
        dims=dims,
        QN=args.anelasticity,
    )
    ds.to_netcdf(args.output)
