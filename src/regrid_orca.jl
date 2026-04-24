using NCDatasets

using Oceananigans.Architectures: architecture
using NumericalEarth.DataWrangling

using NumericalEarth.DataWrangling: native_grid, metadata_path

import Base: size
import NumericalEarth.Bathymetry: _regrid_bathymetry
import NumericalEarth.DataWrangling: z_interfaces
import NumericalEarth.DataWrangling.ORCA: ORCA1Metadatum

Base.size(::ORCA1) = (362, 298, 1)
Base.size(dataset::ORCA1, variable) = size(dataset)

NumericalEarth.DataWrangling.z_interfaces(::NumericalEarth.DataWrangling.ORCA.ORCA1Metadatum) = (0, 1)

function _regrid_bathymetry(target_grid, metadata::ORCA1Metadatum;
                            height_above_water,
                            minimum_depth,
                            interpolation_passes,
                            major_basins)
    if isinteger(interpolation_passes)
        interpolation_passes = convert(Int, interpolation_passes)
    end

    if interpolation_passes isa Nothing || !isa(interpolation_passes, Int) || interpolation_passes ≤ 0
        return throw(ArgumentError("interpolation_passes has to be an integer ≥ 1"))
    end

    arch = architecture(target_grid)

    bathymetry_native_grid = native_grid(metadata, arch; halo = (10, 10, 1))
    FT = eltype(target_grid)

    filepath = metadata_path(metadata)
    dataset = Dataset(filepath, "r")

    z_data = -convert(Array{FT}, dataset["Bathymetry"][:, :])
    close(dataset)

    if !isnothing(height_above_water)
        # Overwrite the height of cells above water.
        # This has an impact on reconstruction. Greater height_above_water reduces total
        # wet area by biasing coastal regions to land during bathymetry regridding.
        land = z_data .> 0
        z_data[land] .= height_above_water
    end

    native_z = Field{Center, Center, Nothing}(bathymetry_native_grid)
    set!(native_z, z_data[:, 35:end])
    fill_halo_regions!(native_z)

    target_z = interpolate_bathymetry_in_passes(native_z, target_grid;
                                                passes = interpolation_passes)

    if minimum_depth > 0
        launch!(arch, target_grid, :xy, _enforce_minimum_depth!, target_z, minimum_depth)
    end

    if major_basins < Inf
        remove_minor_basins!(target_z, major_basins)
    end

    fill_halo_regions!(target_z)

    return target_z
end