module GlobalOceanBioME

export wind_from_atmosphere,
       DustCOMM_iron_deposition_boundary_condition,
       PAR_from_atmosphere

using Adapt

import Adapt: adapt_structure

# TODO:
# - fix rivers??
# - light attenuation under ice
# - gas exchange blocking by ice
# - ice nutrient supply: 15 nmolFe/L according to PISCES - 0.024 Gmol Fe yr−1 total supply, not sure this is valid see *

# * they assume constant concentration in sea ice based and that ice takes up iron when the water freezes and releases it when it melts
# so any excess must be from excess ice mass, presumably from precipitation, but where is the iron coming from in that case?
# Think I'll leave this source for now


include("grids.jl")
include("gas_exchange.jl") # move to OceanBioME
include("rivers.jl") # move NEWS to NumericalEarth
include("regrid_bathymetry.jl")
include("salinity_nudging.jl")
include("light_attenuation.jl")
include("dust.jl") # move DustCOMM to NumericalEarth
include("exchange_state.jl")

# temporary fix
using Oceananigans: CPU

using NumericalEarth.DataWrangling: default_time_indices_in_memory, 
                                    NearestNeighborInpainting, 
                                    oceananigans_fieldnames,
                                    Metadata

using Oceananigans.Architectures: on_architecture,  
                                  architecture,
                                  AbstractArchitecture

using Oceananigans.OutputReaders: FieldTimeSeries, Cyclical 

import NumericalEarth.DataWrangling: DatasetRestoring
import Base: getindex

struct OceanBioMEAlk end

@inline Base.getindex(fields, i, j, k, ::OceanBioMEAlk) = @inbounds fields.Alk[i, j, k]

function NumericalEarth.DataWrangling.DatasetRestoring(
        metadata::Metadata,
        arch_or_grid = CPU();
        rate,
        mask = 1,
        time_indices_in_memory = default_time_indices_in_memory(metadata),
        time_indexing = Cyclical(),
        inpainting = NearestNeighborInpainting(Inf),
        cache_inpainted_data = true,
        variable_name = metadata.name,
        field_name = oceananigans_fieldnames[variable_name]
    )

    download_dataset(metadata)

    fts = FieldTimeSeries(metadata, arch_or_grid;
                          time_indices_in_memory,
                          time_indexing,
                          inpainting,
                          cache_inpainted_data)

    arch = architecture(fts)
    mask = on_architecture(arch, mask)

    # If we pass the grid we do not need to interpolate
    # so we can save parameter space by setting the native grid to nothing
    on_native_grid = arch_or_grid isa AbstractArchitecture
    maybe_native_grid = on_native_grid ? fts.grid : nothing

    return DatasetRestoring(fts, maybe_native_grid, mask, field_name, rate)
end

end # module GlobalOceanBioME
