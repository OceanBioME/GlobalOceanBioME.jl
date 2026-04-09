using CUDA
using NetCDF
using Oceananigans

using Oceananigans.Architectures: architecture, on_architecture

@inline fts_flux(i, j, grid, clock, model_fields, flux) = @inbounds flux[i, j, 1, Oceananigans.Units.Time(clock.time)]

function DustCOMM_iron_deposition_boundary_condition(grid;
                                                     dust_solubility = 0.02,
                                                     dust_iron_content = 0.035,
                                                     dust_fname = "DustCOMM_totdep_Dgr20um_seas_bin.nc")

    arch = architecture(grid)

    total_deposition = ncread(dust_fname, "Total deposition flux")[:, 2, :, :] # kg/m²/yr
    dep_lat = ncread(dust_fname, "lat")
    dep_lon = ncread(dust_fname, "lon")
    dep_times = ncread(dust_fname, "season")
    dep_grid = LatitudeLongitudeGrid(size = (length(dep_lon), length(dep_lat), 1), 
                                    latitude = (minimum(dep_lat), maximum(dep_lat)), 
                                    longitude = (minimum(dep_lon), maximum(dep_lon)), z = (-1, 0))

    iron_deposition = @. dust_solubility * dust_iron_content * total_deposition / 55.8 * 1000 / (365*24*60*60) # mmol Fe/m²/s

    # times are 15/01, 15/04, 15/07, 15/10 ish
    iron_deposition_fts = 
        FieldTimeSeries((Center(), Center(), nothing), 
                        dep_grid, 
                        [1.296e6, 9.1799998272e6, 1.70639996544e7, 2.49479994816e7];
                        time_indexing = Oceananigans.OutputReaders.Cyclical())
    for n in 1:4
        set!(iron_deposition_fts[n], -reshape(iron_deposition[n, :, :], length(dep_lon), length(dep_lat), 1))
    end

    iron_deposition_final = 
        FieldTimeSeries((Center(), Center(), nothing), 
                        on_architecture(Oceananigans.CPU(), grid), 
                        [1.296e6, 9.1799998272e6, 1.70639996544e7, 2.49479994816e7];
                        time_indexing = Oceananigans.OutputReaders.Cyclical())

    for n in 1:4
        Oceananigans.Fields.interpolate!(iron_deposition_final[n], iron_deposition_fts[n])
    end

    CUDA.@allowscalar begin
        iron_deposition_final = on_architecture(arch, iron_deposition_final)
    end 

    return FluxBoundaryCondition(fts_flux; discrete_form = true, parameters = iron_deposition_final)
end