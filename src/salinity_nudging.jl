using OceanBioME, Oceananigans, NumericalEarth, WorldOceanAtlasTools, Oceananigans.Units, CUDA

using Oceananigans: instantiated_location
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: instantiated_location, interpolate
using Oceananigans.Grids: AbstractGrid, node
using Oceananigans.OutputReaders: FlavorOfFTS, Cyclical
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!
using Oceananigans.Operators: Δzᶜᶜᶜ
using NumericalEarth.DataWrangling: NearestNeighborInpainting, download_dataset

using KernelAbstractions: @kernel, @index


@inline function salinity_nudging(i, j, k, grid, clock, model_fields, parameters)
    Sbf, τ = parameters

    # would branching this not be faster rather than interpolating the fts every time?

    S  = @inbounds model_fields.S[i, j, k]
    Sb = @inbounds Sbf[i, j, 1, Time(clock.time)]

    return ifelse(k == grid.Nz, τ * (Sb - S), zero(grid))
end

function salinity_nudging(grid;
                          dataset = WOAMonthly(),
                          dir = "",
                          piston_velocity = 1/6,
                          time_indices_in_memory = 2, # Not more than this if we want to use GPU! ???
                          time_indexing = Cyclical(),
                          inpainting = NearestNeighborInpainting(Inf),
                          cache_inpainted_data = true)

    metadata = Metadata(:salinity; dir, dataset)

    download_dataset(metadata)

    fts_native = FieldTimeSeries(metadata, Oceananigans.Architectures.architecture(grid);
                                 time_indices_in_memory,
                                 time_indexing,
                                 inpainting,
                                 cache_inpainted_data)

    fts = FieldTimeSeries((Center(), Center(), Center()), grid, fts_native.times;
                          time_indexing, indices = (:, :, 1))

    for n in 1:12
        interpolate_surface!(fts[n], fts_native[n])
    end

    Δzˢ = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, grid.Nz, grid)

    τ = (piston_velocity / (Δzˢ * days))

    return Forcing(salinity_nudging, discrete_form = true, parameters = (fts, τ))
end

function interpolate_surface!(target, source)
    grid = target.grid

    Oceananigans.Utils.launch!(architecture(grid), grid, :xy, _interpolate_surface!, target, source, target.grid, source.grid)

    return nothing
end

@kernel function _interpolate_surface!(target, source, target_grid, source_grid)
    i, j = @index(Global, NTuple)

    target_loc = instantiated_location(target)
    source_loc = instantiated_location(source)

    X = node(i, j, target_grid.Nz, target_grid, target_loc...)

    @inbounds target[i, j, 1] = interpolate(X, source, source_loc, source_grid)
end