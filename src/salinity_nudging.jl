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

@inline get_field(::Val{:S}, model_fields, i, j, k) = @inbounds model_fields.S[i, j, k]
@inline get_field(::Val{:T}, model_fields, i, j, k) = @inbounds model_fields.T[i, j, k]
@inline get_field(::Val{:Alk}, model_fields, i, j, k) = @inbounds model_fields.Alk[i, j, k]

@inline function nudging(i, j, k, grid, clock, model_fields, parameters)
    Bf, τ, name, sf = parameters

    # would branching this not be faster rather than interpolating the fts every time?
    F = get_field(name, model_fields, i, j, k)
    B = @inbounds Bf[i, j, 1, Time(clock.time)] * sf

    return ifelse(k == grid.Nz, τ * (B - F), zero(grid))
end

function nudging(grid;
                 metadata = Metadata(:salinity; dir = "", dataset = WOAMonthly()),
                 piston_velocity = 1/6,
                 time_indices_in_memory = 1, # Not more than this if we want to use GPU! ???
                 time_indexing = Cyclical(),
                 inpainting = NearestNeighborInpainting(Inf),
                 cache_inpainted_data = true,
                 name = Val(:S))

    download_dataset(metadata)

    fts_native = FieldTimeSeries(metadata, Oceananigans.Architectures.architecture(grid);
                                 time_indices_in_memory,
                                 time_indexing,
                                 inpainting,
                                 cache_inpainted_data)

    fts = FieldTimeSeries((Center(), Center(), Center()), grid, fts_native.times;
                          time_indexing, indices = (:, :, 1))

    for n in 1:length(fts_native.times)
        interpolate_surface!(fts[n], fts_native[n])
    end

    Δzˢ = CUDA.@allowscalar Δzᶜᶜᶜ(1, 1, grid.Nz, grid)

    τ = (piston_velocity / (Δzˢ * days))

    sf = ifelse(name == Val(:Alk), 1000, 1)

    return Forcing(nudging, discrete_form = true, parameters = (fts, τ, name, sf))
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