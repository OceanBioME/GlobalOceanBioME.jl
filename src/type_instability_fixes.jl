using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: 
    getclosure, 
    top_buoyancy_flux,
    dissipation_length_scaleᶜᶜᶜ

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

using KernelAbstractions: @index, @kernel

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: 
    time_step_catke_equation!,
    compute_CATKE_diffusivities!,
    FlavorOfCATKE,
    get_top_tracer_bcs,
    update_previous_compute_time!

import Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: 
    compute_diffusivities!


function compute_diffusivities!(diffusivities, closure::FlavorOfCATKE, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = get_top_tracer_bcs(model.buoyancy.formulation, tracers)
    Δt = update_previous_compute_time!(diffusivities, model)

    if isfinite(model.clock.last_Δt) # Check that we have taken a valid time-step first.
        # Compute e at the current time:
        #   * update tendency Gⁿ using current and previous velocity field
        #   * use tridiagonal solve to take an implicit step
        time_step_catke_equation!(model, model.timestepper)
    end

    # Update "previous velocities"
    u, v, w = model.velocities
    u⁻, v⁻ = diffusivities.previous_velocities
    parent(u⁻) .= parent(u)
    parent(v⁻) .= parent(v)

    launch!(arch, grid, :xy,
            compute_average_surface_buoyancy_flux_fixed!,
            diffusivities.Jᵇ, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock, Δt)

    launch!(arch, grid, parameters,
            compute_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy)

    return nothing
end

@kernel function compute_average_surface_buoyancy_flux_fixed!(Jᵇ, grid, closure, velocities, tracers,
                                                              buoyancy, top_tracer_bcs, clock, Δt)
    i, j = @index(Global, NTuple)
    k = grid.Nz
    FT = eltype(grid)

    closure = getclosure(i, j, closure)

    model_fields = merge(velocities, tracers)
    Jᵇ★ = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, model_fields)
    ℓᴰ = dissipation_length_scaleᶜᶜᶜ(i, j, k, grid, closure, velocities, tracers, buoyancy, Jᵇ)

    Jᵇᵋ = closure.minimum_convective_buoyancy_flux
    Jᵇᵢⱼ = @inbounds Jᵇ[i, j, 1]
    Jᵇ⁺ = max(Jᵇᵋ, Jᵇᵢⱼ, Jᵇ★) # selects fastest (dominant) time-scale
    t★ = (ℓᴰ^2 / Jᵇ⁺)^FT(1/3)
    ϵ = Δt / t★

    @inbounds Jᵇ[i, j, 1] = (Jᵇᵢⱼ + ϵ * Jᵇ★) / (one(Δt) + ϵ)
end


##### Metadata fields
using ClimaOcean.DataWrangling: 
    metadata_path,
    Metadatum,
    dataset_variable_name,
    Dataset,
    is_three_dimensional,
    reversed_vertical_axis


import ClimaOcean.DataWrangling: retrieve_data

function retrieve_data(metadata::Metadatum)
    path = metadata_path(metadata)
    name = dataset_variable_name(metadata)
    
    # NetCDF shenanigans
    ds = Dataset(path)

    if is_three_dimensional(metadata)
        data = ds[name][:, :, :, 1]

        # Many ocean datasets use a "depth convention" for their vertical axis
        if reversed_vertical_axis(metadata.dataset)
            data = reverse(data, dims=3)
        end
    else
        data = ds[name][:, :, 1]
    end        

    close(ds)
    return convert.(eltype(metadata), data)
end