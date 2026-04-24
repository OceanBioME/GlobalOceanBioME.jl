using Oceananigans, KernelAbstractions

using Oceananigans.Operators: Δxᶠᶜᵃ, Δyᶜᶠᵃ

@inline thickness_diffusion(i, j, k, grid, clock, model_fields, κ) =
    diffuse_seaice(i, j, k, grid, clock, model_fields.h, model_fields.h, model_fields.ℵ, κ)

@inline concentration_diffusion(i, j, k, grid, clock, model_fields, κ) =
    diffuse_seaice(i, j, k, grid, clock, model_fields.ℵ, model_fields.h, model_fields.ℵ, κ)

# doing this manually so we refuse to diffuse through the ice edge
@inline function diffuse_seaice(i, j, k, grid, clock, field, h, ℵ, κ)
    @inbounds begin
        f_ij = field[i, j, k]

        # I think this is the correct spacing location?
        east_flux = (field[i-1, j, k] - f_ij) / Δxᶠᶜᵃ(i, j, k, grid)^2 * (h[i-1, j, k] * ℵ[i-1, j, k] > 0.1)
        west_flux = (field[i+1, j, k] - f_ij) / Δxᶠᶜᵃ(i+1, j, k, grid)^2 * (h[i+1, j, k] * ℵ[i+1, j, k] > 0.1)

        south_flux = (field[i, j-1, k] - f_ij) / Δyᶜᶠᵃ(i, j, k, grid)^2 * (h[i, j-1, k] * ℵ[i, j-1, k] > 0.1)
        north_flux = (field[i, j+1, k] - f_ij) / Δyᶜᶠᵃ(i, j+1, k, grid)^2 * (h[i, j+1, k] * ℵ[i, j+1, k] > 0.1)

        return κ * (east_flux + west_flux + south_flux + north_flux) * (h[i, j, k] * ℵ[i, j, k] > 0.1)
    end
end

using Oceananigans.Units
using ClimaSeaIce: PhaseTransitions, ConductiveFlux, PrescribedTemperature, SlabSeaIceThermodynamics, SeaIceModel
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium
using NumericalEarth.SeaIces: sea_ice_dynamics
using NumericalEarth.Oceans: ocean_surface_salinity

import NumericalEarth.SeaIces: sea_ice_simulation

function sea_ice_simulation(grid, ocean, forcing = NamedTuple();
                            Δt = 5minutes,
                            ice_salinity = 4, # psu
                            advection = nothing, # for the moment
                            tracers = (),
                            ice_heat_capacity = 2100, # J kg⁻¹ K⁻¹
                            ice_consolidation_thickness = 0.05, # m
                            ice_density = 900, # kg m⁻³
                            dynamics = sea_ice_dynamics(grid, ocean),
                            bottom_heat_boundary_condition = nothing,
                            top_heat_boundary_condition = nothing,
                            phase_transitions = PhaseTransitions(; ice_heat_capacity, ice_density),
                            conductivity = 2, # kg m s⁻³ K⁻¹
                            internal_heat_flux = ConductiveFlux(; conductivity))

    # Build consistent boundary conditions for the ice model:
    # - bottom -> flux boundary condition
    # - top -> prescribed temperature boundary condition (calculated in the flux computation)

    if isnothing(top_heat_boundary_condition)
        top_surface_temperature = Field{Center, Center, Nothing}(grid)
        top_heat_boundary_condition = PrescribedTemperature(top_surface_temperature.data)
    end

    if isnothing(bottom_heat_boundary_condition)
        if isnothing(ocean)
            surface_ocean_salinity = 0
        else
            kᴺ = size(grid, 3)
            surface_ocean_salinity = ocean_surface_salinity(ocean)
        end
        bottom_heat_boundary_condition = IceWaterThermalEquilibrium(surface_ocean_salinity)
    end

    ice_thermodynamics = SlabSeaIceThermodynamics(grid;
                                                  internal_heat_flux,
                                                  phase_transitions,
                                                  top_heat_boundary_condition,
                                                  bottom_heat_boundary_condition)

    bottom_heat_flux = Field{Center, Center, Nothing}(grid)
    top_heat_flux    = Field{Center, Center, Nothing}(grid)

    # Build the sea ice model
    sea_ice_model = SeaIceModel(grid;
                                ice_salinity,
                                advection,
                                tracers,
                                ice_consolidation_thickness,
                                ice_thermodynamics,
                                dynamics,
                                bottom_heat_flux,
                                top_heat_flux,
                                forcing)

    verbose = false

    # Build the simulation
    sea_ice = Simulation(sea_ice_model; Δt, verbose)

    return sea_ice
end

using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!

using ClimaSeaIce: SIM, horizontal_div_Uc

import ClimaSeaIce: compute_tracer_tendencies!

function compute_tracer_tendencies!(model::SIM)
    grid = model.grid
    arch = architecture(grid)
   
    launch!(arch, grid, :xy,
            _compute_dynamic_tracer_tendencies!,
            model.timestepper.Gⁿ,
            grid,
            model.clock,
            model.velocities,
            model.advection,
            model.ice_thickness,
            model.ice_concentration,
            model.tracers,
            model.forcing)

    return nothing
end

@kernel function _compute_dynamic_tracer_tendencies!(Gⁿ, 
                                                     grid,
                                                     clock,
                                                     velocities,
                                                     advection,
                                                     ice_thickness,
                                                     ice_concentration,
                                                     tracers,
                                                     forcing)

    i, j = @index(Global, NTuple)
    kᴺ   = size(grid, 3) # Assumption! The sea ice is located at the _top_ of the grid

    model_fields = merge(velocities, (h = ice_thickness, ℵ = ice_concentration), tracers)
 
    @inbounds begin
        Gⁿ.h[i, j, 1] = - horizontal_div_Uc(i, j, kᴺ, grid, advection, velocities, ice_thickness) + forcing.h(i, j, kᴺ, grid, clock, model_fields)
        Gⁿ.ℵ[i, j, 1] = - horizontal_div_Uc(i, j, kᴺ, grid, advection, velocities, ice_concentration) + forcing.ℵ(i, j, kᴺ, grid, clock, model_fields)

        # for (n, θ) in enumerate(tracers)
        #     @inbounds Gⁿ[n] = - horizontal_div_Uc(i, j, 1, grid, advection, velocities, θ)
        # end
    end
end