using NumericalEarth.EarthSystemModels:
    interpolate_state!,
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_sea_ice_fluxes!,
    compute_sea_ice_ocean_fluxes!,
    update_net_fluxes!

using OceanBioME: CompleteBiogeochemistry

import Oceananigans.TimeSteppers: update_state!

const ESMWithBGC = 
    EarthSystemModel{<:Any, <:Any, 
                     <:Simulation{<:HydrostaticFreeSurfaceModel{<:Any, <:Any, <:Any, <:Any, <:Any, 
                                                                <:Any, <:Any, <:Any, <:Any, <:Any, 
                                                                <:Any, 
                                                                <:OceanBioME.DiscreteBiogeochemistry}}}

function update_state!(coupled_model::ESMWithBGC, callbacks=[])

    # The three components
    ocean      = coupled_model.ocean
    sea_ice    = coupled_model.sea_ice
    atmosphere = coupled_model.atmosphere

    exchanger = coupled_model.interfaces.exchanger
    grid      = exchanger.grid
    
    # This function needs to be specialized to allow different component models
    interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model)
    interpolate_state!(exchanger.ocean,      grid, ocean,      coupled_model)
    interpolate_state!(exchanger.sea_ice,    grid, sea_ice,    coupled_model)

    # Compute interface states
    compute_atmosphere_ocean_fluxes!(coupled_model)
    compute_atmosphere_sea_ice_fluxes!(coupled_model)
    compute_sea_ice_ocean_fluxes!(coupled_model)

    # This function needs to be specialized to allow different component models
    update_net_fluxes!(coupled_model, atmosphere)
    update_net_fluxes!(coupled_model, ocean)
    update_net_fluxes!(coupled_model, sea_ice)

    set!(ocean.model.tracers.DIC.boundary_conditions.top.condition.func.ice_concentration,
         sea_ice.model.ice_concentration)

    set!(ocean.model.biogeochemistry.light_attenuation.ice_concentration,
         sea_ice.model.ice_concentration)
    set!(ocean.model.biogeochemistry.light_attenuation.ice_thickness,
         sea_ice.model.ice_thickness)

    return nothing
end