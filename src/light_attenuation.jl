using Adapt
using Oceananigans.Fields: interpolate, Center
using Oceananigans.Units: Time

import Adapt: adapt_structure

@kwdef struct PARfromFTS{SW, GR, LO, PF} <: Function
                shortwave :: SW
                     grid :: GR
                 location :: LO
             PAR_fraction :: PF = 0.45
end

@inline function (par::PARfromFTS)(x, y, t)
    η = par.PAR_fraction 

    SS = par.shortwave

    I =  interpolate((x, y, zero(x)), 
                     Time(t), 
                     SS.data, 
                     par.location, 
                     par.grid,
                     SS.times,
                     SS.backend,
                     SS.time_indexing)

    return η * I
end

Adapt.adapt_structure(to, par::PARfromFTS) =
    PARfromFTS(adapt(to, par.shortwave),
               adapt(to, par.grid),
               adapt(to, par.location),
               adapt(to, par.PAR_fraction))

PAR_from_atmosphere(atmosphere) = 
    PARfromFTS(; shortwave = atmosphere.downwelling_radiation.shortwave,
                 grid = atmosphere.downwelling_radiation.shortwave.grid,
                 location = (Center(), Center(), nothing))


# TODO: this could significantly change with snow
# This is a stupid workaround because I haven't implemented general functions
# for surface light, should just use the getbc infrastructure
# then we can just pass a different surface_PAR which has the ice factor in it
#=struct IceMaskedSurfaceLight{SL, IT, IC, IA} 
        surface_light :: SL
        ice_thickness :: IT
    ice_concentration :: IC
           ice_albedo :: IA
end

@inline function (isl::IceMaskedSurfaceLight)(i, j, )=#

@kwdef struct IceMaskedLightAttenuation{LA, IT, IC, IA, II}
    underlying_light_attenuation :: LA
                   ice_thickness :: IT
               ice_concentration :: IC
                      ice_albedo :: IA = 0.7
                   ice_impedance :: II = 1.5
end

Adapt.adapt_structure(to, par::IceMaskedLightAttenuation) = 
    adapt(to, par.underlying_light_attenuation)

import Oceananigans.Biogeochemistry: 
    biogeochemical_auxiliary_fields, 
    update_biogeochemical_state!

biogeochemical_auxiliary_fields(par::IceMaskedLightAttenuation) = 
    biogeochemical_auxiliary_fields(par.underlying_light_attenuation)

function update_biogeochemical_state!(model, par::IceMaskedLightAttenuation)
    update_biogeochemical_state!(model, par.underlying_light_attenuation)

    arch = architecture(model.grid)

    par_field = biogeochemical_auxiliary_fields(par.underlying_light_attenuation).PAR

    launch!(arch, model.grid, :xy, apply_ice_impedance!, 
            par_field, 
            par.ice_thickness,
            par.ice_concentration, 
            par.ice_albedo, 
            par.ice_impedance,
            model.grid.Nz)

    return nothing
end

@kernel function apply_ice_impedance!(PAR, ice_thickness, ice_concentration, r, K, Nz)
    i, j = @index(Global, NTuple)

    h = @inbounds ice_thickness[i, j, 1]
    ℵ = @inbounds ice_concentration[i, j, 1]

    impedance = (1 - ℵ) + ℵ * (1 - r) * exp(-K * h)

    @inbounds for k in 1:Nz
        PAR[i, j, k] = PAR[i, j, k] * impedance
    end
end
