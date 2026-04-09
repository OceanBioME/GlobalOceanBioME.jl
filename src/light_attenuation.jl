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