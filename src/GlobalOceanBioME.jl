module GlobalOceanBioME

export wind_from_atmosphere,
       DustCOMM_iron_deposition_boundary_condition,
       PAR_from_radiation

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

end # module GlobalOceanBioME
