module GlobalOceanBioME

export wind_from_atmosphere

# TODO:
# - light attenuation under ice
# - ice nutrient supply

include("grids.jl")
include("gas_exchange.jl") # move to OceanBioME
include("rivers.jl") # move to NumericalEarth
include("regrid_bathymetry.jl")
include("salinity_nudging.jl")
include("light_attenuation.jl")

end # module GlobalOceanBioME
