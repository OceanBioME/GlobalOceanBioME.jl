module GlobalOceanBioME

export wind_from_atmosphere,
       DustCOMM_iron_deposition_boundary_condition,
       PAR_from_atmosphere

# TODO:
# - light attenuation under ice
# - ice nutrient supply

include("grids.jl")
include("gas_exchange.jl") # move to OceanBioME
include("rivers.jl") # move NEWS to NumericalEarth
include("regrid_bathymetry.jl")
include("salinity_nudging.jl")
include("light_attenuation.jl")
include("dust.jl") # move DustCOMM to NumericalEarth

end # module GlobalOceanBioME
