module GlobalOceanBioME

#include("ocean_simulation.jl")
#include("type_instability_fixes.jl")
include("grids.jl")
include("gas_exchange.jl")
include("rivers.jl")
include("GM_flux_limiters.jl")

end # module GlobalOceanBioME
