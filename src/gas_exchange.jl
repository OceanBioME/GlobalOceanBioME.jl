# TODO: move to OceanBioME
using OceanBioME, Oceananigans
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Units: Time
import OceanBioME.Models.GasExchangeModel: surface_value

wind_from_atmosphere(atmosphere) = (values(atmosphere.velocities)..., atmosphere.grid)

@inline function OceanBioME.Models.GasExchangeModel.surface_value(f::Tuple{FlavorOfFTS, FlavorOfFTS, <:AbstractGrid}, i, j, grid, clock, args...)
    uf, vf, fgrid = f

    to_time = Time(clock.time)
    target_node = Oceananigans.Grids._node(i, j, grid.Nz, grid, Center(), Center(), Center())

    u = Oceananigans.Fields.interpolate(target_node, to_time, uf, instantiated_location(uf), fgrid)
    v = Oceananigans.Fields.interpolate(target_node, to_time, vf, instantiated_location(vf), fgrid)

    return sqrt(u^2 + v^2)
end