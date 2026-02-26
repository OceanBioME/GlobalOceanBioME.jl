# TODO: move to OceanBioME
using OceanBioME, Oceananigans
using Oceananigans.OutputReaders: FlavorOfFTS
import OceanBioME.Models.GasExchangeModel: surface_value

@inline function OceanBioME.Models.GasExchangeModel.surface_value(f::Tuple{FlavorOfFTS, FlavorOfFTS}, i, j, grid, clock, args...)
    uf, vf = f

    to_time = Time(clock.time)
    target_node = Oceananigans.Grids._node(i, j, grid.Nz, grid, Center(), Center(), Center())

    u = Oceananigans.Fields.interpolate(target_node, to_time, uf, instantiated_location(uf), grid)
    v = Oceananigans.Fields.interpolate(target_node, to_time, vf, instantiated_location(vf), grid)

    return sqrt(u^2 + v^2)
end