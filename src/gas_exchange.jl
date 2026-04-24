using OceanBioME, Oceananigans
using Oceananigans.Fields: instantiated_location
using Oceananigans.Grids: AbstractGrid
using Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Units: Time
import OceanBioME.Models.GasExchangeModel: surface_value

# TODO: move to OceanBioME maybe?
wind_from_atmosphere(atmosphere) = (values(atmosphere.velocities)..., atmosphere.grid)

@inline function OceanBioME.Models.GasExchangeModel.surface_value(f::Tuple{FlavorOfFTS, FlavorOfFTS, <:AbstractGrid}, i, j, grid, clock, args...)
    uf, vf, fgrid = f

    to_time = Time(clock.time)
    target_node = Oceananigans.Grids._node(i, j, grid.Nz, grid, Center(), Center(), Center())

    u = Oceananigans.Fields.interpolate(target_node, to_time, uf, instantiated_location(uf), fgrid)
    v = Oceananigans.Fields.interpolate(target_node, to_time, vf, instantiated_location(vf), fgrid)

    return sqrt(u^2 + v^2)
end

# NumericalEarth specific:
struct IceMaskedGasExchange{G, I} <: Function
         gas_exchange :: G
    ice_concentration :: I
end

Adapt.adapt_structure(to, par::IceMaskedGasExchange) =
    IceMaskedGasExchange(adapt(to, par.gas_exchange),
                         adapt(to, par.ice_concentration))

function IceMaskedGasExchange(gas_exchange::BoundaryCondition, 
                              ice_concentration)
    gas_exchange_function = gas_exchange.condition.func

    return IceMaskedGasExchange(gas_exchange_function, ice_concentration)
end

function IceMaskedGasExchangeBoundaryCondition(gas_exchange, ice_concentration)
    condition = IceMaskedGasExchange(gas_exchange, ice_concentration)

    return FluxBoundaryCondition(condition, discrete_form = true)
end

@inline function (ge::IceMaskedGasExchange)(i, j, grid, clock, model_fields)
    uninhibited_exchange = ge.gas_exchange(i, j, grid, clock, model_fields)
    ice_concentration = @inbounds ge.ice_concentration[i, j, 1]

    return uninhibited_exchange * (1 - ice_concentration) # as done in PISCES (and presumably others)
end