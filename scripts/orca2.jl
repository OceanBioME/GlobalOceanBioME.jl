# # [Coarse global ocean--sea ice simulation](@id coarse-degree-ocean-seaice)
#
# This example configures a global ocean--sea ice simulation on the ["ORCA2" grid](https://www.nemo-ocean.eu/doc/node108.html#Tab_orca_zgr)
# which is tripolar grid which transitions to a Mercator grid in the tropics to better resolve 
# equatorial dynamics (as the sign of the coriolis coefficient switches on the equator).
# The grid nominally has a 2° resolution with meridional refinement to 0.5° in the tropics 
# and near Antarctica, and refinement to ~0.5° in the Mediterranean, Red, Black and Caspian Seas.
# The grid has been refined carefully designed by NEMO to carefully capture important physics
# and maintain anisotropy close to 1 in the ocean, especially in strongly eddying regions such 
# as the gulf stream. They also provide bathymetry which is refined to represent important straits.
#
# The model is forced with by repeat-year JRA55 atmospheric reanalysis and initialized by
# temperature, salinity, sea ice concentration, and sea ice thickness from the ECCO state estimate. 
# It includes a few closures:
# - "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`,
# - `CATKEVerticalDiffusivity` for vertical convective mixing,
# - `HorizontalScalarBiharmonicDiffusivity` to damp grid scale noise,
# - and `VerticalScalarDiffusivity` emulating mixing from internal tides
#
# For this example, we need Oceananigans, NumericalEarth, Dates, CUDA, and
# CairoMakie to visualize the simulation.

pickup = false

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using Printf
using Statistics
using CUDA
using OceanBioME
using GlobalOceanBioME

using Oceananigans.Architectures: on_architecture

const data_path = has_cuda_gpu() ? "/cephfs/home/js2430/store/Global/data" : "data" #"/home/js2430/rds/hpc-work/GlobalOceanBioME/data"#
# ### Grid and Bathymetry
arch = GPU()
Nz = 30
z = ExponentialDiscretization(Nz, -5500, 0; scale = 1240)
grid = ORCATripolarGrid(arch; dataset = ORCA2(), z, Nz, remove_closed_basins = true, halo = (5, 5, 4))

# ### Closures
#
# We include a Gent-McWilliams isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization.

@inline νhb(i, j, k, grid, λ) = Oceananigans.Operators.Az(i, j, k, grid, Center(), Center(), Center())^2 / λ
νh = CenterField(grid)
set!(νh, KernelFunctionOperation{Center, Center, Center}(νhb, grid, 40days))
horizontal_viscosity = HorizontalScalarBiharmonicDiffusivity(ν=νh) 

@inline henyey_diffusivity(x, y, z) = max(2e-6, 3e-5 * abs(sind(y)))
ν_henyey = CenterField(grid)
set!(ν_henyey, henyey_diffusivity)
vertical_diffusivity = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=ν_henyey, ν=3e-5)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=500, κ_symmetric=500)#1000)#500)

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity

vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(); 
                                           maximum_tracer_diffusivity = 1,
                                           maximum_tke_diffusivity = 1,
                                           maximum_viscosity = 1)

free_surface       = SplitExplicitFreeSurface(grid; substeps=20*3)
momentum_advection = WENOVectorInvariant(order=5)

tracer_advection = 
    (T = WENO(order=5),
     S = WENO(order=5, bounds=(0.0, Inf)),
     P = WENO(order=5, bounds=(0.0, Inf)),
     Z = WENO(order=5, bounds=(0.0, Inf)),
     DOM = WENO(order=5, bounds=(0.0, Inf)),
     sPOM = WENO(order=5, bounds=(0.0, Inf), minimum_buffer_upwind_order=1),
     bPOM = WENO(order=5, bounds=(0.0, Inf), minimum_buffer_upwind_order=1),
     DIC = WENO(order=5, bounds=(0.0, Inf)),
     Alk = WENO(order=5, bounds=(0.0, Inf)),
     Fe = WENO(order=5, bounds=(0.0, Inf)),
     NO₃ = WENO(order=5, bounds=(0.0, Inf)),
     NH₄ = WENO(order=5, bounds=(0.0, Inf)))#WENO(order=5, minimum_buffer_upwind_order=1) # if we don't do this sinking particles get messed up

dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 11, 1)
mask = LinearlyTaperedPolarMask(southern=(-80, -70), northern=(70, 90), z=(-100, 0))
salinity = Metadata(:salinity;  dates, dataset=ECCO4DarwinMonthly(), dir = data_path)
rate = 1/6days
FS = DatasetRestoring(salinity, grid; mask, rate)

# ### Atmospheric forcing

# We force the simulation with a JRA55-do atmospheric reanalysis.
radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(2920),
                                       include_rivers_and_icebergs = true,
                                       dir = data_path)

# BGC

modifiers = (ScaleNegativeTracers((:NO₃, :NH₄, :P, :Z, :sPOM, :bPOM, :DOM); invalid_fill_value = zero(grid)),
             ScaleNegativeTracers((:Fe, :P, :Z, :sPOM, :bPOM, :DOM); scalefactors = (1/4.6375e-5, 1, 1, 1, 1, 1), invalid_fill_value = zero(grid)),
             ScaleNegativeTracers((:DIC, :P, :Z, :sPOM, :bPOM, :DOM); scalefactors = (1/6.56, 1, 1, 1, 1, 1), invalid_fill_value = zero(grid)))

surface_PAR = PAR_from_atmosphere(atmosphere)
underlying_light_attenuation = TwoBandPhotosyntheticallyActiveRadiation(; grid, surface_PAR)
light_attenuation = GlobalOceanBioME.IceMaskedLightAttenuation(; underlying_light_attenuation, 
                                                                 ice_thickness = Field{Center, Center, Nothing}(grid), 
                                                                 ice_concentration = Field{Center, Center, Nothing}(grid))

biogeochemistry = LOBSTER(; grid, 
                            nutrients = NitrateAmmoniaIron(),
                            carbonate_system = CarbonateSystem(),
                            detritus = TwoParticleAndDissolved(grid; large_particle_sinking_speed = 80/days), # so we don't violate the CFL condition near the surface
                            light_attenuation,
                            modifiers)

underlying_CO₂_flux = CarbonDioxideGasExchangeBoundaryCondition(; air_concentration = 357.21,#278, # omip-bgc does pre industrial baseline for spinup
                                                                  wind_speed = wind_from_atmosphere(atmosphere))

CO₂_flux = GlobalOceanBioME.IceMaskedGasExchangeBoundaryCondition(underlying_CO₂_flux, Field{Center, Center, Nothing}(grid))

iron_flux = DustCOMM_iron_deposition_boundary_condition(grid)

boundary_conditions = (; Fe = FieldBoundaryConditions(top = iron_flux),
                         DIC = FieldBoundaryConditions(top = CO₂_flux))

rivers = GlobalOceanBioME.river_exports(grid; tracers = (:DIN, :DON, :DOC))

forcing = (S = FS,
           DOM = Forcing(rivers.DON),
           NO₃ = Forcing(rivers.DIN),
           DIC = Forcing(rivers.DIC),
           Alk = Forcing(rivers.Alk),
           Fe  = Forcing(rivers.Fe))

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity),
                         forcing,
                         boundary_conditions,
                         biogeochemistry)

sea_ice = sea_ice_simulation(grid, ocean; dynamics = nothing)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
dataset = ECCO4Monthly()
darwin = ECCO4DarwinMonthly()
ecco_temperature           = Metadatum(:temperature; date, dataset, dir = data_path)
ecco_salinity              = Metadatum(:salinity; date, dataset, dir = data_path)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset, dir = data_path)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset, dir = data_path)
ecco_nitrate               = Metadatum(:nitrate; date, dataset = darwin, dir = data_path)
ecco_iron                  = Metadatum(:dissolved_iron; date, dataset = darwin, dir = data_path)
ecco_dic                   = Metadatum(:dissolved_inorganic_carbon; date, dataset = darwin, dir = data_path)
ecco_alk                   = Metadatum(:alkalinity; date, dataset = darwin, dir = data_path)
#=
using JLD2
restart_file = jldopen("../NumericalEarth.jl/checkpointer_orca2_iteration1273120.jld2")
=#
set!(ocean.model, T=ecco_temperature, 
                  S=ecco_salinity, 
                  NO₃ = ecco_nitrate, 
                  Fe = ecco_iron, 
                  DIC = ecco_dic, 
                  Alk = ecco_alk)
#=
ocean.model.velocities.u.data.parent .= on_architecture(arch, restart_file["simulation/model/ocean/model/velocities/u/data"])
ocean.model.velocities.v.data.parent .= on_architecture(arch, restart_file["simulation/model/ocean/model/velocities/v/data"])
ocean.model.velocities.w.data.parent .= on_architecture(arch, restart_file["simulation/model/ocean/model/velocities/w/data"])
ocean.model.tracers.T.data.parent .= on_architecture(arch, restart_file["simulation/model/ocean/model/tracers/T/data"])
ocean.model.tracers.S.data.parent .= on_architecture(arch, restart_file["simulation/model/ocean/model/tracers/S/data"])
=#
tracers = ocean.model.tracers

for tracer in (tracers.NO₃, tracers.DIC, tracers.Alk, tracers.Fe)
    tracer .*= 1000
    tracer.data.parent .= max.(tracer.data.parent, 0)
end

set!(ocean.model, P = 1e-3, 
                  Z = 1e-4, 
                  NH₄ = 1e-7, 
                  DOM = 1e-7, 
                  sPOM = 1e-7,
                  bPOM = 1e-7)
set!(sea_ice.model, h=ecco_sea_ice_thickness, 
                    ℵ=ecco_sea_ice_concentration)

#sea_ice.model.ice_thickness.data.parent .= on_architecture(arch, restart_file["simulation/model/sea_ice/model/ice_thickness/data"])
#sea_ice.model.ice_concentration.data.parent .= on_architecture(arch, restart_file["simulation/model/sea_ice/model/ice_concentration/data"])

#close(restart_file)

# ### Coupled simulation

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# With Runge-Kutta 3rd order time-stepping we can safely use a timestep of 20 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=90minutes, stop_time=295*365days)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    P = ocean.model.tracers.P
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg5 = @sprintf(", wall time: %s", prettytime(step_time))
    msg6 = @sprintf(", Δt = %s, %.1f SYPD\n", prettytime(sim.Δt), Units.day / (step_time * 365 / 10))
    msg7 = @sprintf(", P ∈ [%.2f, %.2f]", minimum(P), maximum(P))
    @info msg1 * msg2 * msg4 * msg5 * msg6 * msg7

    wall_time[] = time_ns()

    return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, TimeInterval(10days))

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature))

u, v, w = ocean.model.velocities

diagnostics = (drake_transport = Field(Integral(view(u, 108, 20:33, :), dims = (2, 3))),
               mean_T = Field(Average(ocean.model.tracers.T)),
               mean_S = Field(Average(ocean.model.tracers.S)),
               mean_SSH = Field(Average(ocean.model.free_surface.displacement)),
               total_TKE = Field(Integral(ocean.model.tracers.e)))

vertical_slices = 
    (atlantic = view(ocean.model.tracers.T, 125, :, :),
     pacific = view(ocean.model.tracers.T, 66, :, :))

surface_flux = (; CO₂ = BoundaryConditionOperation(ocean.model.tracers.DIC, :top, ocean.model))

fname_suffix = "orca2_bgc_ice_blocking"

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(365days/12),
                                            filename = "surface_"*fname_suffix,
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = !pickup)

ocean.output_writers[:surface_CO2] = JLD2Writer(ocean.model, surface_flux;
                                                schedule = AveragedTimeInterval(5days),
                                                filename = "surface_CO2_"*fname_suffix,
                                                overwrite_existing = !pickup)

ocean.output_writers[:diagnostics] = JLD2Writer(ocean.model, diagnostics;
                                                schedule = TimeInterval(365days/12),
                                                filename = "diagnostics_"*fname_suffix,
                                                overwrite_existing = !pickup)

ocean.output_writers[:vslices] = JLD2Writer(ocean.model, vertical_slices;
                                            schedule = TimeInterval(365days/12),
                                            filename = "basin_slice_"*fname_suffix,
                                            overwrite_existing = !pickup)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              schedule = TimeInterval(365days/12),
                                              filename = "sea_ice_"*fname_suffix,
                                              overwrite_existing = !pickup)

simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                        prefix = "checkpointer_"*fname_suffix,
                                                        schedule = TimeInterval(2*365days),
                                                        cleanup = true,
                                                        overwrite_existing = !pickup)

# ### Ready to run

# We are ready to press the big red button and run the simulation.
run!(simulation; pickup)
