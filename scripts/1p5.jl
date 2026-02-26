# # One-degree global ocean--sea ice simulation
#
# This example configures a global ocean--sea ice simulation at 1ᵒ horizontal resolution with
# realistic bathymetry and a few closures including the "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`.
# The simulation is forced by repeat-year JRA55 atmospheric reanalysis
# and initialized by temperature, salinity, sea ice concentration, and sea ice thickness
# from the ECCO state estimate.
#
# For this example, we need Oceananigans, ClimaOcean, Dates, CUDA, and
# CairoMakie to visualize the simulation.

using ClimaOcean
using OceanBioME
using Oceananigans
using GlobalOceanBioME
using Oceananigans.Units
using Dates
using Printf
using Statistics
using CUDA
using NetCDF

# ### Grid and Bathymetry

# We start by constructing an underlying TripolarGrid at ~1 degree resolution,

arch = CPU()#GPU()

grid = GlobalOceanBioME.one_p_five_degree_grid(arch)

# ### Closures
#
# We include a Gent-McWilliams isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization.

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation()) 
vertical_mixing = ClimaOcean.OceanSimulations.default_ocean_closure()
κh = 500
#=background_lateral_diffusion = ScalarDiffusivity(Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization(), 
                                                 Oceananigans.TurbulenceClosures.HorizontalFormulation(); 
                                                 κ = (e = 0, T = 0, S = 0, NO₃ = κh, NH₄ = κh, Fe = κh, P = κh, Z = κh, sPOM = κh, bPOM = κh, DOM = κh, DIC = κh, Alk = κh))#20000)=#
background_vertical_diffusion = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); κ=1e-5, ν=1e-4)

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with 70 substeps for the barotropic mode.

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)

# ### Atmospheric forcing

# We force the simulation with a JRA55-do atmospheric reanalysis.
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(80),
                                       include_rivers_and_icebergs = false)
radiation  = Radiation(arch)

# ### Biogeochemistry
@kwdef struct PARfromFTS{SW, GR, LO, PF} <: Function
                shortwave :: SW
                     grid :: GR
                 location :: LO
             PAR_fraction :: PF = 0.45
end

@inline function (par::PARfromFTS)(x, y, t)
    η = par.PAR_fraction 

    SS = par.shortwave

    I =  Oceananigans.Fields.interpolate((x, y, zero(x)), 
                                         Oceananigans.Units.Time(t), 
                                         SS.data, 
                                         par.location, 
                                         par.grid,
                                         SS.times,
                                         SS.backend,
                                         SS.time_indexing)

    return η * I
end

using Adapt
import Adapt: adapt_structure

Adapt.adapt_structure(to, par::PARfromFTS) =
    PARfromFTS(adapt(to, par.shortwave),
               adapt(to, par.grid),
               adapt(to, par.location),
               adapt(to, par.PAR_fraction))

PAR = PARfromFTS(; shortwave = atmosphere.downwelling_radiation.shortwave,
                   grid = atmosphere.downwelling_radiation.shortwave.grid,
                   location = (Center(), Center(), nothing))

large_particle_sinking_speed = ZFaceField(grid)
small_particle_sinking_speed = ZFaceField(grid)

large_particle_sinking_kfo = 
    KernelFunctionOperation{Center, Center, Face}(
        (i, j, k, grid) -> -200/24/60/60 * !(Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, grid)),
        grid)
small_particle_sinking_kfo = 
    KernelFunctionOperation{Center, Center, Face}(
        (i, j, k, grid) -> -3.47e-5 * !(Oceananigans.ImmersedBoundaries.immersed_cell(i, j, k, grid)),
        grid)

set!(large_particle_sinking_speed, large_particle_sinking_kfo)
set!(small_particle_sinking_speed, small_particle_sinking_kfo)

biogeochemistry = LOBSTER(; grid, 
                            nutrients = OceanBioME.Models.LOBSTERModel.NitrateAmmoniaIron(),
                            carbonate_system = CarbonateSystem(),
                            surface_photosynthetically_active_radiation = PAR,
                            scale_negatives = true,
                            detritus = TwoParticleAndDissolved(grid; large_particle_sinking_speed, small_particle_sinking_speed))

# TODO: 
# - co2 surfgace exchange with the prognostic wind [-]
# - runoff [-]
# - dust deposition (maybe I put this in oceanbiome already?) [-]

CO₂_flux = CarbonDioxideGasExchangeBoundaryCondition(; air_concentration = 278, # omip-bgc does pre industrial baseline for spinup
                                                       wind_speed = values(atmosphere.velocities))

# dust - TODO: add disolution through the water column not just the surface
dust_solubility = 0.02
dust_iron_content = 0.035
# this is a contemparary dust dataset so idk if we need to change to some preindustrial
dust_fname = "DustCOMM_totdep_Dgr20um_seas_bin.nc"
total_deposition = ncread(dust_fname, "Total deposition flux")[:, 2, :, :] # kg/m²/yr
dep_lat = ncread(dust_fname, "lat")
dep_lon = ncread(dust_fname, "lon")
dep_times = ncread(dust_fname, "season")
dep_grid = LatitudeLongitudeGrid(size = (length(dep_lon), length(dep_lat), 1), 
                                 latitude = (minimum(dep_lat), maximum(dep_lat)), 
                                 longitude = (minimum(dep_lon), maximum(dep_lon)), z = (-1, 0))

iron_deposition = @. dust_solubility * dust_iron_content * total_deposition / 55.8 * 1000 / (365*24*60*60) # mmol Fe/m²/s

# times are 15/01, 15/04, 15/07, 15/10 ish
iron_deposition_fts = 
    FieldTimeSeries((Center(), Center(), Center()), 
                    dep_grid, 
                    (1.296e6, 9.1799998272e6, 1.70639996544e7, 2.49479994816e7);
                    time_indexing = Oceananigans.OutputReaders.Cyclical())
for n in 1:4
    set!(iron_deposition_fts[n], -reshape(iron_deposition[n, :, :], length(dep_lon), length(dep_lat), 1))
end

iron_deposition_final = 
    FieldTimeSeries((Center(), Center(), Center()), 
                    grid, 
                    (1.296e6, 9.1799998272e6, 1.70639996544e7, 2.49479994816e7);
                    time_indexing = Oceananigans.OutputReaders.Cyclical())

for n in 1:4
    Oceananigans.Fields.interpolate!(iron_deposition_final[n], iron_deposition_fts[n])
end

iron_flux = FluxBoundaryCondition(iron_deposition_final)

# rivers
river_fluxs = GlobalOceanBioME.river_nutrients(grid)

CUDA.@allowscalar begin
    total_DIC_flux =  Field(Integral(river_fluxs.DOC))[1, 1, 1] * 0.6 / 0.4 # mmol C/s
end
total_Fe_flux = 1.45e12 / (365*24*24) / 55.8 * 1000 # mmolFe/s
Fe_flux_scalefactor = total_Fe_flux / total_DIC_flux

forcing = (DOM = Forcing(rivers.DON),
           NO₃ = Forcing(rivers.DIN),
           DIC = Forcing(rivers.DOC * 0.6 / 0.4),
           Alk = Forcing(rivers.DOC * 0.6 / 0.4),
           Fe  = Forcing(rivers.Fe * 0.6 / 0.4 * Fe_flux_scalefactor))

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         biogeochemistry,
                         forcing,
                         timestepper = :SplitRungeKutta3,
                         closure=(eddy_closure, vertical_mixing), #background_lateral_diffusion, background_vertical_diffusion),
                         tracers = (:T, :S, :NO₃, :NH₄, :Fe, :P, :Z, :sPOM, :bPOM, :DOM, :DIC, :Alk), # temp otherwise advection doesn't get turned on
                         boundary_conditions = (; Fe = FieldBoundaryConditions(top = iron_flux)))

@info "We've built an ocean simulation with model:"
@show ocean.model

# ### Sea Ice simulation
#
# We also build a sea ice simulation. We use the default configuration:
# EVP rheology and a zero-layer thermodynamic model that advances thickness
# and concentration.

sea_ice = sea_ice_simulation(grid, ocean; advection=tracer_advection)

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
ecco = ECCO4Monthly()
darwin = ECCO4DarwinMonthly()#ECCO4Monthly()
ecco_temperature           = Metadatum(:temperature; date, dataset = darwin)
ecco_salinity              = Metadatum(:salinity; date, dataset = darwin)
ecco_nitrate               = Metadatum(:nitrate; date, dataset = darwin)
ecco_iron                  = Metadatum(:dissolved_iron; date, dataset = darwin)
ecco_dic                   = Metadatum(:dissolved_inorganic_carbon; date, dataset = darwin)
ecco_alk                   = Metadatum(:alkalinity; date, dataset = darwin)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset = ecco)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset = ecco)

set!(ocean.model, T=ecco_temperature, S=ecco_salinity, NO₃=ecco_nitrate, Fe=ecco_iron, DIC=ecco_dic, Alk=ecco_alk)
set!(ocean.model, P = (x, y, z) -> 1e-3, Z = (x, y, z) -> 1e-4, NH₄ = 1e-7, DOM = 1e-7, sPOM = 1e-7, bPOM = 1e-7)
set!(sea_ice.model, h=ecco_sea_ice_thickness, ℵ=ecco_sea_ice_concentration)

ocean.model.tracers.Fe  .*= 1000
ocean.model.tracers.DIC .*= 1000
ocean.model.tracers.Alk .*= 1000
ocean.model.tracers.NO₃ .*= 1000

ocean.model.tracers.NO₃.data .= max.(0, ocean.model.tracers.NO₃.data)

# ### Coupled simulation 

# Now we are ready to build the coupled ocean--sea ice model and bring everything
# together into a `simulation`.

# With Runge-Kutta 3rd order time-stepping we can safely use a timestep of 20 minutes.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=20minutes, stop_time=10*365days)

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
    Tmin, Tmax, Tavg = minimum(T), maximum(T), mean(view(T, :, :, ocean.model.grid.Nz))
    Pmin, Pmax, Pavg = minimum(P), maximum(P), mean(view(P, :, :, ocean.model.grid.Nz))
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", extrema(To): (%.1f, %.1f) ᵒC, mean(To(z=0)): %.1f ᵒC", Tmin, Tmax, Tavg)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msgP = @sprintf(", extrema(P): (%.1e, %.1e) mmolN/m³, mean(P(z=0)): %.1e mmolN/m³", Pmin, Pmax, Pavg)
    msg5 = @sprintf(", wall time: %s", prettytime(step_time))
    msg6 = @sprintf(", Δt = %s, %.1f SYPD\n", prettytime(sim.Δt), Units.day / (step_time * 365 / 5))

    @info msg1 * msg2 * msg3 * msg4 * msgP * msg5 * msg6

    wall_time[] = time_ns()

    return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, TimeInterval(5days))

# ### Output
#
# We are almost there! We need to save some output. Below we choose to save _only surface_
# fields using the `indices` keyword argument. We save all the velocity and tracer components.
# Note, that besides temperature and salinity, the CATKE vertical mixing parameterization
# also uses a prognostic turbulent kinetic energy, `e`, to diagnose the vertical mixing length.

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities, (; PAR = biogeochemistry.light_attenuation.field))
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
                         sea_ice.model.velocities)

fname_suffix = "low_res"#high_res"

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = TimeInterval(1days),
                                            filename = "ocean_one_degree_surface_fields_bgc_"*fname_suffix,
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(ocean.model, sea_ice_outputs;
                                              schedule = TimeInterval(1days),
                                              filename = "sea_ice_one_degree_surface_fields_bgc_"*fname_suffix,
                                              overwrite_existing = true)

add_callback!(simulation, Oceananigans.Models.NaNChecker(; fields = (; P = ocean.model.tracers.P), erroring = true), IterationInterval(100))

# ### Ready to run

# We are ready to press the big red button and run the simulation.
#simulation.stop_time = 20days
simulation.Δt = 10minutes
run!(simulation)
#=
import Oceananigans.Simulations: TimeStepWizard
(wizard::TimeStepWizard)(simulation::Simulation{<:OceanSeaIceModel}) =
    simulation.Δt = Oceananigans.Simulations.new_time_step(simulation.Δt, wizard, simulation.model.ocean.model)

conjure_time_step_wizard!(simulation, max_change = 2, min_change = 0.1, cfl = 0.5, schedule = IterationInterval(100))=#
#simiulaiton.stop_time = 365*10Units.days
#run!(simulation)
# ### A movie
#
# We load the saved output and make a movie of the simulation. First we plot a snapshot:
using CairoMakie

# We suffix the ocean fields with "o":
uo = FieldTimeSeries("ocean_one_degree_surface_fields_bgc_"*fname_suffix*".jld2",  "u"; backend = OnDisk())
vo = FieldTimeSeries("ocean_one_degree_surface_fields_bgc_"*fname_suffix*".jld2",  "v"; backend = OnDisk())
To = FieldTimeSeries("ocean_one_degree_surface_fields_bgc_"*fname_suffix*".jld2",  "T"; backend = OnDisk())
eo = FieldTimeSeries("ocean_one_degree_surface_fields_bgc_"*fname_suffix*".jld2",  "e"; backend = OnDisk())
Po = FieldTimeSeries("ocean_one_degree_surface_fields_bgc_"*fname_suffix*".jld2",  "P"; backend = OnDisk())

# and sea ice fields with "i":
ui = FieldTimeSeries("sea_ice_one_degree_surface_fields_bgc_"*fname_suffix*".jld2", "u"; backend = OnDisk())
vi = FieldTimeSeries("sea_ice_one_degree_surface_fields_bgc_"*fname_suffix*".jld2", "v"; backend = OnDisk())
hi = FieldTimeSeries("sea_ice_one_degree_surface_fields_bgc_"*fname_suffix*".jld2", "h"; backend = OnDisk())
ℵi = FieldTimeSeries("sea_ice_one_degree_surface_fields_bgc_"*fname_suffix*".jld2", "ℵ"; backend = OnDisk())
Ti = FieldTimeSeries("sea_ice_one_degree_surface_fields_bgc_"*fname_suffix*".jld2", "T"; backend = OnDisk())

times = uo.times
Nt = length(times)
n = Observable(Nt)

# We create a land mask and use it to fill land points with `NaN`s.
land = interior(To.grid.immersed_boundary.bottom_height) .≥ 0

Toₙ = @lift begin
    Tₙ = interior(To[$n])
    Tₙ[land] .= NaN
    view(Tₙ, :, :, 1)
end

eoₙ = @lift begin
    eₙ = interior(eo[$n])
    eₙ[land] .= NaN
    view(eₙ, :, :, 1)
end

Poₙ = @lift begin
    Pₙ = interior(Po[$n])
    Pₙ[land] .= NaN
    view(Pₙ, :, :, 1)
end

heₙ = @lift begin
    hₙ = interior(hi[$n])
    ℵₙ = interior(ℵi[$n])
    hₙ[land] .= NaN
    view(hₙ, :, :, 1) .* view(ℵₙ, :, :, 1)
end

# We compute the surface speeds for the ocean and the sea ice.
uoₙ = Field{Face, Center, Nothing}(uo.grid)
voₙ = Field{Center, Face, Nothing}(vo.grid)

uiₙ = Field{Face, Center, Nothing}(ui.grid)
viₙ = Field{Center, Face, Nothing}(vi.grid)

so = Field(sqrt(uoₙ^2 + voₙ^2))
si = Field(sqrt(uiₙ^2 + viₙ^2))

soₙ = @lift begin
    parent(uoₙ) .= parent(uo[$n])
    parent(voₙ) .= parent(vo[$n])
    compute!(so)
    soₙ = interior(so)
    soₙ[land] .= NaN
    view(soₙ, :, :, 1)
end

siₙ = @lift begin
    parent(uiₙ) .= parent(ui[$n])
    parent(viₙ) .= parent(vi[$n])
    compute!(si)
    siₙ = interior(si)
    hₙ = interior(hi[$n])
    ℵₙ = interior(ℵi[$n])
    he = hₙ .* ℵₙ
    siₙ[he .< 1e-7] .= 0
    siₙ[land] .= NaN
    view(siₙ, :, :, 1)
end

# Finally, we plot a snapshot of the surface speed, temperature, and the turbulent
# eddy kinetic energy from the CATKE vertical mixing parameterization as well as the
# sea ice speed and the effective sea ice thickness.
fig = Figure(size=(1200, 1000))

year_day(time) = "$(floor(Int, time/365days)) years "*prettytime(mod(time, 365days))

title = @lift string("Global ~1.4ᵒ ocean simulation after ", year_day(times[$n] - times[1]))

axso = Axis(fig[1, 1])
axsi = Axis(fig[1, 3])
axTo = Axis(fig[2, 1])
axhi = Axis(fig[2, 3])
axeo = Axis(fig[3, 1])
axPo = Axis(fig[3, 3])

hmo = heatmap!(axso, soₙ, colorrange = (0, 0.5), colormap = :deep,  nan_color=:lightgray)
hmi = heatmap!(axsi, siₙ, colorrange = (0, 0.5), colormap = :greys, nan_color=:lightgray)
Colorbar(fig[1, 2], hmo, label = "Ocean Surface speed (m s⁻¹)")
Colorbar(fig[1, 4], hmi, label = "Sea ice speed (m s⁻¹)")

hmo = heatmap!(axTo, Toₙ, colorrange = (-1, 32), colormap = :magma, nan_color=:lightgray)
hmi = heatmap!(axhi, heₙ, colorrange =  (0, 4),  colormap = :blues, nan_color=:lightgray)
Colorbar(fig[2, 2], hmo, label = "Surface Temperature (ᵒC)")
Colorbar(fig[2, 4], hmi, label = "Effective ice thickness (m)")

hm = heatmap!(axeo, eoₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color=:lightgray)
Colorbar(fig[3, 2], hm, label = "Turbulent Kinetic Energy (m² s⁻²)")

hmP = heatmap!(axPo, Poₙ, colorrange = (0, 0.05), colormap = Reverse(:bamako), nan_color=:lightgray)
Colorbar(fig[3, 4], hmP, label = "Phytoplankton (mmolN m⁻³)")

for ax in (axso, axsi, axTo, axhi, axeo, axPo)
    hidedecorations!(ax)
end

Label(fig[0, :], title)

save("global_snapshot_bgc.png", fig)
nothing #hide

# ![](global_snapshot.png)

# And now a movie:

CairoMakie.record(fig, "$(fname_suffix)_ocean_surface_bgc.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](one_degree_global_ocean_surface.mp4)
