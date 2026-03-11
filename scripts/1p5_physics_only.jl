# # One-degree global ocean--sea ice simulation
#
# This example configures a global ocean--sea ice simulation at 1ᵒ horizontal resolution with
# realistic bathymetry and a few closures including the "Gent-McWilliams" `IsopycnalSkewSymmetricDiffusivity`.
# The simulation is forced by repeat-year JRA55 atmospheric reanalysis
# and initialized by temperature, salinity, sea ice concentration, and sea ice thickness
# from the ECCO state estimate.
#
# For this example, we need Oceananigans, NumericalEarth, Dates, CUDA, and
# CairoMakie to visualize the simulation.

@info "better outputs"#, sea ice dynamics on"

using NumericalEarth
using OceanBioME
using Oceananigans
using GlobalOceanBioME
using Oceananigans.Units
using Dates
using Printf
using Statistics
using CUDA
using NetCDF
using Oceananigans.Architectures: on_architecture
using Oceananigans.ImmersedBoundaries: immersed_cell

const data_path = "/cephfs/home/js2430/store/Global/data"

# ### Grid and Bathymetry

# We start by constructing an underlying TripolarGrid at ~1 degree resolution,

arch = GPU()#Distributed(GPU(); partition = Partition(2, 1))#GPU()

grid = GlobalOceanBioME.one_p_five_degree_grid(arch; Nz = 42)

# ### Closures
#
# We include a Gent-McWilliams isopycnal diffusivity as a parameterization for the mesoscale
# eddy fluxes. For vertical mixing at the upper-ocean boundary layer we include the CATKE
# parameterization.

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation, HorizontalFormulation, VerticallyImplicitTimeDiscretization

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, 
                                                 skew_flux_formulation=AdvectiveFormulation())

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    CATKEMixingLength,
    CATKEEquation

mixing_length = CATKEMixingLength(Cᵇ=0.01)
turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ=1.0)
vertical_mixing = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(); 
                                           mixing_length, 
                                           turbulent_kinetic_energy_equation, 
                                           minimum_tke = 1e-6,
                                           maximum_tracer_diffusivity = 1,
                                           maximum_tke_diffusivity = 1,
                                           maximum_viscosity = 1,
                                           minimum_convective_buoyancy_flux = 1e-10)

# horizontal biharmonic momentum viscosity
νx = Field(0.1 * xspacings(grid)^4/5days)
νy = Field(0.1 * yspacings(grid)^4/5days)
ν_bmc = CenterField(grid)

biharmonic_ν = CenterField(grid)
CUDA.@allowscalar begin
    set!(biharmonic_ν, max.(interior(νx), interior(νy)))#, interior(ν_bmc)))
end
Oceananigans.ImmersedBoundaries.mask_immersed_field!(biharmonic_ν)

horizontal_viscosity = ScalarBiharmonicDiffusivity(HorizontalFormulation(); ν=biharmonic_ν) # change to ν = C Δx^4 / tau with tau=~5days

# vertical tracer mixing
κ_vertical = CenterField(grid)
@inline function κ_vertical_kernel(i, j, k, grid, bottom_height) 
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    κbb = @inbounds 1e-4 * exp(-(z - bottom_height[i, j, 1])./200)

    roughness = sqrt(Oceananigans.Operators.∂xᶜᶜᶜ(i, j, 1, grid, bottom_height)^2 + Oceananigans.Operators.∂yᶜᶜᶜ(i, j, 1, grid, bottom_height)^2)

    n_neighbours = zero(grid)

    for di in -1:1, dj in -1:1, dk in -1:1
        n_neighbours += ifelse((di == 0)&(dj == 0)&(dk == 0), 0, !immersed_cell(i+di, j+dj, k+dk, grid))
    end

    κrt = 1e-3 * roughness / 0.057 * exp(-(z - bottom_height[i, j, 1])./200)

    return 1e-5 + κbb + κrt
end
κ_vertical_kfo = KernelFunctionOperation{Center, Center, Center}(κ_vertical_kernel, grid, grid.immersed_boundary.bottom_height)
set!(κ_vertical, κ_vertical_kfo)
Oceananigans.ImmersedBoundaries.mask_immersed_field!(κ_vertical)

vertical_diffusivity = 
    VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); 
                              κ=NamedTuple{(:T, :S, :e)}([[κ_vertical for _ in 1:2]..., 0]))

# ### Ocean simulation
# Now we bring everything together to construct the ocean simulation.
# We use a split-explicit timestepping with ~70~ 20 substeps for the barotropic mode.

free_surface       = SplitExplicitFreeSurface(grid; substeps=20)#
momentum_advection = VectorInvariant(; vorticity_scheme = UpwindBiased(order=3),
                                       divergence_scheme = UpwindBiased(order = 1))
tracer_advection   = (; T = UpwindBiased(order = 3),
                        S = UpwindBiased(order = 3),
                        e = UpwindBiased(order = 1))

# ### Atmospheric forcing

# We force the simulation with a JRA55-do atmospheric reanalysis.
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(2920), # put it all in GPU memory
                                       include_rivers_and_icebergs = true,
                                       dir = data_path)
radiation = Radiation(arch)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         timestepper = :SplitRungeKutta3,
                         closure=(eddy_closure, vertical_mixing, horizontal_viscosity, vertical_diffusivity),
                         tracers = (:T, :S)) 

@info "Built"

# ### Sea Ice simulation
#
# We also build a sea ice simulation. We use the default configuration:
# EVP rheology and a zero-layer thermodynamic model that advances thickness
# and concentration.

sea_ice = sea_ice_simulation(grid, ocean; dynamics = nothing)#advection = UpwindBiased(order = 1))

# ### Initial condition

# We initialize the ocean and sea ice models with data from the ECCO state estimate.

date = DateTime(1993, 1, 1)
ecco = ECCO4Monthly()
darwin = ECCO4DarwinMonthly()#ECCO4Monthly()
ecco_temperature           = Metadatum(:temperature; date, dataset = darwin, dir = data_path)
ecco_salinity              = Metadatum(:salinity; date, dataset = darwin, dir = data_path)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset = ecco, dir = data_path)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset = ecco, dir = data_path)

set!(ocean.model, T=ecco_temperature, S=ecco_salinity)
set!(sea_ice.model, h=ecco_sea_ice_thickness, ℵ=ecco_sea_ice_concentration)


coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt=20minutes, stop_time=60*365days)

# ### A progress messenger
#
# We write a function that prints out a helpful progress message while the simulation runs.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg4 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg5 = @sprintf(", wall time: %s", prettytime(step_time))
    msg6 = @sprintf(", Δt = %s, %.1f SYPD\n", prettytime(sim.Δt), Units.day / (step_time * 365 / 5))

    @info msg1 * msg2 * msg4 * msg5 * msg6

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

ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
sea_ice_outputs = merge((h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration,
                         T = sea_ice.model.ice_thermodynamics.top_surface_temperature),
)#sea_ice.model.velocities)

u, v, w = ocean.model.velocities

diagnostics = (drake_transport = Field(Integral(view(u, 161, 11:19, :))),
               mean_T = Field(Average(ocean.model.tracers.T)),
               mean_S = Field(Average(ocean.model.tracers.S)),
               mean_SSH = Field(Average(ocean.model.free_surface.displacement)),
               total_TKE = Field(Integral(ocean.model.tracers.e)),
               )#northern_ice_extent = Field(Integral(view(sea_ice.model.ice_concentration, :, 64:128, 1) > 0.15)),
                #southern_ice_extent = Field(Integral(view(sea_ice.model.ice_concentration, :, 1:64, 1) > 0.15))) # all numbers

vertical_slices = 
    (atlantic = Field(Average(view(ocean.model.tracers.T, 160:220, :, :), dims = 1)),
     pacific = view(ocean.model.tracers.T, 101, :, :))

horizontal_slices = 
    (T = Field(Average(view(ocean.model.tracers.T, :, :, 20:21), dims = 3)),
     S = Field(Average(view(ocean.model.tracers.S, :, :, 20:21), dims = 3)))

mld = (; mld = NumericalEarth.Diagnostics.MixedLayerDepthField(ocean.model.buoyancy, grid, ocean.model.tracers))

fname_suffix = "working_less_substepping_upwind"

slice_frequency = AveragedTimeInterval(30.4166666667days; stride = 12)

ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                            schedule = slice_frequency,
                                            filename = "surface_monthly_"*fname_suffix,
                                            indices = (:, :, grid.Nz),
                                            overwrite_existing = true)

ocean.output_writers[:diagnostics] = JLD2Writer(ocean.model, diagnostics;
                                                schedule = AveragedTimeInterval(1days),
                                                filename = "diagnostics_daily_"*fname_suffix,
                                                overwrite_existing = true)

ocean.output_writers[:vslices] = JLD2Writer(ocean.model, vertical_slices;
                                            schedule = slice_frequency,
                                            filename = "deep_slice_monthly_"*fname_suffix,
                                            overwrite_existing = true)

ocean.output_writers[:hslices] = JLD2Writer(ocean.model, horizontal_slices;
                                            schedule = slice_frequency,
                                            filename = "basin_slices_monthly_"*fname_suffix,
                                            overwrite_existing = true)

ocean.output_writers[:mld] = JLD2Writer(ocean.model, mld;
                                        schedule = slice_frequency,
                                        filename = "mld_monthly_"*fname_suffix,
                                        overwrite_existing = true)

ocean.output_writers[:free_surface] = JLD2Writer(ocean.model, (; η = ocean.model.free_surface.displacement),
                                                 schedule = slice_frequency,
                                                 filename = "ssh_monthly_"*fname_suffix,
                                                 indices = (:, :, grid.Nz+1),
                                                 overwrite_existing = true)

sea_ice.output_writers[:surface] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                              schedule = slice_frequency,
                                              filename = "sea_ice_monthly_"*fname_suffix,
                                              overwrite_existing = true)

ocean.output_writers[:all] = JLD2Writer(ocean.model, ocean_outputs;
                                        schedule = AveragedTimeInterval(365days; stride = 24),
                                        filename = "yearly_means_"*fname_suffix,
                                        overwrite_existing = true)

simulation.output_writers[:checkpointer] = Checkpointer(simulation.model;
                                                        prefix = "checkpointer_"*fname_suffix,
                                                        schedule = TimeInterval(10*365days),
                                                        cleanup = true,
                                                        overwrite_existing = true)

# ### Ready to run

# We are ready to press the big red button and run the simulation.
simulation.Δt = 1minutes
simulation.stop_time = 1days
run!(simulation)

simulation.Δt = 5minutes
simulation.stop_time = 10days
run!(simulation)


simulation.Δt = 10minutes
simulation.stop_time = 20days
run!(simulation)

#=simulation.callbacks[:nan_check] = Callback(IterationInterval(10)) do sim
    for (name, tracer) in pairs((T = ocean.model.tracers.T, u = ocean.model.velocities.u, v = ocean.model.velocities.v))
        if isnan(sum(interior(tracer)))
            T_cpu = on_architecture(CPU(), interior(tracer))
            idx = Tuple(argmax(isnan.(T_cpu)))
            @error "NaN in $name at $idx, day $(sim.model.clock.time/86400)"
        end
    end
end=#

simulation.Δt = 20minutes
simulation.stop_time = 365days
run!(simulation)

simulation.Δt = 30minutes
simulation.stop_time = 60*365days
run!(simulation)
#=
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

# and sea ice fields with "i":
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

heₙ = @lift begin
    hₙ = interior(hi[$n])
    ℵₙ = interior(ℵi[$n])
    hₙ[land] .= NaN
    view(hₙ, :, :, 1) .* view(ℵₙ, :, :, 1)
end

# We compute the surface speeds for the ocean and the sea ice.
uoₙ = Field{Face, Center, Nothing}(uo.grid)
voₙ = Field{Center, Face, Nothing}(vo.grid)

so = Field(sqrt(uoₙ^2 + voₙ^2))

soₙ = @lift begin
    parent(uoₙ) .= parent(uo[$n])
    parent(voₙ) .= parent(vo[$n])
    compute!(so)
    soₙ = interior(so)
    soₙ[land] .= NaN
    view(soₙ, :, :, 1)
end

# Finally, we plot a snapshot of the surface speed, temperature, and the turbulent
# eddy kinetic energy from the CATKE vertical mixing parameterization as well as the
# sea ice speed and the effective sea ice thickness.
fig = Figure(size=(1200, 1000))

year_day(time) = "$(floor(Int, time/365days)) years "*prettytime(mod(time, 365days))

title = @lift string("Global ~1.4ᵒ ocean simulation after ", year_day(times[$n] - times[1]))

axso = Axis(fig[1, 1])
axTo = Axis(fig[1, 3])
axhi = Axis(fig[2, 1])
axeo = Axis(fig[2, 3])

hmo = heatmap!(axso, soₙ, colorrange = (0, 0.5), colormap = :deep,  nan_color=:lightgray)
Colorbar(fig[1, 2], hmo, label = "Ocean Surface speed (m s⁻¹)")

hmo = heatmap!(axTo, Toₙ, colorrange = (-1, 32), colormap = :magma, nan_color=:lightgray)
hmi = heatmap!(axhi, heₙ, colorrange =  (0, 4),  colormap = :blues, nan_color=:lightgray)
Colorbar(fig[1, 4], hmo, label = "Surface Temperature (ᵒC)")
Colorbar(fig[2, 2], hmi, label = "Effective ice thickness (m)")

hm = heatmap!(axeo, eoₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color=:lightgray)
Colorbar(fig[2, 4], hm, label = "Turbulent Kinetic Energy (m² s⁻²)")

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
=#
