using Oceananigans

using NumericalEarth.OceanSimulations: default_planet_rotation_rate,
                                   default_gravitational_acceleration,
                                   default_radiative_forcing,
                                   default_or_override,
                                   u_immersed_bottom_drag,
                                   v_immersed_bottom_drag,
                                   BarotropicPotentialForcing,
                                   XDirection, YDirection,
                                   u_quadratic_bottom_drag,
                                   v_quadratic_bottom_drag,
                                   hasclosure

function ocean_simulation(grid;
                          Δt = estimate_maximum_Δt(grid),
                          closure = default_ocean_closure(),
                          tracers = (:T, :S),
                          free_surface = default_free_surface(grid),
                          reference_density = 1020,
                          rotation_rate = default_planet_rotation_rate,
                          gravitational_acceleration = default_gravitational_acceleration,
                          bottom_drag_coefficient = Default(0.003),
                          forcing = NamedTuple(),
                          biogeochemistry = nothing,
                          timestepper = :QuasiAdamsBashforth2,
                          coriolis = Default(HydrostaticSphericalCoriolis(; rotation_rate)),
                          momentum_advection = WENOVectorInvariant(),
                          tracer_advection = WENO(order=7),
                          equation_of_state = TEOS10EquationOfState(; reference_density),
                          boundary_conditions::NamedTuple = NamedTuple(),
                          radiative_forcing = default_radiative_forcing(grid),
                          warn = true,
                          verbose = false)

    FT = eltype(grid)

    if grid isa RectilinearGrid # turn off Coriolis unless user-supplied
        coriolis = default_or_override(coriolis, nothing)
    else
        coriolis = default_or_override(coriolis)
    end

    # Detect whether we are on a single column grid
    Nx, Ny, _ = size(grid)
    single_column_simulation = Nx == 1 && Ny == 1

    if single_column_simulation
        # Let users put a bottom drag if they want
        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient, zero(grid))

        # Don't let users use advection in a single column model
        tracer_advection = nothing
        momentum_advection = nothing

        # No immersed boundaries in a single column grid
        u_immersed_bc = DefaultBoundaryCondition()
        v_immersed_bc = DefaultBoundaryCondition()
    else
        if warn && !(grid isa ImmersedBoundaryGrid) && verbose
            msg = """Are you totally, 100% sure that you want to build a simulation on

                   $(summary(grid))

                   rather than on an ImmersedBoundaryGrid?
                   """
            @warn msg
        end

        bottom_drag_coefficient = default_or_override(bottom_drag_coefficient)

        u_immersed_drag = FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
        v_immersed_drag = FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

        u_immersed_bc = ImmersedBoundaryCondition(bottom=u_immersed_drag)
        v_immersed_bc = ImmersedBoundaryCondition(bottom=v_immersed_drag)

        # Forcing for u, v
        u_barotropic_potential = Field{Center, Center, Nothing}(grid)
        v_barotropic_potential = Field{Center, Center, Nothing}(grid)
        u_forcing = BarotropicPotentialForcing(XDirection(), u_barotropic_potential)
        v_forcing = BarotropicPotentialForcing(YDirection(), v_barotropic_potential)

        :u ∈ keys(forcing) && (u_forcing = (u_forcing, forcing[:u]))
        :v ∈ keys(forcing) && (v_forcing = (v_forcing, forcing[:v]))
        forcing = merge(forcing, (u=u_forcing, v=v_forcing))
    end

    if !isnothing(radiative_forcing)
        if :T ∈ keys(forcing)
            T_forcing = (forcing.T, radiative_forcing)
        else
            T_forcing = radiative_forcing
        end
        forcing = merge(forcing, (; T=T_forcing))
    end

    bottom_drag_coefficient = convert(FT, bottom_drag_coefficient)

    # Set up boundary conditions using Field
    top_zonal_momentum_flux      = τx = Field{Face, Center, Nothing}(grid)
    top_meridional_momentum_flux = τy = Field{Center, Face, Nothing}(grid)
    top_ocean_heat_flux          = Jᵀ = Field{Center, Center, Nothing}(grid)
    top_salt_flux                = Jˢ = Field{Center, Center, Nothing}(grid)

    # Construct ocean boundary conditions including surface forcing and bottom drag
    u_top_bc = FluxBoundaryCondition(τx)
    v_top_bc = FluxBoundaryCondition(τy)
    T_top_bc = FluxBoundaryCondition(Jᵀ)
    S_top_bc = FluxBoundaryCondition(Jˢ)

    u_bot_bc = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)
    v_bot_bc = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=bottom_drag_coefficient)

    default_boundary_conditions = (u = FieldBoundaryConditions(top=u_top_bc, bottom=u_bot_bc, immersed=u_immersed_bc),
                                   v = FieldBoundaryConditions(top=v_top_bc, bottom=v_bot_bc, immersed=v_immersed_bc),
                                   T = FieldBoundaryConditions(top=T_top_bc),
                                   S = FieldBoundaryConditions(top=S_top_bc))

    # Merge boundary conditions with preference to user
    # TODO: support users specifying only _part_ of the bcs for u, v, T, S (ie adding the top and immersed
    # conditions even when a user-bc is supplied).
    boundary_conditions = merge(default_boundary_conditions, boundary_conditions)
    buoyancy = SeawaterBuoyancy(FT; gravitational_acceleration, equation_of_state)

    if tracer_advection isa NamedTuple
        tracer_advection = with_tracers(tracers, tracer_advection, default_tracer_advection())
    else
        tracer_advection = NamedTuple(name => tracer_advection for name in tracers)
    end

    if hasclosure(closure, CATKEVerticalDiffusivity)
        # Magically add :e to tracers
        if !(:e ∈ tracers)
            tracers = tuple(tracers..., :e)
        end

        # Turn off CATKE tracer advection
        tke_advection = (; e=nothing)
        tracer_advection = merge(tracer_advection, tke_advection)
    end

    ocean_model = HydrostaticFreeSurfaceModel(; grid,
                                                buoyancy,
                                                closure,
                                                biogeochemistry,
                                                tracer_advection,
                                                momentum_advection,
                                                tracers,
                                                timestepper,
                                                free_surface,
                                                coriolis,
                                                forcing,
                                                boundary_conditions,
    clock = Clock{FT, FT, FT, FT}(0, Inf, Inf, FT(0), FT(0)))

    ocean = Simulation(ocean_model; Δt, verbose)

    return ocean
end