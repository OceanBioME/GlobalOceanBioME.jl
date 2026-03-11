using Oceananigans, NumericalEarth, Statistics, CUDA, KernelAbstractions
using Oceananigans.Utils: launch!

one_p_five_fills = ((152, 69:70), # central america
                    (23, 71),     # Malacca Strait
                    (175, 123),   # artic archipeligo area
                    (174, 126),   # artic archipeligo area
                    (247, 85),    # persian gulf (Strait of Hormuz)
                    (164, 12))    # Alexander island hole

one_p_five_opens = ((170:171, 112:114), # Hudson straits
                    #(74:77, 88), # Gebralta straits
                    (208:210, 104:105), # Baltic
                    #(151, 70:71), # Indonesian Throughflow (north and gap)
                    #(165:166, 60:61), #
                    (47:48, 86:87), # Sea of Japan
                    (45:47, 82:83), # Sea of Japan
                    (39:41, 79:81), # Sea of Japan
                    (77:78, 110:114), #Bering strait
                    (236:237, 72:73)) # Bab el Mandeb

const one_p_five_diffusion_mask = begin
    diffusion_mask = zeros(256, 128)
    diffusion_mask[28:52, 43:63] .= 0.2
    diffusion_mask[64, 17] = 0.5
    diffusion_mask[132, 17] = 0.5
    #diffusion_mask[190:191, 17:19] .= 0.5
    #diffusion_mask[173:174, 24:25] .= 0.5
    # drake passage
    diffusion_mask[170:195, 15:30] .= 0.2 
    diffusion_mask[168:180, 23:29] .= 0.4
    diffusion_mask[132:145, 15:25] .= 0.2
    diffusion_mask[165, 15] = 0.2

    diffusion_mask[168:178, 23:28] .= 0.5
    # Southwest Indian Ridge
    diffusion_mask[225:235, 22:33] .= 0.2
    diffusion_mask
end

zf(k, p) = -p.z0 + p.Δz0 * k + p.Δz′ * log(cosh((k - p.k0)/p.kc)) * p.kc
Δzf(k, p) = p.Δz0 + p.Δz′ * tanh((k - p.k0)/p.kc) 

using NLsolve

# this is the same as NEMO ORCA: https://www.nemo-ocean.eu/doc/node21.html#DOM_zgr_coef
function compute_zs(; Nz = 42, H = 5000, kc = 3, Δzf0 = 5, Δzf1 = 200)
    function z_objective!(F, x)
        p = NamedTuple{(:z0, :Δz0, :Δz′, :k0, :kc, :Nz)}([x..., kc, Nz])

        F[1] = zf(1, p) + H
        F[2] = zf(Nz+1, p)
        F[3] = Δzf(1, p) - Δzf1
        F[4] = Δzf(Nz+1, p) - Δzf0
    end

    sol = @show nlsolve(z_objective!, [3000.0, 250.0, -250.0, 10.0])

    p = @show NamedTuple{(:z0, :Δz0, :Δz′, :k0, :kc, :Nz)}([sol.zero..., kc, Nz])

    zfa = map(k->zf(k, p), 1:Nz+1)
    zfa[end] = 0

    return zfa
end

function one_p_five_degree_grid(arch = GPU(), FT = Float64;
                                Nx = 256, Ny = 128,
                                Nz = 42, H = 5000, kc = 3, Δzf0 = 5, Δzf1 = 200,
                                interpolation_passes=3, 
                                diffusion_mask = one_p_five_diffusion_mask, 
                                diffusion_passes = 3,
                                bottom_type = GridFittedBottom)
    z = compute_zs(; Nz, H, kc, Δzf0, Δzf1)

    underlying_grid = TripolarGrid(arch, FT; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

    CUDA.@allowscalar begin
        bottom_height = on_architecture(CPU(), NumericalEarth.Bathymetry.maybe_extend_longitude(regrid_bathymetry(underlying_grid; minimum_depth=0, interpolation_passes, major_basins=Inf), Periodic()))
    end

    for (i, j) in one_p_five_fills
        bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j] .= 10000
    end

    bottom_height[bottom_height .> 0] .= 0

    for (i, j) in one_p_five_opens
        md = mean(bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j])
        bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j] .= md
    end

    if !isnothing(diffusion_mask)
        for _ in 1:diffusion_passes
            launch!(CPU(), underlying_grid, :xy, diffuse_bottom_height!, bottom_height, deepcopy(bottom_height), diffusion_mask)
        end
    end

    # fill in big
    @inbounds for i in 1:Nx, j in 1:Ny
        b  = bottom_height[i, j]
        b1 = bottom_height[i-1, j]
        b2 = bottom_height[i+1, j]
        b3 = bottom_height[i, j-1]
        b4 = bottom_height[i, j+1]

        n = sum(map(bn -> bn >= 0, (b1, b2, b3, b4)))

        if (b <= 0) & (b < 1.5 * (b1 + b2 + b3 + b4) / n)
            bottom_height[i, j] = 0.5 * b + 0.5 * (b1 + b2 + b3 + b4) / n
        end
    end

    NumericalEarth.Bathymetry.remove_minor_basins!(bottom_height, 1, (underlying_grid.Nx, underlying_grid.Ny))# close everything else

    bathymetry_final = Field{Center, Center, Nothing}(underlying_grid)
    set!(bathymetry_final, bottom_height[1:underlying_grid.Nx, 1:underlying_grid.Ny])

    return ImmersedBoundaryGrid(underlying_grid, bottom_type(bathymetry_final);
                                active_cells_map=true)
end

function one_degree_grid(arch = GPU(), FT = Float64; Nx = 512, Ny = 256, Nz = 42, depth = 6000, δ = 6, bathymetry = nothing, zstar = true)
    z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/δ, mutable = zstar)

    underlying_grid = TripolarGrid(arch, FT; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

    if isnothing(bathymetry)
        CUDA.@allowscalar begin
            bottom_height = on_architecture(CPU(), NumericalEarth.Bathymetry.maybe_extend_longitude(regrid_bathymetry(underlying_grid; minimum_depth=0, interpolation_passes=3, major_basins=Inf), Periodic()))
        end

     #=  for (i, j) in one_p_five_fills
            bottom_height[i, j] = 10000
        end

        bottom_height[bottom_height .> 0] .= 0

        for (i, j) in one_p_five_opens
            md = mean(bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j])
            bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j] .= md
        end
=#
      #  NumericalEarth.Bathymetry.remove_minor_basins!(bottom_height, 1, (underlying_grid.Nx, underlying_grid.Ny))# close everything else

        bathymetry_final = Field{Center, Center, Nothing}(underlying_grid)
        set!(bathymetry_final, bottom_height[1:underlying_grid.Nx, 1:underlying_grid.Ny])
    else
        bathymetry_final = Field{Center, Center, Nothing}(underlying_grid)
        set!(bathymetry_final, bathymetry)
    end

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry_final);
                                active_cells_map=true)
end

@kernel function diffuse_bottom_height!(bottom_height_updated, bottom_height, diffusion_strength)
    i, j = @index(Global, NTuple)

    @inbounds begin
        α = diffusion_strength[i, j]
        bottom_height_updated[i, j] =
            (1-α) * bottom_height[i, j] +
            α / 6 * (bottom_height[i+1, j] +
                     bottom_height[i, j+1] +
                     bottom_height[i-1, j] +
                     bottom_height[i, j-1] + 
                     bottom_height[i+1, j+1]/2 +
                     bottom_height[i-1, j+1]/2 +
                     bottom_height[i+1, j-1]/2 +
                     bottom_height[i-1, j-1]/2)
    end
end

# for diagnosing steepness
function R(bottom_height)
    R = zeros(size(bottom_height)...)
    for i in 2:grid.Nx-1, j in 2:grid.Ny-1
        Rx1 = ifelse(bottom_height[i-1, j] == 0, 0, -abs((bottom_height[i, j] - bottom_height[i-1, j])) / (bottom_height[i, j] + bottom_height[i-1, j]))
        Rx2 = ifelse(bottom_height[i+1, j] == 0, 0, -abs((bottom_height[i+1, j] - bottom_height[i, j])) / (bottom_height[i+1, j] + bottom_height[i, j]))

        Ry1 = ifelse(bottom_height[i, j-1] == 0, 0, -abs((bottom_height[i, j] - bottom_height[i, j-1])) / (bottom_height[i, j] + bottom_height[i, j-1]))
        Ry2 = ifelse(bottom_height[i, j+1] == 0, 0, -abs((bottom_height[i, j+1] - bottom_height[i, j])) / (bottom_height[i, j+1] + bottom_height[i, j]))

        R[i, j] = ifelse(bottom_height[i, j] == 0, 0, max(Rx1, Rx2, Ry1, Ry2))
    end

    return R
end