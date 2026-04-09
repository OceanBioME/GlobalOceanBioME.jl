using Oceananigans, NumericalEarth, Statistics, CUDA, KernelAbstractions
using Oceananigans.Utils: launch!

one_p_five_fills = ((152:153, 70-1), # central america
                    #(23, 71),     # Malacca Strait
                    (175, 123),   # artic archipeligo area
                    (174, 126),   # artic archipeligo area
                    (247, 85),    # persian gulf (Strait of Hormuz)
                    )#(164, 12))    # Alexander island hole

one_p_five_opens = ((170:171, 112:114), # Hudson straits
                    #(74:77, 88), # Gebralta straits
                    (208, 104+1:105+1), # Baltic 
                    (209:211, 104), # Baltic 
                    #(151, 70:71), # Indonesian Throughflow (north and gap)
                    #(165:166, 60:61), #
                    #(47:48, 86:87), # Sea of Japan
                    #(45:47, 82:83), # Sea of Japan
                    #(39:41, 79:81), # Sea of Japan
                    (77:78, 110:114), #Bering strait
                    (237:238, 72:73), # Bab el Mandeb
                    (236:237, 72+1:73+1)) # Bab el Mandeb

# these are out of date
one_p_five_opens_refined = 
    ((171:172, 113:115), # Hudson straits
     (180, 123:125), # artic archipeligo area
     (207:212, 106), # Baltic 
     # Maybe this is too open:
     (237:238, 72+4:73+4), # Bab el Mandeb 
     (236:237, 72+1+4:73+1+4)) # Bab el Mandeb

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

    sol = nlsolve(z_objective!, [3000.0, 250.0, -250.0, 10.0])

    p = NamedTuple{(:z0, :Δz0, :Δz′, :k0, :kc, :Nz)}([sol.zero..., kc, Nz])

    zfa = map(k->zf(k, p), 1:Nz+1)
    zfa[end] = 0

    return zfa
end

function one_p_five_degree_grid(arch = GPU(), FT = Float64;
                                Nx = 256, Ny = 128,
                                Nz = 42, H = 5000, kc = 3, Δzf0 = 5, Δzf1 = 200,
                                halo = (5, 5, 4),
                                shapiro_passes = 2,
                                shapiro_strength = 0.1,
                                bottom_type = GridFittedBottom,
                                refine_equator = false,
                                metadata = Metadatum(:bottom_height; dataset = ETOPO2022()),
                                cache = true,
                                overwrite_cache = false,
                                fill_points = (refine_equator ? (1:-1) : one_p_five_fills),
                                open_points =  (refine_equator ? one_p_five_opens_refined : one_p_five_opens),
                                R_criteria_smooth = false,
                                zstar = false)
    z = compute_zs(; Nz, H, kc, Δzf0, Δzf1)

    z = zstar ? MutableVerticalDiscretization(z) : z

    #=φ_transformation = refine_equator ? 
                       Oceananigans.OrthogonalSphericalShellGrids.ArctanRefinedEquator() :
                       nothing=#

    underlying_grid = TripolarGrid(arch, FT; size = (Nx, Ny, Nz), halo, z)#, φ_transformation)

    config = MedianBathymetryRegridding(underlying_grid, metadata; label = "_median_smoothed$(shapiro_passes)*$(shapiro_strength)_and_filled")

    if cache
        cached_data = load_bathymetry_cache(config)
        if !isnothing(cached_data)
            target_z = Field{Center, Center, Nothing}(underlying_grid)
            set!(target_z, cached_data)
            fill_halo_regions!(target_z)
            return ImmersedBoundaryGrid(underlying_grid, bottom_type(target_z);
                                        active_cells_map=true)
        end
    end

    CUDA.@allowscalar begin
        bottom_height = on_architecture(CPU(), NumericalEarth.Bathymetry.maybe_extend_longitude(GlobalOceanBioME.regrid_bathymetry(underlying_grid; metadata, cache, overwrite_cache), Periodic()))
    end

    for _ in 1:shapiro_passes
        launch!(CPU(), underlying_grid, :xy, shapiro_filter!, deepcopy(bottom_height), bottom_height, shapiro_strength)
    end 

    for (i, j) in fill_points
        bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j] .= 10000
    end

    bottom_height[bottom_height .> 0] .= 0

    for (i, j) in open_points
        md = mean(bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j])
        bottom_height[((i isa Number) ? (i:i) : i), (j isa Number) ? (j:j) : j] .= md
    end

    launch!(CPU(), underlying_grid, :xy, shapiro_filter!, deepcopy(bottom_height), bottom_height, shapiro_strength)

    NumericalEarth.Bathymetry.remove_minor_basins!(bottom_height, 1, (underlying_grid.Nx, underlying_grid.Ny))# close everything else

    if R_criteria_smooth
        for _ in 1:2
            launch!(CPU(), underlying_grid, :xy, shapiro_filter_R_criteria!, deepcopy(bottom_height), bottom_height, 1)
        end
    end

    bathymetry_final = Field{Center, Center, Nothing}(underlying_grid)
    set!(bathymetry_final, bottom_height[1:underlying_grid.Nx, 1:underlying_grid.Ny])

    if cache|overwrite_cache
        bottom_height = Array(interior(bathymetry_final, :, :, 1))
        save_bathymetry_cache(config, bottom_height)
    end

    return ImmersedBoundaryGrid(underlying_grid, bottom_type(bathymetry_final);
                                active_cells_map=true)
end

@kernel function shapiro_filter!(ϕ, ϕn, shapiro_strength = 0.1)
    i, j = @index(Global, NTuple)
          
    @inbounds ϕij = ϕ[i, j]

    if ϕij < 0
        ϕn[i, j] = (1 - shapiro_strength) * ϕij + shapiro_strength * (ϕ[i+1, j] + ϕ[i-1, j] + ϕ[i, j+1] + ϕ[i, j-1])/4
    end

    nothing
end

@kernel function shapiro_filter_R_criteria!(ϕ, ϕn, shapiro_strength = 0.1, R_limit = 0.4)
    i, j = @index(Global, NTuple)
          
    @inbounds begin
        ϕij = ϕ[i, j]

        Rx1 = (ϕ[i-1, j] < 0) * (abs(ϕ[i, j] - ϕ[i-1, j]) / abs(ϕ[i, j] + ϕ[i-1, j]))
        Rx2 = (ϕ[i+1, j] < 0) * (abs(ϕ[i, j] - ϕ[i+1, j]) / abs(ϕ[i, j] + ϕ[i+1, j]))
        Ry1 = (ϕ[i, j-1] < 0) * (abs(ϕ[i, j] - ϕ[i, j-1]) / abs(ϕ[i, j] + ϕ[i, j-1]))
        Ry2 = (ϕ[i, j+1] < 0) * (abs(ϕ[i, j] - ϕ[i, j+1]) / abs(ϕ[i, j] + ϕ[i, j+1]))

        R = max(Rx1, Rx2, Ry1, Ry2)

        if (ϕij < 0) & (R > R_limit)
            ϕn[i, j] = (1 - shapiro_strength) * ϕij + shapiro_strength * (ϕ[i+1, j] + ϕ[i-1, j] + ϕ[i, j+1] + ϕ[i, j-1])/4
        end

    end
    nothing
end

# for diagnosing steepness
function R(bottom_height, grid)
    R = zeros(size(bottom_height)...)
    for i in 2:grid.Nx-1, j in 2:grid.Ny-1
        Rx1 = ifelse(bottom_height[i-1, j] == 0, 0, abs((bottom_height[i, j] - bottom_height[i-1, j])) / abs(bottom_height[i, j] + bottom_height[i-1, j]))
        Rx2 = ifelse(bottom_height[i+1, j] == 0, 0, abs((bottom_height[i+1, j] - bottom_height[i, j])) / abs(bottom_height[i+1, j] + bottom_height[i, j]))

        Ry1 = ifelse(bottom_height[i, j-1] == 0, 0, abs((bottom_height[i, j] - bottom_height[i, j-1])) / abs(bottom_height[i, j] + bottom_height[i, j-1]))
        Ry2 = ifelse(bottom_height[i, j+1] == 0, 0, abs((bottom_height[i, j+1] - bottom_height[i, j])) / abs(bottom_height[i, j+1] + bottom_height[i, j]))

        R[i, j] = ifelse(bottom_height[i, j] == 0, 0, max(Rx1, Rx2, Ry1, Ry2))
    end

    return R
end