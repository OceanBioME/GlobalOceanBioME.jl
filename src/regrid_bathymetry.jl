using NumericalEarth, Oceananigans, KernelAbstractions, Statistics, NCDatasets

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: x_domain, y_domain, topology

using NumericalEarth.Bathymetry: BathymetryRegridding, load_bathymetry_cache, save_bathymetry_cache

function MedianBathymetryRegridding(grid, metadata; label = "_median")
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)
    lon = x_domain(grid)
    lat = y_domain(grid)
    FT = eltype(grid)
    grid_type_name = string(typeof(grid).name.wrapper)
    dataset_name = string(typeof(metadata.dataset))*label

    return BathymetryRegridding(grid_type_name,
                                (Nx, Ny),
                                (Float64(lon[1]), Float64(lon[2])),
                                (Float64(lat[1]), Float64(lat[2])),
                                (Symbol(TX), Symbol(TY)),
                                Symbol(FT),
                                nothing,
                                Float64(Inf),
                                Int(1),
                                Float64(Inf),
                                dataset_name)
end

function regrid_bathymetry(target_grid::AbstractGrid{FT};
                           metadata = Metadatum(:bottom_height; dataset = ETOPO2022()),
                           cache = true) where FT

    config = MedianBathymetryRegridding(target_grid, metadata)

    if cache
        cached_data = load_bathymetry_cache(config)
        if !isnothing(cached_data)
            target_z = Field{Center, Center, Nothing}(target_grid)
            set!(target_z, cached_data)
            fill_halo_regions!(target_z)
            return target_z
        end
    end

    arch = CPU()#architecture(target_grid) # don't think this will work on GPU

    bathymetry_native_grid = NumericalEarth.DataWrangling.native_grid(metadata, arch; halo = (10, 10, 1))

    filepath = NumericalEarth.DataWrangling.metadata_path(metadata)
    dataset = Dataset(filepath, "r")
    z_data = convert(Array{FT}, dataset["z"][:, :])
    close(dataset)

    native_z = Field{Center, Center, Nothing}(bathymetry_native_grid)

    set!(native_z, z_data)
    fill_halo_regions!(native_z)

    target_z = on_architecture(CPU(), Field{Center, Center, Nothing}(target_grid))

    Oceananigans.Utils.launch!(CPU(), target_grid, :xy, median_bathymetry!, target_z, native_z, target_grid, bathymetry_native_grid)

    fill_halo_regions!(target_z)

    if cache
        bottom_height = Array(interior(target_z, :, :, 1))
        save_bathymetry_cache(config, bottom_height)
    end
    
    return target_z
end

@kernel function median_bathymetry!(target_z, native_z, target_grid, bathymetry_native_grid)
    i, j = @index(Global, NTuple)

    Nx = bathymetry_native_grid.Nx
    Ny = bathymetry_native_grid.Ny

    λ_native = bathymetry_native_grid.λᶜᵃᵃ
    φ_native = bathymetry_native_grid.φᵃᶜᵃ

    Δλ = bathymetry_native_grid.Δλᶜᵃᵃ
    Δφ = bathymetry_native_grid.Δφᵃᶜᵃ

    λ₀ = @inbounds λ_native[1]
    φ₀ = @inbounds φ_native[1]

    λl = Oceananigans.Grids.λnode(i,   j, target_grid, Face(), Center())
    λr = ifelse(j == target_grid.Ny, 
                Oceananigans.Grids.λnode(i+1, j-1, target_grid, Face(), Center()), 
                Oceananigans.Grids.λnode(i+1, j, target_grid, Face(), Center()))
    φl = Oceananigans.Grids.φnode(i,   j, target_grid, Center(), Face())
    φr = ifelse(j == target_grid.Ny, 
                Oceananigans.Grids.φnode(i+1, j-1, target_grid, Center(), Face()), 
                Oceananigans.Grids.φnode(i, j+1, target_grid, Center(), Face()))

    λl, λr = mod(λl, 360)-180, mod(λr, 360)-180 # remap to -180 to 180 space

    φl, φr = ifelse(j == target_grid.Ny, (min(φr, φl), max(φr, φl)), (φl, φr))
    
    # assuming a regularly spaced lat/lon grid for the native
    is = floor(Int, (λl - λ₀)/Δλ) + 1:floor(Int, (λr - λ₀)/Δλ)
    js = floor(Int, (φl - φ₀)/Δφ) + 1:floor(Int, (φr - φ₀)/Δφ)

    if λr < λl # 180E
        native_zs = @inbounds [native_z[floor(Int, (λl - λ₀)/Δλ) + 1:Nx, js, 1]..., native_z[1:floor(Int, (λr - λ₀)/Δλ), js, 1]...]
    else
        native_zs = @inbounds native_z[is, js, 1]
    end

    @inbounds target_z[i, j, 1] = !isempty(native_zs) ? median(native_zs) : zero(Δλ) # happens at fold point

    nothing
end
