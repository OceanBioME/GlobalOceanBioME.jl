using NumericalEarth, Oceananigans, KernelAbstractions, Statistics, NCDatasets

using Oceananigans.Architectures: architecture

function regrid_bathymetry(target_grid::AbstractGrid{FT}, dataset = ETOPO2022();
                           metadata = Metadatum(:bottom_height; dataset = ETOPO2022())) where FT

    arch = CPU()#architecture(target_grid) # don't think this will work on GPU

    bathymetry_native_grid = NumericalEarth.DataWrangling.native_grid(metadata, arch; halo = (10, 10, 1))

    filepath = NumericalEarth.DataWrangling.metadata_path(metadata)
    dataset = Dataset(filepath, "r")
    z_data = convert(Array{FT}, dataset["z"][:, :])
    close(dataset)

    native_z = Field{Center, Center, Nothing}(bathymetry_native_grid)

    set!(native_z, z_data)
    Oceananigans.BoundaryConditions.fill_halo_regions!(native_z)

    target_z = on_architecture(CPU(), Field{Center, Center, Nothing}(target_grid))

    Oceananigans.Utils.launch!(CPU(), target_grid, :xy, median_bathymetry!, target_z, native_z, target_grid, bathymetry_native_grid)

    Oceananigans.BoundaryConditions.fill_halo_regions!(target_z)
    
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

    φl, φr = ifelse(j == target_grid.Ny, (min(φr, φl), max(φr, φl)), (φl, φr))
    
    # assuming a regularly spaced lat/lon grid for the native
    is = floor(Int, (λl - λ₀)/Δλ) + 1:floor(Int, (λr - λ₀)/Δλ)
    js = floor(Int, (φl - φ₀)/Δφ) + 1:floor(Int, (φr - φ₀)/Δφ)

    native_zs = @inbounds native_z[is, js, 1]

    if λr < λl # 180E
        native_zs = @inbounds [native_z[floor(Int, (λl - λ₀)/Δλ) + 1:Nx, js, 1]..., native_z[1:floor(Int, (λr - λ₀)/Δλ), js, 1]...]
    end

    target_z[i, j, 1] = !isempty(native_zs) ? median(native_zs) : zero(Δλ) # happens at fold point

    nothing
end
