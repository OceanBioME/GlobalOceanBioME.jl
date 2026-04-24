using CSV, DataFrames, CUDA

using Oceananigans.Architectures: on_architecture, CPU

# Mg/yr elements (e.g. Mg N/yr)

const exports = (:DIN, :DIP, :DOC, :DON, :DOP, :DSi, :PN, :POC, :PP, :TSS)

# hydrology can be Qnat which is the estimated non-athropogenically disturbed outflow
# outflows are km^3/yr converted to kg/m^3/s so on flat grid kg/m^2/s
# Mg/yr to mmol / s : 10^6 = g/yr, 1/14 = mol/yr
function river_exports(grid; tracers = (:DIN, :DON, :DOC), hydrology = :Qact, scalefactor = (10^6/14, 10^6/14, 10^6/12, 10^9) ./ (365*24*60*60) .* 1000)
    (tracers != (:DIN, :DON, :DOC)) && throw(ArgumentError("Haven't implemented this function correctly so have to use default tracers"))
    basins_df = CSV.read("rivers/basins.csv", DataFrame)
    export_df = CSV.read("rivers/exports.csv", DataFrame)
    hydrology_df = CSV.read("rivers/hydrology.csv", DataFrame)

    # throw away the yield
    select!(export_df, Not([map(e->Symbol(:Yld_, e), exports)...]))

    # throw away the rivers that don't do anyting
    export_df = filter(row -> !all(map(e->row[Symbol(:Ld_, e)] == 0, exports)), export_df)
    basins_df = filter(row -> row.basinid in export_df.basinid, basins_df)
    hydrology_df = filter(row -> row.basinid in export_df.basinid, hydrology_df)

    # throw away the enclosed seas
    basins_df = filter(row -> (row.sea != "Land") & (row.sea != "Black Sea") & (row.sea != "Caspian Sea") & (row.sea != "Aral Sea"), basins_df)
    export_df = filter(row -> row.basinid in basins_df.basinid, export_df)
    hydrology_df = filter(row -> row.basinid in basins_df.basinid, hydrology_df)

    mouth_is = zeros(size(export_df, 1))
    mouth_js = zeros(size(export_df, 1))

    λ = on_architecture(CPU(), λnodes(grid, Center(), Center(), Center()))
    φ = on_architecture(CPU(), φnodes(grid, Center(), Center(), Center()))

    points = collect(zip([φ...], [λ...]))
    CUDA.@allowscalar begin
        immersed = [[Oceananigans.ImmersedBoundaries.immersed_cell(i, j, grid.Nz, grid) for i=1:grid.Nx, j=1:grid.Ny]...]
    end

    is = zeros(size(export_df, 1))
    js = zeros(size(export_df, 1))

    forcing_fields = NamedTuple{(tracers..., hydrology)}(map(n -> CenterField(grid), 1:length(tracers)+ifelse(isnothing(hydrology), 0, 1)))
    # TODO: do this properly
    CUDA.@allowscalar for n in 1:size(export_df, 1)
        lat = basins_df.mouth_lat[n]
        lon = mod(basins_df.mouth_lon[n], 360)

        objective = [(sqrt((lat-node[1])^2 + (lon-mod(node[2], 360))^2) + immersed[m] * Inf) for (m, node) in enumerate(points)]

        list_index = findmin(objective)[2]

        # wow this is horrible
        i, j = Tuple(CartesianIndices((grid.Nx, grid.Ny))[list_index])
        total_vol = 0.0
        kb = grid.Nz
        while !Oceananigans.ImmersedBoundaries.immersed_cell(i, j, kb, grid)
            total_vol += Oceananigans.Operators.volume(i, j, grid.Nz, grid, Center(), Center(), Center())
            kb -= 1
        end

        for k in grid.Nz:-1:kb
            for (m, tracer) in enumerate(tracers)
                forcing_fields[tracer][i, j, k] += export_df[n, Symbol(:Ld_, tracer)] * scalefactor[m] / total_vol 
            end

            if !isnothing(hydrology)
                forcing_fields[hydrology][i, j, k] += hydrology_df[n, hydrology] * scalefactor[end] / total_vol 
            end
        end
    end
    # constant DIC to Fe ratio to get 1.45 Tg Fe yr−1
    # MARBL does 1:1 DIC and Alk from rivers because reasons https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002647
    # Estimate of 40% DOC vs 60% DIC from 10.1029/95GB02925 to give DIC and Alk
    # They also say its 1Gt total but NEWS2 doesn't add up to that its like 0.42Gt

    CUDA.@allowscalar begin
        total_DIC_flux =  Field(Integral(forcing_fields.DOC))[1, 1, 1] * 0.6 / 0.4 # mmol C/s
    end
    total_Fe_flux = 1.45e12 / (365*24*60*60) / 55.8 * 1000 # mmolFe/s
    Fe_flux_scalefactor = total_Fe_flux / total_DIC_flux

    DIC_Alk_flux = Field(forcing_fields.DOC * 0.6 / 0.4)

    forcing_fields = merge(forcing_fields, (; DIC = DIC_Alk_flux,
                                              Alk = DIC_Alk_flux,
                                              Fe  = DIC_Alk_flux * Fe_flux_scalefactor))

    return forcing_fields
end