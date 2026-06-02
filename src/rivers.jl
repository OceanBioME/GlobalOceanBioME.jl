using CSV, DataFrames, CUDA

using Oceananigans.Architectures: on_architecture, CPU

# Mg/yr elements (e.g. Mg N/yr)

const exports = (:DIN, :DIP, :DOC, :DON, :DOP, :DSi, :PN, :POC, :PP, :TSS)

# hydrology can be Qnat which is the estimated non-athropogenically disturbed outflow
# outflows are km^3/yr converted to kg/m^3/s so on flat grid kg/m^2/s
# Mg/yr to mmol / s : 10^6 = g/yr, 1/14 = mol/yr
function river_exports(grid; tracers = (:DIN, :DON, :DOC),#, :DIP, :DOP), 
                             hydrology = :Qact, 
                             scalefactor = (10^6/14, 10^6/14, 10^6/12, #10^6/15, 10^6/15, 
                                            10^9) ./ (365*24*60*60) .* 1000, 
                             spread=3) # not implemented
    all(tracers in (:DIN, :DON, :DOC, :DIP, :DOP)) && throw(ArgumentError("Haven't implemented this function correctly so have to use default tracers"))
    ((length(tracers) + !isnothing(hydrology)) == length(scalefactor)) || throw(ArgumentError("You need top supply scalefactors for all tracers ± hydrology"))

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

    λ = on_architecture(CPU(), λnodes(grid, Center(), Center(), Center()))
    φ = on_architecture(CPU(), φnodes(grid, Center(), Center(), Center()))

    points = collect(zip([φ...], [λ...]))
    CUDA.@allowscalar begin
        immersed = [[Oceananigans.ImmersedBoundaries.immersed_cell(i, j, grid.Nz, grid) for i=1:grid.Nx, j=1:grid.Ny]...]
    end

    forcing_fields = NamedTuple{(tracers..., hydrology, :DIC)}(map(n -> CenterField(grid), 1:length(tracers)+ifelse(isnothing(hydrology), 0, 1)+1))

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

            DIC_flux = ifelse(basins_df.basinid[n] == 1,
                              2.54e12/(365*24*60*60)/total_vol, # as ECCO for the Amazon
                              export_df[n, :Ld_DOC] * scalefactor[3] / total_vol / 0.4)

            forcing_fields[:DIC][i, j, k] += DIC_flux

            if !isnothing(hydrology)
                forcing_fields[hydrology][i, j, k] += hydrology_df[n, hydrology] * scalefactor[end] / total_vol 
            end
        end
    end
    # constant DIC to Fe ratio to get 1.45 Tg Fe yr−1 
    # MARBL does 1:1 DIC and Alk from rivers because reasons https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002647 # out of date
    # Estimate of 40% DOC vs 60% DIC from 10.1029/95GB02925 to give DIC and Alk # out of date
    # They also say its 1Gt total but NEWS2 doesn't add up to that its like 0.42Gt # out of date
    # Also, this gives a within range estimate for total DIC:DOC ratios from 10.1016/j.ecolind.2017.04.049
    # It is too much for the amazon so going to scale down like ECCO: https://gmd.copernicus.org/articles/19/867/2026/

    CUDA.@allowscalar begin
        total_DIC_flux =  Field(Integral(forcing_fields.DIC))[1, 1, 1] # mmol C/s
    end
    total_Fe_flux = 1.45e12 / (365*24*60*60) / 55.8 * 1000 # mmolFe/s
    Fe_flux_scalefactor = total_Fe_flux / total_DIC_flux

    Alk_flux = forcing_fields.DIC * 0.98 # as ECCO

    #Fe_flux = (forcing_fields.DIP + forcing_fields.DOP) * 3e-4 # as ECCO
    Fe_flux = forcing_fields.DIC * Fe_flux_scalefactor

    forcing_fields = merge(forcing_fields, (; Alk = Alk_flux,
                                              Fe  = Fe_flux))

    return forcing_fields
end

# these totals match the ones compared to in ECCO:
# 356TgC/yr DIC # this matches ECCO but is a bit low compared to other estimates
# 171TgC/yr DOC # same
# 1.3TgP/yr DIP # matches other estimats but not ECCO
# 0.6TgP/yr DOP (matches ECCO estimate, no comparison data)
# 22 TgN/yr DIN (slightly high compared to comparison data, bigger than ECCO)
# 11 TgN/yr DON
# Fe is 100x too small if I do 3e-4 ratio to P, so going back to constant ratio to DIC to give 1.45Tg/yr (which seems like a severe overestimate compared to other sources)
# their ratio should give around 4x higher than 1.45Tg/yr so I messed something up somewhere