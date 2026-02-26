using Oceananigans, ClimaOcean

function one_degree_grid(arch = GPU(); Nx = 360, Ny = 180, Nz = 50, depth = 5000meters)
    z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4)
    underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

    bottom_height = regrid_bathymetry(underlying_grid; minimum_depth=5, interpolation_passes=10, major_basins=Inf)
    view(bottom_height, 31, 98, 1) .= 1000 # close Panama canal
    view(bottom_height, 35, 98, 1) .= 1000 # close Panama canal
    view(bottom_height, 289:290, 157, 1) .= -80 # open Bering strait
    view(bottom_height, 103:107, 123:124, 1) .= -1000 # open Gebralta strait
    view(bottom_height, 114, 147:148) .= -7 # open Baltic
    view(bottom_height, 111, 148) .= -7 # open Baltic
    view(bottom_height, 115, 157) .= 100 # close top of Baltic
    view(bottom_height, 152:153, 103:104) .= -10 # open Strait of Hormuz
    #view(bottom_height, 152:153, 103:104) .= -10 # open Bab el Mandeb - think its fine
    view(bottom_height, 72:73, 172:178) .= -10 # open Hudson Strait
    view(bottom_height, 53, 161) .= -10 # open Hudson Bays
    view(bottom_height, 61, 154) .= -10 # open Hudson Bays
    view(bottom_height, 212:215, 89:91) .= 1000 # close Malacca Strait
    view(bottom_height, 228, 77) .= -10 # open Indonesian Throughflow passages
    view(bottom_height, 232, 87) .= -10 # open Indonesian Throughflow passages
    view(bottom_height, 220, 84) .= -10 # open Indonesian Throughflow passages
    view(bottom_height, 191:192, 99:100) .= 1000 # close Palk Strait

    ClimaOcean.Bathymetry.remove_minor_basins!(bottom_height, 1)# close everything else

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                                active_cells_map=true)
end

function one_p_five_degree_grid(arch = GPU(); Nx = 256, Ny = 128, Nz = 32, depth = 6000, δ = 6)
    z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/δ)
    underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

    bottom_height = regrid_bathymetry(underlying_grid; minimum_depth=5, interpolation_passes=10, major_basins=Inf)

    view(bottom_height, 23, 70, 1) .= 1000 # close Central America
    view(bottom_height, 42, 98, 1) .= 1000 # close Cabot Strait
    view(bottom_height, 39, 114, 1) .= -200 # open Hudson bay
    view(bottom_height, 43, 113:114, 1) .= -200 
    view(bottom_height, 47:48, 121:126, 1) .= -200 # open Hudson straits
    view(bottom_height, 73:77, 87:88, 1) .= -100 # open Gebralta strait
    view(bottom_height, 76, 99, 1) .= 1000 # close English channel
    view(bottom_height, 79:81, 104:105, 1) .= -200 # open Baltic
    view(bottom_height, 107:109, 72:74, 1) .= -200 # open Strait of Hormuz
    view(bottom_height, 117, 84, 1) .= -200 # open Bab el Mandeb
    view(bottom_height, 136, 71, 1) .= 1000 # close Palk strait
    view(bottom_height, 150, 70, 1) .= -100 # open Indonesian Throughflow (north and gap)
    view(bottom_height, 165, 60:61, 1) .= -100
    view(bottom_height, 174, 86:87, 1) .= -100 # open Sea of Japan
    view(bottom_height, 204:205, 110:114, 1) .= -50 # open Bering strait

    ClimaOcean.Bathymetry.remove_minor_basins!(bottom_height, 1)# close everything else

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                                active_cells_map=true)
end

function three_degree_grid(arch = GPU(); Nx = 128, Ny = 64, Nz = 32, depth = 6000, δ = 6)
    z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/δ)
    underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

    bottom_height = regrid_bathymetry(underlying_grid; minimum_depth=5, interpolation_passes=10, major_basins=Inf)

    view(bottom_height, 12, 35) .= 1000 # close Central America
    view(bottom_height, 101:102, 56) .= -50 # open Bering strait
    view(bottom_height, 25:26, 61:64) .= -400 # open Hudson strait
    view(bottom_height, 26, 56) .= -500 
    view(bottom_height, 89, 27) .= 1000 # close Torres Strait


    ClimaOcean.Bathymetry.remove_minor_basins!(bottom_height, 1)# close everything else

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                                active_cells_map=true)
end