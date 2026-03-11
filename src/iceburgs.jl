using NetCDF

function load_icebergs(grid; times, path = "ICB_ORCA025.L75-GNJ002.nc")
    ib_times = ncread(path, "time_centered")
    lat = ncread(path, "nav_lat")
    lon = ncread(path, "nav_lon")
