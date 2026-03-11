import Oceananigans.TurbulenceClosures: calc_tapering, tapering_factor

# This somehow makes the model run ~2x slower

# Danabasoglu & McWilliams (1995)
@kwdef struct DanabasogluMcWilliams{FT}
           max_slope :: FT = 0.004
    transition_width :: FT = 0.001
end

@inline function calc_tapering(bx, by, bz, grid, slope_model, slope_limiter::DanabasogluMcWilliams)

    bz = max(bz, slope_model.minimum_bz)

    Sx = - bx / bz
    Sy = - by / bz

    S = sqrt(Sx^2 + Sy^2)

    limiter = (one(grid) + tanh((slope_limiter.max_slope - S)/slope_limiter.transition_width)) / 2

    return ifelse(bz <= 0, zero(grid), min(one(grid), limiter))
end

@inline function tapering_factor(Sx, Sy, slope_limiter::DanabasogluMcWilliams)
    S = sqrt(Sx^2 + Sy^2)
    limiter = (one(S) + tanh((slope_limiter.max_slope - S)/slope_limiter.transition_width)) / 2
    return min(one(S), limiter)
end