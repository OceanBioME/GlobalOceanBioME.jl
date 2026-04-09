# GlobalOceanBioME.jl
A collection of utilities for running OceanBioME with NumericalEarth. Everything in this repo is liable to change and/or move to OceanBioME or NumericalEarth

Provided is:
- a 1.5(ish) degree grid with holes in the bathymetry filled: `GlobalOceanBioME.one_p_five_degree_grid`
- some packaging to get wind speed (for gas exchange from the atmosphere): `CO₂_flux = CarbonDioxideGasExchangeBoundaryCondition(; air_concentration = 278,  wind_speed = wind_from_atmosphere(atmosphere))`
- river exports of DIN/DON/DOC (and a ~~guess~~ derived DIC/Alk/Fe): `GlobalOceanBioME.river_nutrients(grid)`
- iron deposition from dust: `DustCOMM_iron_deposition_boundary_condition()`
- light from the atmosphere: `biogeochemistry = LOBSTER(grid; surface_photosynthetically_available_radiation = PAR_from_atmosphere(atmosphere)))`

