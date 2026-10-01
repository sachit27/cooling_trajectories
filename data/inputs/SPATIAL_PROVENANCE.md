# Prepared spatial input

The retained citywide grid is a terminal, checksummed input. Project records
identify population/age data from Canton of Zurich and City neighbourhood totals,
canopy derived from swissSURFACE3D, buildings from swissBUILDINGS3D 2.0, and
FITNAH night-temperature/cold-air-flow context. Exact source releases, original
rasters, canopy thresholds, and parts of the upstream joins and aggregation
were not retained. The current FITNAH catalogue cannot identify the release
used retrospectively. Downstream scaling and temperature interpolation are
in the provided code; upstream layer construction is not fully reproducible.
