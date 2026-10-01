# Spatial input sources

The prepared screening grid combines population data from City/Canton Zurich,
canopy height from Meta/WRI Canopy Height Maps v2, buildings from swisstopo
swissBUILDINGS3D 2.0, and Canton Zurich Klimamodell 2024 status-quo night-temperature
and cold-air-flow fields. The retained population reference date is 31 December 2025.

Meta/WRI source: https://registry.opendata.aws/dataforgood-fb-forestsv2/
FITNAH source: https://geolion.zh.ch/geodatensatz/4989

FITNAH columns were verified against the source rasters using 5 x 5 pixel means.
Windows containing negative cold-air source codes are flagged in
`data_external/fitnah_coldflow_extraction_quality.csv` and excluded from cold-air
analyses. The checksum manifest identifies the input files used by the pipeline.
