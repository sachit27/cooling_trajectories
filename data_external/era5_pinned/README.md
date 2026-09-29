# Model-pinned ERA5 validation inputs

Retrieved from the Open-Meteo historical API on **2026-07-28** using
`pipeline/v3_00_fetch_era5.py`. Every request explicitly included
`models=era5`, `cell_selection=nearest`, UTC hourly data from 2020-06-01 through
2025-08-31, and cloud cover (total/low/mid/high), 2 m temperature and 10 m wind.

Open-Meteo's JSON response does not echo the requested model. Reproducibility
therefore rests on the retained fetch code, these immutable response files and
their checksums—not on an inferred model label from response metadata.

The five requested locations resolve to only **two distinct ERA5 grid cells**:
47.50°N, 8.50°E (centre/NW/NE) and 47.25°N, 8.50°E (SW/SE). The files are kept
per request for an auditable request-to-grid mapping, but `v3_10` de-duplicates
them before analysis. ERA5 can validate temporal cloud variation here; it is too
coarse to establish cloud homogeneity within the Zurich basin.

## SHA-256

```text
964c9b06587dcb905eb85aa56eb744d024f6bc5e13b5b4c82371621a37e82ecd  openmeteo_era5_zurich_centre.json
42d19a8c9dfa9ee26663d8d8107e145cada02b69649a4501c7ab70e842bb50ad  openmeteo_era5_zurich_ne.json
2e296e4c920a3e433b41d9e15a0cfa9b61c066b087ca82fe36821aed424ae34a  openmeteo_era5_zurich_nw.json
ac00c778f47e33f0e99b01cdc53c9238546ba136b00f09f466123551a44efe2a  openmeteo_era5_zurich_se.json
5387daa5a9aad571c837a7083b46ecc5c0cf9bdf82db76b3375f37911e33ed7c  openmeteo_era5_zurich_sw.json
```
