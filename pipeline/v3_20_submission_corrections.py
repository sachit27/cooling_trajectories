"""Targeted submission corrections after v3_14, v3_17 and the original analyses.

Recomputes calendar-time secondary inference, regression, duplicate weighting,
solar support and time-matched Fluntern agreement. Retains original study period
and primary dusk definition. Run before v3_13 publication figures.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import duckdb
import v3_14_trajectory_framework as trajectory
import v3_17_metrology_temporal as temporal
import v3_06_priority_screen as screen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs_robust'


def solar_hours(dates):
    """Original NOAA 365-day approximation, apparent 90.833-degree events."""
    idx = pd.DatetimeIndex(dates)
    g = 2*np.pi/365*(idx.dayofyear.to_numpy()-1)
    eq = 229.18*(.000075+.001868*np.cos(g)-.032077*np.sin(g)
                -.014615*np.cos(2*g)-.040849*np.sin(2*g))
    dec = (.006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)
           +.000907*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g))
    lat = np.deg2rad(47.3769)
    ha = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833))/(np.cos(lat)*np.cos(dec))
                             -np.tan(lat)*np.tan(dec)))
    return (720-4*(8.5417+ha)-eq)/60, (720-4*(8.5417-ha)-eq)/60


def matched_reference():
    con = duckdb.connect()
    raw = con.execute("""SELECT CAST(timestamp AS TIMESTAMPTZ) ts,
        TRY_CAST(value AS DOUBLE) t FROM read_parquet(?)
        WHERE locationID='Zürich-Fluntern' AND TRY_CAST(qc_flag AS INT)=0
        AND TRY_CAST(value AS DOUBLE) IS NOT NULL""",
        [str(ROOT/'heat/meteoblue_temperature.parquet')]).df()
    con.close()
    raw['ts'] = pd.to_datetime(raw.ts, utc=True)
    raw = raw[raw.ts.dt.year.between(2020, 2025)]
    assert not raw.ts.duplicated().any()
    raw['hour_end'] = raw.ts.dt.ceil('h')
    hourly = raw.groupby('hour_end').t.agg(['mean', 'size'])
    hourly = hourly[hourly['size'] == 4]
    smn = pd.read_csv(ROOT/'data'/'inputs'/'official_meteoswiss_hourly_zurich_synoptic.csv')
    smn = smn[smn.station_abbr == 'SMA'].copy()
    smn['hour_end'] = pd.to_datetime(smn.timestamp_utc, utc=True)
    matched = smn.merge(hourly, on='hour_end', validate='one_to_one')
    matched['hour_mid'] = ((matched.hour_end-pd.Timedelta(minutes=30))
                          .dt.tz_convert('Europe/Zurich').dt.tz_localize(None))
    matched = matched[matched.hour_mid.dt.month.isin([6, 7, 8])].copy()
    matched['night_date'] = (matched.hour_mid.dt.floor('D') -
        pd.to_timedelta((matched.hour_mid.dt.hour < 12).astype(int), unit='D'))
    matched = matched.merge(trajectory.sunset_local(matched.night_date), on='night_date')
    matched['hss'] = (matched.hour_mid-matched.sunset_local).dt.total_seconds()/3600
    rows = []
    for support, mask in [
        ('all_JJA_hours', np.ones(len(matched), bool)),
        ('night_20_08', (matched.hour_mid.dt.hour >= 20)|(matched.hour_mid.dt.hour < 8)),
        ('dusk_0_1', matched.hss.between(0, 1)),
    ]:
        d = matched[mask].dropna(subset=['mean', 'tre200h0'])
        error = d['mean']-d.tre200h0
        rows.append(dict(support=support, n=len(d), bias=error.mean(),
                         rmse=np.sqrt((error**2).mean()), mae=error.abs().mean(),
                         q025=error.quantile(.025), q975=error.quantile(.975)))
    return pd.DataFrame(rows)


def main():
    nights = temporal.load_nights().rename(columns={'city_mean_tmin_c': 'warmth'})
    metrics = pd.read_csv(OUT/'v3_17_dusk_metric_by_night.csv', parse_dates=['night_date'])
    print('Calendar-block trajectory and regression intervals', flush=True)
    assoc = temporal.trajectory_calendar_associations(nights)
    assoc.to_csv(OUT/'v3_17_trajectory_associations.csv', index=False)
    reg = temporal.all_night_calendar_block_regression(metrics, nights)
    reg.to_csv(OUT/'v3_17_calendar_block_regression.csv', index=False)
    print(assoc.to_string(index=False), flush=True)
    print(reg[reg.term == 'warmth_c'].to_string(index=False), flush=True)

    traj = pd.read_csv(ROOT/'outputs_v3/v3_traj_pernight.csv', parse_dates=['night_date'])
    primary = 'dusk_sd__dynamic_all__published_corrected_values__interval_midpoint'
    check = traj.merge(metrics[['night_date', primary]], on='night_date')
    np.testing.assert_allclose(check.dusk_sd, check[primary], atol=1e-12, equal_nan=True)
    _, sunset = solar_hours(nights.night_date)
    sunrise, _ = solar_hours(nights.night_date+pd.Timedelta(days=1))
    duration = 24+sunrise-sunset
    solar = nights[['night_date', 'regime']].assign(sunset_to_sunrise_hours=duration)
    solar.to_csv(OUT/'v3_20_solar_horizon.csv', index=False)

    specs = pd.read_csv(ROOT/'outputs_v3/v3_specification_curve.csv')
    unique = pd.concat([
        specs[specs['sample'].eq('all_nights')].drop_duplicates(
            ['night_window', 'spread_metric', 'adjustment_set', 'years', 'sample']),
        specs[specs['sample'].eq('calm_clear_only')],
    ])
    unique.to_csv(OUT/'v3_20_distinct_specifications.csv', index=False)
    grid = pd.read_csv(ROOT/'data'/'inputs'/'citywide_grid_priority_screen.csv').merge(
        pd.read_csv(OUT/'v3_spatial_rebuild_grid.csv')[[
            'recordid', 'gp_adjusted_mean_night_min_c',
            'gp_adjusted_mean_night_min_c_sd', 'idw2_adjusted_mean_night_min_c']],
        on='recordid', validate='one_to_one')
    masks, designs = [], []
    for field in ['rebuilt_gp', 'rebuilt_idw']:
        for name, weights in screen.WEIGHT_SCENARIOS.items():
            for lens, pop in screen.POPULATION_LENSES.items():
                score = screen.score(screen.components(grid, field, pop), weights).to_numpy()
                masks.append(screen.top_k_mask(score, .1))
                designs.append(dict(field=field, weights=name, population=lens,
                    signature=hashlib.sha256(np.round(score, 10).tobytes()).hexdigest()))
    designs = pd.DataFrame(designs)
    designs.to_csv(OUT/'v3_20_decision_duplicates.csv', index=False)
    masks = np.asarray(masks)
    keep = ~designs.signature.duplicated().to_numpy()
    freq = masks[keep].mean(axis=0)
    pd.DataFrame(dict(recordid=grid.recordid, baseline_top10=masks[0],
                     selection_frequency_distinct=freq)).to_csv(
                         OUT/'v3_20_distinct_decision_frequency.csv', index=False)
    matched = matched_reference()
    matched.to_csv(OUT/'v3_20_fluntern_matched_hourly.csv', index=False)
    print(matched.to_string(index=False), flush=True)
    summary = dict(
        n_study_nights=len(nights), n_complete9=int(traj.complete9.sum()),
        n_complete7=int(traj.complete7.sum()),
        exact_dusk_max_error=float(np.nanmax(np.abs(check.dusk_sd-check[primary]))),
        solar_after_sunrise=int((duration < 9).sum()),
        selected_after_sunrise=int(((duration < 9)&nights.regime.eq('calm_clear')).sum()),
        shortest_night_h=float(duration.min()), longest_night_h=float(duration.max()),
        nominal_specifications=len(specs), distinct_specifications=len(unique),
        distinct_median=float(unique.rho.median()), distinct_positive_pct=100*float((unique.rho>0).mean()),
        nominal_decisions=len(designs), distinct_decisions=int(keep.sum()),
        distinct_baseline_ge80=int((freq[masks[0]] >= .8).sum()),
        distinct_baseline_ge90=int((freq[masks[0]] >= .9).sum()),
    )
    (OUT/'v3_20_correction_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
