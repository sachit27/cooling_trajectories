"""Check the manuscript against the analysis outputs.

1. Every main and supplementary figure/table is cited, in numbered order.
2. Key numbers quoted in the text are found in the outputs (76 automatic checks)
   and further values are printed for manual comparison.
Run from the repository root: python3 checks/verify_manuscript.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = os.path.join(ROOT, "manuscript")
print("== Cross-references ==")
import re
raw=open(os.path.join(MS,"main.tex")).read()+"\n"+open(os.path.join(MS,"results_body.tex")).read()
raw="\n".join(re.sub(r'(?<!\\)%.*','',l) for l in raw.split("\n"))
body=re.sub(r"\s+"," ",raw)
# main figures/tables in order of environment
envs=re.findall(r'\\begin\{(figure|table)\}.*?\\label\{([^}]+)\}',body,re.S)
refs=[m.group(1) for m in re.finditer(r'\\ref\{([^}]+)\}',body)]
# first reference position, excluding refs inside captions
nocap=re.sub(r'\\caption\{(?:[^{}]|\{[^{}]*\})*\}','',body)
first={}
for m in re.finditer(r'\\ref\{([^}]+)\}',nocap):
    first.setdefault(m.group(1),m.start())
envpos={lab:body.find('\\label{'+lab+'}') for _,lab in envs}
for kind in ['figure','table']:
    labs=[l for k,l in envs if k==kind]
    print(kind,'numbering order:',labs)
    order=sorted([l for l in labs if l in first], key=lambda l:first[l])
    print('  first-cited order:',order, 'OK' if order==labs else 'MISMATCH')
    for l in labs:
        if l not in first: print('  NOT CITED in text:',l)
# captions referencing other figures
for m in re.finditer(r'\\caption\{((?:[^{}]|\{[^{}]*\})*)\}',body):
    if re.search(r'Fig\.|Figure|Table~|\\ref',m.group(1)): print('CAPTION XREF:',m.group(1)[:80])
# supplementary
sup=open(os.path.join(MS,'supplementary.tex')).read()
nt=len(re.findall(r'\\begin\{table\}',sup)); nf=len(re.findall(r'\\begin\{figure\}',sup))
st=[int(x) for x in re.findall(r'Supplementary Tables?~S(\d+)',nocap)]
sf=[int(x) for x in re.findall(r'Supplementary Fig\.~S(\d+)',nocap)]
def firsts(l):
    out=[]
    for x in l:
        if x not in out: out.append(x)
    return out
print('supp tables in file',nt,'cited first-order',firsts(st))
print('supp figs in file',nf,'cited first-order',firsts(sf))
print('missing tables',set(range(1,nt+1))-set(st),'missing figs',set(range(1,nf+1))-set(sf))
# sequence check combined

print("\n== Numbers ==")
import json, re, pandas as pd, numpy as np
R=ROOT+'/'
t=open(os.path.join(MS,'main.tex')).read()+open(os.path.join(MS,'results_body.tex')).read()+open(os.path.join(MS,'supplementary.tex')).read()
T=re.sub(r'\s+',' ',t)
S=json.load(open(R+'outputs_revision/rev_summary.json'))
V=json.load(open(R+'outputs_v3/v3_verified_numbers.json'))
v=lambda k: V[k]['value']
sc=pd.read_csv(R+'outputs_v3/v3_specification_curve.csv')
ass=pd.read_csv(R+'outputs_revision/rev_dusk_window_associations.csv')
prim=pd.read_csv(R+'outputs_robust/v3_17_dusk_associations.csv')
dec=pd.read_csv(R+'outputs_revision/rev_decomposition.csv')
tr=pd.read_csv(R+'outputs_revision/rev_station_trait_models.csv')
vent=pd.read_csv(R+'outputs_revision/rev_ventilation_summary.csv')
cv=pd.read_csv(R+'outputs_robust/v3_spatial_rebuild_cv_metrics.csv'); cvm=cv[cv.row_type=='median'].iloc[0]; q25=cv[cv.row_type=='q025'].iloc[0]; q975=cv[cv.row_type=='q975'].iloc[0]
diag=pd.read_csv(R+'outputs_robust/v3_spatial_rebuild_diagnostics.csv').set_index('metric').numeric_value
reg=pd.read_csv(R+'outputs_robust/v3_17_calendar_block_regression.csv').set_index('term')
hac=pd.read_csv(R+'outputs_robust/v3_17_calendar_hac_models.csv')
traj=pd.read_csv(R+'outputs_robust/v3_17_trajectory_associations.csv')
prof=pd.read_csv(R+'outputs_revision/rev_profile_sunset.csv')
fit=pd.read_csv(R+'outputs_revision/rev_functional_fit.csv')
ms=pd.read_csv(R+'outputs_v3/v3_priority_decision_multiverse_summary.csv')
fl=pd.read_csv(R+'outputs_robust/v3_20_fluntern_matched_hourly.csv').set_index('support')
checks=[]
def c(label, val, fmt, must=None):
    s=(must or fmt).format(val) if must is None else must
    s=fmt.format(val)
    checks.append((label, s, s in T))
c('spec min', sc.rho.min(), '{:.3f}'); c('spec max', sc.rho.max(), '{:.3f}')
c('spec median', sc.rho.median(), '{:.3f}'); c('spec pos%', (sc.rho>0).mean()*100, '{:.1f}\\%')
c('unadj median', sc[sc.adjustment_set=='unadjusted'].rho.median(), '{:.3f}')
for a in ['day_radiation','dayrad+wind','full(dayrad+clear+wind)','full+precip']: print('adj',a, round(sc[sc.adjustment_set==a].rho.median(),3))
for a in ['clearness','clearness+wind']: print('adj',a, round(sc[sc.adjustment_set==a].rho.median(),3))
def a_(scope,win,adj,an):
    return ass[(ass.scope==scope)&(ass.window==win)&(ass.adjust==adj)&(ass.analysis==an)].iloc[0]
for scope in ['development','evaluation']:
    r=a_(scope,'dusk_0_1','day_radiation','dusk_partial'); c('partial rad '+scope, r.rho,'{:.3f}'); c('lo',r.ci_low,'{:.3f}'); c('hi',r.ci_high,'{:.3f}')
    r=a_(scope,'dusk_0_1','day_radiation+season+year','dusk_partial'); c('partial rad+ '+scope, r.rho,'{:.3f}')
    for w in ['pre_m1_0','w_3_5']:
        r=a_(scope,w,'none','window'); c(w+scope, r.rho,'{:.3f}')
    r=a_(scope,'dusk_0_1','none','dusk_vs_day_radiation'); c('rad corr '+scope, r.rho,'{:.2f}')
p=prim[(prim.estimand=='spearman')&(prim.station_set=='dynamic_all')&(prim.radiation_correction_policy=='published_corrected_values')&(prim.timestamp_convention=='interval_midpoint')]
for sc_ in ['development','internal_temporal_evaluation']:
    r=p[p.analysis_scope==sc_].iloc[0]; c('dusk '+sc_, r.rho,'{:.3f}'); c('lo',r.ci_low,'{:.3f}'); c('hi',r.ci_high,'{:.3f}')
for q in ['sd_t_even','sd_tmin','sd_cool_total']:
    r=dec[dec.column==q].iloc[0]; c(q,r.rho,'{:.3f}'); c(q+'lo',r.ci_low,'{:.3f}'); c(q+'hi',r.ci_high,'{:.3f}'); c(q+'cool',r.sd_cool_tercile,'{:.2f}'); c(q+'warm',r.sd_warm_tercile,'{:.2f}')
c('partial ct', dec[dec.column=='partial'].rho.iloc[0], '{:.3f}')
c('reg warmth', reg.loc['warmth_c','estimate'],'{:.3f}'); c('reg lo', reg.loc['warmth_c','ci_low'],'{:.3f}'); c('reg hi', reg.loc['warmth_c','ci_high'],'{:.3f}')
c('R2', hac.r_squared.iloc[0]*100,'{:.0f}\\%')
for m,s_ in [('integrated_0_9h','development'),('integrated_0_9h','internal_temporal_evaluation'),('post_peak_slope_0_9h','development'),('post_peak_slope_0_9h','internal_temporal_evaluation'),('integrated_0_7h','development'),('integrated_0_7h','internal_temporal_evaluation')]:
    r=traj[(traj.metric==m)&(traj.analysis_scope==s_)].iloc[0]; c(m+s_, r.rho, '{:.3f}')
for t_ in ['warm','cool']:
    g=prof[prof.tercile==t_].set_index('offset_h').sd_mean
    for o in [-2.0,0.0,9.0]: c(f'prof {t_} {o}', g.loc[o], '{:.2f}')
g=prof[prof.tercile=='cool'].set_index('offset_h').sd_mean; c('cool peak',g.max(),'{:.2f}'); print('cool peak at',g.idxmax())
for lab in ['warm (90th pct)','cool (10th pct)']:
    f=fit[fit.warmth_level==lab].set_index('hss').sd_fit; c(lab+' max',f.max(),'{:.2f}'); c(lab+' drop',f.max()-f.loc[9.0],'{:.2f}'); print(lab,'argmax',f.idxmax())
m=tr[(tr.model=='three')]
for tg,p_ in [('dusk_anom','bldg_frac'),('integ_anom','bldg_frac'),('decay_tend','canopy')]:
    r=m[(m.target==tg)&(m.predictor==p_)].iloc[0]; c('dR2 '+tg, r.delta_r2_median,'{:.2f}'); print(tg,'cvR2',round(r.cv_r2_median,2))
print('four dusk R2', round(tr[(tr.model=='four')&(tr.target=='dusk_anom')].cv_r2_median.iloc[0],2))
v22=vent[vent.threshold_c==22].iloc[0]
for k,f in [('city_median_delay_h','{:.2f}'),('rho_bldg','{:.2f}'),('delay_low_bldg_quartile','{:.1f}'),('delay_high_bldg_quartile','{:.2f}'),('rho_elev','{:.2f}'),('rho_canopy','{:.2f}'),('share_censored','{:.2f}')]:
    print('vent',k,f.format(v22[k]))
print('vent p10/p90', v22.station_median_delay_p10, v22.station_median_delay_p90)
c('cv R2', cvm.r2,'{:.2f}'); c('cv lo', q25.r2,'{:.2f}'); c('cv hi', q975.r2,'{:.2f}'); c('rmse',cvm.rmse_c,'{:.2f}'); c('rho',cvm.spearman_rho,'{:.2f}')
c('cov', cvm.gp_coverage95*100,'{:.1f}\\%'); c('width', cvm.gp_mean_interval_width_c,'{:.2f}')
c('ls', diag['matern_length_scale'],'{:.1f}'); c('range', diag['matern_correlation_0p05_range'],'{:.1f}'); c('noise',diag['white_noise_sd_c'],'{:.2f}')
c('median dist', diag['grid_nearest_station_median'],'{:.0f}'); print('p90 dist', diag['grid_nearest_station_p90'])
print('raw vs adj', diag['raw_vs_adjusted_station_spearman'])
for k in ['fluntern_reference_n','fluntern_reference_mean_bias_c','fluntern_reference_pearson_r','koller_n_nights','koller_mean_diff_c','koller_rmse_diff_c']:
    print(k, v(k))
print('flu dusk', fl.loc['dusk_0_1'].to_dict() if 'dusk_0_1' in fl.index else fl.index.tolist())
print('fluntern shift', {k:S['fluntern'][k] for k in ['best_shift_evening_min','rmse_evening_at_0','rmse_evening_at_best','bias_evening_at_0','bias_evening_at_best','evening_diff_vs_dayrad_rho','n_days']})
print('hour diffs', S['fluntern']['hour_mean_diff'])
print('screen', S['dusk_field_screen'], S['share65_screen'], S['indicator_rho'])
print('fitnah', S['fitnah'])
print(ms[ms.top_percent==10][['scope','n_baseline_frequency_ge_0_80','n_baseline_frequency_ge_0_90']])
bad=[x for x in checks if not x[2]]
print(len(checks),'automatic checks;',len(bad),'not found:'); [print('  ',b) for b in bad]
