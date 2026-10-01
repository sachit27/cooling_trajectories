"""Optional geographic validation with coverage adjustment fitted inside each training fold."""
from pathlib import Path
import sys,json,warnings
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
import os
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'pipeline'))
OUT=ROOT/'outputs_revision'
OUT.mkdir(exist_ok=True)
import v3_18_spatial_rebuild as S
p=S.load_analysis_panel();tg=pd.read_csv('outputs_robust/v3_spatial_rebuild_station_targets.csv');coords=tg[['EKoord','NKoord']].to_numpy();x=(coords-coords.mean(0))/1000;rows=[]
for rep in range(20):
 labels=KMeans(n_clusters=6,n_init=1,random_state=S.SEED+rep).fit_predict(x);pred=np.zeros(len(tg));truth=np.zeros(len(tg));sd=np.zeros(len(tg));se=np.zeros(len(tg))
 for fold in range(6):
  train=labels!=fold;test=~train;ids=tg.loc[train,'locationID'];pp=p[p.locationID.isin(ids)]
  tt,_=S.fit_two_way_fixed_effects(pp);tt=tt.set_index('locationID').loc[ids].reset_index()
  q=pp.merge(tt[['locationID','station_effect_c']],on='locationID');q['night_component']=q.night_min_c-q.station_effect_c;gamma=q.groupby('night_date').night_component.mean();base=float(gamma.mean())
  held=p[p.locationID.isin(tg.loc[test,'locationID'])].merge(gamma,on='night_date');held['adjusted']=held.night_min_c-held.night_component+base
  yy=held.groupby('locationID').adjusted.mean();aa=held.groupby(['locationID','year']).adjusted.mean();ss=aa.groupby('locationID').agg(lambda z:z.std(ddof=1)/np.sqrt(len(z)))
  model=S.fit_gp(x[train],tt.adjusted_mean_night_min_c.to_numpy(),tt.between_summer_se_c.to_numpy(),seed=(S.SEED+rep)*100+fold,restarts=0)
  pred[test],sd[test]=model.predict(x[test],return_std=True);truth[test]=yy.loc[tg.loc[test,'locationID']];se[test]=ss.loc[tg.loc[test,'locationID']]
 rows.append(dict(repeat=rep,r2=float(r2_score(truth,pred)),rmse=float(np.sqrt(np.mean((truth-pred)**2))),rho=float(stats.spearmanr(truth,pred).statistic),coverage=float(np.mean(np.abs(truth-pred)<=1.96*sd)),target_augmented_coverage=float(np.mean(np.abs(truth-pred)<=1.96*np.sqrt(sd**2+se**2))),mean_interval_width=float(np.mean(3.92*sd))))
 if rep in [0,9,19]:print('finished repeat',rep+1,flush=True)
d=pd.DataFrame(rows);d.to_csv(OUT/'rev_fold_contained_spatial_cv.csv',index=False);sm=d.drop(columns='repeat').median().to_dict();(OUT/'rev_fold_contained_gp_summary.json').write_text(json.dumps(sm,indent=2));print(json.dumps(sm,indent=2))
