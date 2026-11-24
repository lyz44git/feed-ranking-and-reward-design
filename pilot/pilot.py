import duckdb,sys,numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
d=sys.argv[1]; c=duckdb.connect()
R=f"read_csv_auto('{d}/log_random_4_22_to_5_08_pure.csv')"
S1=f"read_csv_auto('{d}/log_standard_4_08_to_4_21_pure.csv')"
V=f"read_csv_auto('{d}/video_features_basic_pure.csv')"
U=f"read_csv_auto('{d}/user_features_pure.csv')"
P="user_id%5=0"   # pilot users only
print('== duration buckets, random exposures, pilot users, tab 1')
for x in c.sql(f"""select case when duration_ms<=7000 then 'a<=7s' when duration_ms<=18000 then 'b 7-18s' when duration_ms<=60000 then 'c 18-60s' when duration_ms<=180000 then 'd 1-3min' else 'e >3min' end b,
 count(*) n, avg(is_click) valid_play, avg(long_view) long_view, avg(play_time_ms)/1000 play_s, avg(least(play_time_ms/nullif(duration_ms,0),1)) ratio, avg(is_like) lk, avg(is_hate)*1000 hate_per_k
 from {R} where {P} and tab=1 group by 1 order by 1""").fetchall(): print([round(v,4) if isinstance(v,float) else v for v in x])
def frame(t):
    return c.sql(f"""select l.user_id, l.is_click, l.long_view, l.play_time_ms/1000.0 play_s, l.tab, l.hourmin, v.video_duration/1000.0 dur, try_cast(split_part(v.tag,',',1) as int) tag1,
     (v.video_type='AD')::int ad, v.server_width w, v.server_height h, v.music_type mt, date_diff('day', try_cast(v.upload_dt as date), date '2022-04-22') age,
     u.follow_user_num, u.fans_user_num, u.friend_user_num, u.register_days, u.is_lowactive_period, u.is_live_streamer, u.is_video_author,
     u.onehot_feat0,u.onehot_feat1,u.onehot_feat2,u.onehot_feat3,u.onehot_feat4,u.onehot_feat5,u.onehot_feat6,u.onehot_feat7,u.onehot_feat8,u.onehot_feat9,u.onehot_feat10,u.onehot_feat11
     from {t} l join {V} v using(video_id) join {U} u using(user_id) where l.{P} and l.tab=1""").df()
r=frame(R); s=frame(S1); print('pilot rows random',len(r),'standard',len(s))
X=[k for k in r.columns if k not in('user_id','is_click','long_view','play_s','tab')]
rng=np.random.default_rng(0); users=r.user_id.unique(); te=set(rng.choice(users,len(users)//2,replace=False))
rt=r[~r.user_id.isin(te)]; rv=r[r.user_id.isin(te)]
def fit(tr,y): 
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,verbose=-1); m.fit(tr[X],tr[y]); return m
for y in ('is_click','long_view'):
    for name,tr in (('trained on standard',s[~s.user_id.isin(te)]),('trained on random',rt)):
        m=fit(tr,y); p=m.predict_proba(rv[X])[:,1]
        top=p>=np.quantile(p,0.9)
        print(y,name,'AUC on random holdout',round(roc_auc_score(rv[y],p),4),'| top10%: valid_play',round(rv.is_click[top].mean(),4),'long_view',round(rv.long_view[top].mean(),4),'play_s',round(rv.play_s[top].mean(),2),'mean dur',round(rv.dur[top].mean(),1),'| all:',round(rv.is_click.mean(),4),round(rv.long_view.mean(),4),round(rv.play_s.mean(),2),round(rv.dur.mean(),1), '| pred mean',round(p.mean(),4),'n top',int(top.sum()))
