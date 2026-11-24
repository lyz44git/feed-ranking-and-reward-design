import duckdb,sys
d=sys.argv[1]
c=duckdb.connect()
R=f"read_csv_auto('{d}/log_random_4_22_to_5_08_pure.csv')"
S1=f"read_csv_auto('{d}/log_standard_4_08_to_4_21_pure.csv')"
S2=f"read_csv_auto('{d}/log_standard_4_22_to_5_08_pure.csv')"
def q(s): 
    r=c.sql(s); print(r.columns); 
    for x in r.fetchall(): print([round(v,4) if isinstance(v,float) else v for v in x])
for n,t in (('random',R),('std1',S1),('std2',S2)):
    print('==',n); q(f"select count(*) n,count(distinct user_id) u,count(distinct video_id) v,min(date),max(date),avg(is_rand) rnd,avg(is_click) clk,avg(long_view) lv,avg(is_like) lk,avg(is_follow) fo,avg(is_hate) hate,avg(play_time_ms)/1000 pt,median(play_time_ms)/1000 mpt,avg(duration_ms)/1000 dur from {t}")
    q(f"select tab,count(*) n,avg(is_click) clk,avg(long_view) lv,avg(play_time_ms)/1000 pt from {t} group by 1 order by n desc limit 8")
print('== per-user counts'); 
for n,t in (('random',R),('std1',S1),('std2',S2)):
    q(f"with a as (select user_id,count(*) k from {t} group by 1) select '{n}',avg(k),quantile_cont(k,[.1,.5,.9,.99]),max(k) from a")
print('== random per-video uniformity'); q(f"with a as (select video_id,count(*) k from {R} group by 1) select count(*),avg(k),quantile_cont(k,[.01,.1,.5,.9,.99]),min(k),max(k) from a")
V=f"read_csv_auto('{d}/video_features_basic_pure.csv')"
print('== videos'); q(f"select video_type,count(*) from {V} group by 1"); q(f"select count(*),count(distinct split_part(tag,',',1)) t1,count(distinct tag) tags, avg(video_duration)/1000, avg((tag is null)::int) from {V}")
q(f"select split_part(tag,',',1) t,count(*) n from {V} group by 1 order by n desc limit 15")
print('== random log joined: reward by first tag (top 10)')
q(f"select split_part(v.tag,',',1) t,count(*) n,avg(is_click) clk,avg(long_view) lv from {R} r join {V} v using(video_id) group by 1 order by n desc limit 10")
print('== overlap users'); q(f"select (select count(distinct user_id) from {R}) r,(select count(distinct user_id) from {S1}) s1,(select count(distinct user_id) from {S2}) s2")
