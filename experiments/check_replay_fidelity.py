import json,sys
from concurrent.futures import ProcessPoolExecutor as Pool
sys.path.insert(0,'src'); import implant_env as E
FAC_COV={'Implant_128':(5,240),'Implant_132':(3,240),'Implant_91':(5,480)}
RULES=['lssu','hold15','hold30','cov']
def pol(r,f):
    if r=='lssu': return E.lssu
    if r.startswith('hold'): return E.hold(int(r[4:])*60)
    q,t=FAC_COV[f]; return E.thr(q,t*60,cov=True)
def job(x):
    f,s,r,ss=x
    a=E.load_chains(f,s,12*3600)[0]
    sm=E.summarize(E.Sim(f,a,pol(r,f),seed=ss).run())
    return x,sm['wait_h'],sm['setup_ratio']
jobs=[(f,s,r,ss) for f in E.FAMILIES for s in (1,2,3) for r in RULES for ss in (1,2)]
if __name__=='__main__':
    with Pool(8) as p: res=list(p.map(job,jobs))
    rep={}
    for (f,s,r,ss),w,sr in res:
        d=rep.setdefault((f,r,s),[0,0]); d[0]+=w/2; d[1]+=sr/2
    fac={}
    for r in RULES:
      for s in (1,2,3):
        d=json.load(open(f'results/implant_setup/SMT2020_HVLM_fifo_{r}_Demand_730d_s{s}/implant_info.json'))
        for f,(n,w) in d['lot_waiting_at_machine'].items():
            ft=d['family_time'][f]
            fac[(f,r,s)]=[w/n/3600, ft['setup_s']/(ft['n']*730*86400)]
    out=[]
    for f in E.FAMILIES:
      for r in RULES:
        R=[rep[(f,r,s)] for s in (1,2,3)]; F=[fac[(f,r,s)] for s in (1,2,3)]
        Rl=[rep[(f,'lssu',s)] for s in (1,2,3)]; Fl=[fac[(f,'lssu',s)] for s in (1,2,3)]
        m=lambda L,i:sum(x[i] for x in L)/3
        out.append((f,r,m(F,0),m(R,0),m(F,0)/m(Fl,0)-1,m(R,0)/m(Rl,0)-1,m(F,1),m(R,1)))
        print(f"{f:12s} {r:7s} 대기 공장 {m(F,0):.2f}h 재생 {m(R,0):.2f}h | lssu대비 공장 {100*(m(F,0)/m(Fl,0)-1):+5.0f}% 재생 {100*(m(R,0)/m(Rl,0)-1):+5.0f}% | 교체비율 공장 {100*m(F,1):4.1f}% 재생 {100*m(R,1):4.1f}%")
    json.dump(out,open('/private/tmp/claude-501/-Users-sye----/eb1ebf82-6856-4af6-868e-56c3cd022359/scratchpad/fid.json','w'))
