"""07_diffusion_bernoulli_vi.py 모델에서 도착률을 0.6~1.15배로 바꿔 Min과 VI 정답의 차이를 본다 (2026-11-13)."""
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("m", "notebooks/07_diffusion_bernoulli_vi.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for mul in [0.6, 1.0, 1.1, 1.15]:
    fams = [dict(name="109", p=0.209*mul, bmin=3, bmax=4, T=7), dict(name="001", p=0.209*mul, bmin=5, bmax=6, T=8)]
    V, pol, Q = m.solve(fams)
    opt = lambda qs, r: pol[tuple(qs) + (r,)]
    res = {pn: m.simulate(fams, p, Q) for pn, p in [("Min", m.rule("Min", fams)), ("Max", m.rule("Max", fams)), ("VI", opt)]}
    print(mul, {k: round(v[0], 3) for k, v in res.items()}, "gap Min-VI %.2f%%" % (100*(res['Min'][0]/res['VI'][0]-1)))
