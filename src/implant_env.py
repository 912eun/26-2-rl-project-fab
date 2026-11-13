"""implant 장비군 하나의 도착 기록 재생 이산사건 환경 (EDA 06).

공장 전체 시뮬레이터 기록(events_v2.csv)의 도착을 그대로 넣고, 장비군 안의 규칙만 시뮬레이터(PySCFabSim-revised)대로 옮겼다.
- 장비가 비면 지금 가스의 lot을 먼저 처리한다. 그 안에서는 hot lot, 먼저 온 lot 순이다.
- 교체(셋업)를 시작하면 최소 7 lot을 그 가스로 처리해야 한다. 그동안은 다른 가스로 못 바꾼다.
- PM은 처리 매수 기반이고 lot 투입 때 발생하며, 가스 상태를 지운다(instance.dispatch).
- 고장은 지수 간격이다. 가동 중이면 끝나는 시각을 미루고, 쉬는 중이면 가스 상태만 지운다(instance.handle_breakdown).
결정 지점: 비어 있는 장비에 지금 가스의 lot이 없고 다른 가스 lot이 있을 때. policy가 가스를 고르거나 기다림을 돌려준다.
"""
import copy
import csv
import gzip
import heapq
import random
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "Implant_128": dict(n=10, setup=72 * 60, foa=(8000, 34700, 104100), every=(8000, 34700, 104100)),
    "Implant_132": dict(n=8, setup=60 * 60, foa=(6806, 29453, 88358), every=(6875, 29750, 89250)),
    "Implant_91": dict(n=6, setup=80 * 60, foa=(6134, 26526, 79580), every=(6667, 28833, 86500)),
}
PM_DUR = ((13.29, 2.66), (26.59, 5.32), (53.18, 10.64))  # 시간, uniform(m ± l/2)
PIECES = 25
MIN_RUN = 7
MTTF, MTTR = 10080 * 60, 604.8 * 60
DONE, WAKE, BREAK, ARRIVE = 0, 1, 2, 3
WARMUP = 60 * 86400
MEAS_END = 220 * 86400  # 지표는 60~220일에 도착한 lot으로 잰다. 시뮬레이션은 기록 끝(약 240일)까지 돌려 측정 대상이 끝나게 한다


def record_path(seed, name):
    """공장 도착 기록 파일 위치. 레포에 올린 압축본(data/records)을 먼저 찾고, 없으면 공장을 직접 돌린 결과 폴더를 찾는다."""
    cands = [ROOT / f"data/records/HVLM_lssu_240d_s{seed}/{name}.gz",
             ROOT / f"data/records/HVLM_lssu_240d_s{seed}/{name}",
             ROOT / f"results/implant_setup/SMT2020_HVLM_fifo_lssu_Demand_240d_s{seed}/{name}"]
    for c in cands:
        if c.exists():
            return c
    return cands[-1]


def open_record(path):
    return gzip.open(path, "rt", newline="") if str(path).endswith(".gz") else open(path, newline="")


def load_arrivals(family, seed):
    """[(t, gas, hot, machine_time)] (시간순). gas는 0,1,2."""
    path = record_path(seed, "events_v2.csv")
    out, gases = [], {}
    with open_record(path) as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "arrive" and r["family"] == family:
                g = gases.setdefault(r["setup"], len(gases))
                out.append((float(r["t"]), g, int(r["priority"]) >= 20, float(r["machine_time"])))
    out.sort(key=lambda x: x[0])
    return out, gases


def make_schedule(n, seed, horizon=400 * 86400):
    """고장과 PM 길이를 미리 뽑는다. 규칙이 달라도 같은 고장을 겪게 하려는 것이다(공통 난수)."""
    sched = []
    for i in range(n):
        r = random.Random(seed * 1000 + i)
        br, t = [], r.expovariate(1 / MTTF)
        while t < horizon:
            ln = r.expovariate(1 / MTTR)
            br.append((t, ln))
            t += ln + r.expovariate(1 / MTTF)
        pm = [[r.uniform(c - l / 2, c + l / 2) * 3600 for _ in range(400)] for c, l in PM_DUR]
        sched.append((br, pm))
    return sched


def fab_schedule(family, rec_seed, fallback_seed=0):
    """공장 기록(downs.csv)의 고장·PM을 그대로 쓰는 일정. 형식은 make_schedule과 같다.

    - 고장: 공장 장비 번호를 정렬해 0..n-1로 맞추고, 그 장비의 (시각, 길이)를 그대로 쓴다.
    - 처리 매수 기반 PM: 공장 그 장비에서 뽑힌 길이를 순서대로 쓴다. 재생에서 PM이 더 많이 오면 fallback 난수로 채운다.
      (PM이 오는 시각은 재생에서 그 장비가 처리한 매수로 정해지므로 공장과 다를 수 있다.)
    공장 기록이 없으면 None.
    """
    path = record_path(rec_seed, "downs.csv")
    if not path.exists():
        return None
    rows = {}
    with open_record(path) as fh:
        for r in csv.DictReader(fh):
            if r["family"] == family:
                rows.setdefault(r["machine"], []).append((float(r["t"]), r["kind"], float(r["length"])))
    n = FAMILIES[family]["n"]
    ids = sorted(rows, key=int)
    base = make_schedule(n, fallback_seed)
    out = []
    for i in range(n):
        xs = sorted(rows.get(ids[i], [])) if i < len(ids) else []
        br = [(t, ln) for t, k, ln in xs if k == "br"]
        pm = []
        for j in range(3):
            got = [ln for t, k, ln in xs if k == f"ppm{j}"]
            pm.append(got + base[i][1][j][len(got):])
        out.append((br, pm))
    return out


def load_chains(family, seed, max_gap=float("inf")):
    """되먹임 판. 같은 장비군 재방문을 연결한다.

    반환: (뿌리 도착 목록, 평평한 도착 목록). 뿌리는 (t, gas, hot, machine_time, link)이고
    max_gap(초)보다 왕복이 긴 재방문은 연결하지 않고 기록된 시각에 도착시킨다.
    link는 None 또는 (왕복 시간, 다음 방문 노드)다. 왕복 시간 = 원 기록의 다음 도착 − (이번 시작 + 이번 셋업).
    다음 방문은 이번 처리의 시작 + 셋업 + 왕복 시간에 도착한다. 평평한 목록은 과거로 미래를 상상할 때 쓴다.
    """
    path = record_path(seed, "events_v2.csv")
    st = FAMILIES[family]["setup"]
    by_lot, gases = {}, {}
    prev = load_prev(seed)
    with open_record(path) as fh:
        for r in csv.DictReader(fh):
            if r["family"] == family:
                g = gases.setdefault(r["setup"], len(gases))
                by_lot.setdefault(r["lot"], []).append(
                    (float(r["t"]), r["kind"], g, int(r["priority"]) >= 20, float(r["machine_time"]), r["machine_setup_before"] != r["setup"],
                     prev.get((r["lot"], r["step"], round(float(r["t"]), 3))) if r["kind"] == "arrive" else None))
    roots, flat = [], []
    for ev in by_lot.values():
        ev.sort(key=lambda x: (x[0], 0 if x[1] == "arrive" else 1))
        visits = []  # [도착 기록, 시작 뒤 끝 시각]
        prev_end = None
        for x in ev:
            if x[1] == "arrive":
                visits.append([x, prev_end, None])
                flat.append((x[0], x[2], x[3], x[4]))
                prev_end = None
            else:
                prev_end = x[0] + (st if x[5] else 0)
        # 뒤에서부터 묶는다. prev_end가 있으면 앞 방문과 연결한다.
        node = None
        for x, pe, _ in reversed(visits):
            cur = (x[0], x[2], x[3], x[4], node, *(x[6] or (None, None)))
            if pe is None or x[0] - pe > max_gap:  # 왕복이 길면 고정 도착으로 둔다
                roots.append(cur)
                node = None
            else:
                node = (x[0] - pe, cur)
    roots.sort(key=lambda x: x[0])
    flat.sort(key=lambda x: x[0])
    return roots, flat


def load_prev(seed):
    """(lot, step, 도착 시각) -> (앞 1공정 진입부터 도착까지, 앞 2공정 진입부터 도착까지) 초. 없으면 빈 dict.

    정의는 공장 쪽 `implant_state` U1/U2와 같다: route상 바로 앞 1번째 공정에 있으면 U1, 앞 1·2번째 공정에 있으면 U2.
    앞 2번째 공정을 건너뛰었으면 U2는 앞 1번째 공정 진입부터 센다.
    """
    path = record_path(seed, "prev_times.csv")
    out = {}
    if not path.exists():
        return out
    with open_record(path) as fh:
        for r in csv.DictReader(fh):
            t = float(r["t"])
            e1 = float(r["enter_prev1"]) if r["enter_prev1"] else None
            e2 = float(r["enter_prev2"]) if r["enter_prev2"] else e1
            d1 = t - e1 if e1 is not None and t > e1 else None
            d2 = t - e2 if e2 is not None and t > e2 else None
            out[(r["lot"], r["step"], round(t, 3))] = (d1, d2)
    return out


class M:
    __slots__ = ("idx", "free", "gas", "left", "lock", "pieces", "hold_since", "wake_at", "forced_until", "pm_i", "br_i")

    def __init__(self, idx, foa, n):
        self.idx, self.free, self.gas, self.left, self.lock = idx, True, -1, None, -1
        self.pieces = [f / n * (idx + 1) for f in foa]
        self.hold_since, self.wake_at, self.forced_until = None, None, None
        self.pm_i, self.br_i = [0, 0, 0], 0

    def clone(self):
        c = M.__new__(M)
        c.idx, c.free, c.gas, c.left, c.lock = self.idx, self.free, self.gas, self.left, self.lock
        c.pieces = list(self.pieces)
        c.hold_since, c.wake_at, c.forced_until = self.hold_since, self.wake_at, self.forced_until
        c.pm_i, c.br_i = list(self.pm_i), self.br_i
        return c


class Sim:
    def __init__(self, family, arrivals, policy, seed=0, use_pm=True, use_br=True, warmup=WARMUP, n_gas=3, sched=None):
        """sched: None이면 seed로 고장·PM을 뽑는다(make_schedule). fab_schedule(...)을 주면 공장 기록의 고장을 재생한다."""
        p = FAMILIES[family]
        self.setup, self.every, self.arr, self.policy = p["setup"], p["every"], arrivals, policy
        self.use_pm, self.warmup, self.n_gas = use_pm, warmup, n_gas
        self.ms = [M(i, p["foa"], p["n"]) for i in range(p["n"])]
        self.q = [deque() for _ in range(n_gas)]   # 보통 lot: (도착 시각, machine_time)
        self.qh = [deque() for _ in range(n_gas)]  # hot lot
        self.ev, self.seq, self.i, self.t = [], 0, 0, 0.0
        self.sched = sched if sched is not None else make_schedule(p["n"], seed)
        self.busy_id = [0] * p["n"]
        self.nq = 0
        self.area = 0.0     # 누적 대기 면적 + 교체 지연 (rollout의 비용)
        self.setup_s = self.after_setup_s = 0.0
        self.n_switch = 0
        self.rec = []       # (도착 시각, 대기, 셋업, hot)
        self.meas_end = MEAS_END
        self.hot_breaks_wait = False  # True면 forced_until(고정 시간 기다리기) 중이라도 hot lot이 있으면 다시 결정한다
        self.qarea_win = 0.0  # 측정 기간 안의 대기 lot 수 × 시간 (재공·리틀 대기용)
        self.n_arr_win = 0    # 측정 기간에 도착한 lot 수 (재방문 포함)
        self.log = True
        self.no_chain = False  # 상상 중에는 재방문을 평평한 과거 도착으로 대신한다
        # 앞 공정 lot 수 (RL 상태용). up_heap: (진입 시각, k, 가스), up_cnt[k][g]: 진입했고 아직 도착 안 한 lot 수
        self.up_heap = [(a[0] - a[5 + k], k, a[1]) for a in arrivals if len(a) > 6 for k in (0, 1) if a[5 + k] is not None]
        heapq.heapify(self.up_heap)
        self.up_cnt = [[0] * n_gas, [0] * n_gas]
        if use_br:
            for m in self.ms:
                if self.sched[m.idx][0]:
                    self._push(self.sched[m.idx][0][0][0], BREAK, m.idx)

    def clone(self, log=False):
        c = copy.copy(self)
        c.ms = [m.clone() for m in self.ms]
        c.q = [deque(x) for x in self.q]
        c.qh = [deque(x) for x in self.qh]
        c.ev = list(self.ev)
        c.busy_id = list(self.busy_id)
        c.up_heap = list(self.up_heap)
        c.up_cnt = [list(x) for x in self.up_cnt]
        c.log, c.rec = log, []
        return c

    def imagine(self, arrivals, sched):
        """복제본의 미래를 다른 도착과 다른 고장으로 바꾼다(미래를 모르는 계획용)."""
        self.arr, self.i = arrivals, 0
        self.sched = sched
        self.no_chain = True
        self.ev = [e for e in self.ev if e[2] not in (BREAK, ARRIVE)]
        heapq.heapify(self.ev)
        for m in self.ms:
            br = sched[m.idx][0]
            j = next((k for k, x in enumerate(br) if x[0] > self.t), len(br) - 1)
            m.br_i = j
            m.pm_i = [0, 0, 0]
            self._push(br[j][0], BREAK, m.idx)

    def _up_advance(self):
        while self.up_heap and self.up_heap[0][0] <= self.t:
            _, k, g = heapq.heappop(self.up_heap)
            self.up_cnt[k][g] += 1

    def _up_arrive(self, a):
        if len(a) > 6:
            self._up_advance()
            for k in (0, 1):
                if a[5 + k] is not None:
                    self.up_cnt[k][a[1]] -= 1

    def upstream(self):
        """[앞 1공정 lot 수, 앞 1·2공정 lot 수], 각각 가스별 목록."""
        self._up_advance()
        return [list(x) for x in self.up_cnt]

    def _push(self, t, kind, m, extra=0):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, kind, m, extra))

    def waiting(self, g):
        return len(self.q[g]) + len(self.qh[g])

    def oldest_gas(self):
        """lssu의 교체 대상: hot lot이 있으면 가장 오래된 hot, 없으면 가장 오래된 lot의 가스."""
        best, bt = -1, None
        for g in range(self.n_gas):
            if self.qh[g] and (bt is None or self.qh[g][0][0] < bt):
                best, bt = g, self.qh[g][0][0]
        if best >= 0:
            return best, True
        for g in range(self.n_gas):
            if self.q[g] and (bt is None or self.q[g][0][0] < bt):
                best, bt = g, self.q[g][0][0]
        return best, False

    def tools_on(self, g):
        return sum(1 for m in self.ms if m.gas == g)

    def _dispatch(self, m, g):
        lot = self.qh[g].popleft() if self.qh[g] else self.q[g].popleft()
        self.nq -= 1
        st = self.setup if m.gas != g else 0
        if st > 0:
            self.n_switch += 1
            if m.left is None:
                m.left, m.lock = MIN_RUN, g
            self.area += st
            if lot[0] >= self.warmup or not self.log:
                self.after_setup_s += st
        m.gas = g
        mt = lot[1]
        for i in range(3):
            m.pieces[i] -= PIECES
            if m.pieces[i] <= 0 and self.use_pm:
                mt += self.sched[m.idx][1][i][m.pm_i[i]]
                m.pm_i[i] += 1
                m.pieces[i] = self.every[i]
                m.gas = -1
        if m.left is not None:
            m.left -= 1
            if m.left <= 0:
                m.left, m.lock = None, -1
        m.free, m.hold_since, m.wake_at, m.forced_until = False, None, None, None
        self.busy_id[m.idx] += 1
        self._push(self.t + mt + st, DONE, m.idx, self.busy_id[m.idx])
        link = lot[3] if len(lot) > 3 else None
        if link is not None and not self.no_chain:
            ta = self.t + st + link[0]
            self._push(ta, ARRIVE, -1, link[1])
            nx = link[1]
            if len(nx) > 6:
                for k in (0, 1):
                    if nx[5 + k] is not None:
                        heapq.heappush(self.up_heap, (max(ta - nx[5 + k], self.t), k, nx[1]))
        if self.log and lot[0] >= self.warmup:
            self.rec.append((lot[0], self.t - lot[0], st, lot[2]))

    def decide(self):
        if self.nq == 0:
            return
        for m in self.ms:  # 1단계: 지금 가스의 lot을 처리한다
            if m.free and m.gas >= 0 and (m.left is None or m.lock == m.gas) and self.waiting(m.gas):
                self._dispatch(m, m.gas)
                if self.nq == 0:
                    return
        for m in self.ms:  # 2단계: 교체 결정
            if not m.free or self.nq == 0:
                continue
            if m.left is not None:  # 최소 연속 처리 중이면 그 가스만 된다
                if self.waiting(m.lock):
                    self._dispatch(m, m.lock)
                continue
            if m.forced_until is not None and self.t < m.forced_until and not (self.hot_breaks_wait and any(self.qh)):
                continue
            if m.hold_since is None:
                m.hold_since = self.t
            g, wake = self.policy(self, m)
            if g is not None and g >= 0:
                self._dispatch(m, g)
            elif wake is not None and wake != m.wake_at:
                m.wake_at = wake
                self._push(wake, WAKE, m.idx)

    def step(self):
        """다음 사건 시각까지 진행한다. 끝나면 False."""
        na = self.arr[self.i][0] if self.i < len(self.arr) else None
        ne = self.ev[0][0] if self.ev else None
        if na is None and (ne is None or self.nq == 0 and all(m.free for m in self.ms)):
            return False
        nt = ne if na is None or (ne is not None and ne <= na) else na
        self.area += self.nq * (nt - self.t)
        ov = min(nt, self.meas_end) - max(self.t, self.warmup)
        if ov > 0:
            self.qarea_win += self.nq * ov
        self.t = nt
        while self.i < len(self.arr) and self.arr[self.i][0] <= self.t:
            a = self.arr[self.i]
            self._up_arrive(a)
            (self.qh if a[2] else self.q)[a[1]].append((a[0], a[3], a[2], a[4] if len(a) > 4 else None))
            if self.warmup <= a[0] < self.meas_end:
                self.n_arr_win += 1
            self.nq += 1
            self.i += 1
        while self.ev and self.ev[0][0] <= self.t:
            _, _, kind, mi, extra = heapq.heappop(self.ev)
            if kind == ARRIVE:  # 재방문 도착: 시각은 이번 시뮬레이션에서 계산한 값이다
                a = extra
                self._up_arrive(a)
                (self.qh if a[2] else self.q)[a[1]].append((self.t, a[3], a[2], a[4]))
                if self.warmup <= self.t < self.meas_end:
                    self.n_arr_win += 1
                self.nq += 1
                continue
            m = self.ms[mi]
            if kind == DONE:
                if extra == self.busy_id[mi] and not m.free:
                    m.free = True
            elif kind == BREAK:
                length = self.sched[mi][0][m.br_i][1]
                m.br_i += 1
                m.gas = -1
                if not m.free:  # 가동 중이면 끝나는 시각을 미룬다
                    new = []
                    for e in self.ev:
                        if e[2] == DONE and e[3] == mi:
                            self.seq += 1
                            new.append((e[0] + length, self.seq, DONE, mi, e[4]))
                        else:
                            new.append(e)
                    heapq.heapify(new)
                    self.ev = new
                if m.br_i < len(self.sched[mi][0]):
                    self._push(self.sched[mi][0][m.br_i][0], BREAK, mi)
        self.decide()
        return True

    def run(self, until=None):
        if until is None:  # 공장 실행과 같이 기록의 끝에서 멈춘다
            until = self.arr[-1][0] if self.arr else self.t
        while self.step():
            if until is not None and self.t >= until:
                break
        return self


# ---- 규칙. policy(sim, m) -> (가스 또는 None, 깨울 시각 또는 None)
def lssu(s, m):
    return s.oldest_gas()[0], None


def hold(T):
    def f(s, m):
        g, hot = s.oldest_gas()
        if hot or m.gas < 0:
            return g, None
        end = m.hold_since + T
        return (g, None) if s.t >= end else (None, end)
    return f


def thr(Q, Tmax, cov=False):
    def f(s, m):
        g, hot = s.oldest_gas()
        if hot or m.gas < 0:
            return g, None
        # 대상: 대기열(cov면 장비당 대기열)이 가장 긴 가스
        def score(x):
            w = s.waiting(x)
            return w / (s.tools_on(x) + 1) if cov else w
        best = max((x for x in range(s.n_gas) if s.waiting(x)), key=score)
        if score(best) >= Q:
            return best, None
        end = m.hold_since + Tmax
        return (g, None) if s.t >= end else (None, end)
    return f


def rollout(base, H=8 * 3600, hold_step=900, future=None, k=1, start=WARMUP - 86400):
    """행동마다 복제해 base 규칙으로 H초 앞까지 돌려 지연이 가장 적은 행동을 고른다.

    future=None이면 실제 도착과 고장을 쓴다(미래를 아는 값).
    future(sim, i) -> (도착 목록, 고장 일정)이면 그것으로 상상한다(미래를 모르는 값). 이때 시나리오 k개의 평균을 쓴다.
    """
    def f(s, m):
        if s.t < start:
            return base(s, m)
        g0, hot = s.oldest_gas()
        if hot or m.gas < 0:
            return g0, None
        acts = [g for g in range(s.n_gas) if s.waiting(g)] + [None]
        scen = [future(s, i) for i in range(k)] if future is not None else [None]
        best, bc = None, None
        for a in acts:
            tot = 0.0
            for sc in scen:
                c = s.clone()
                c.policy = base
                if sc is not None:
                    c.imagine(*sc)
                cm = c.ms[m.idx]
                a0 = c.area
                if a is None:
                    cm.forced_until = s.t + hold_step
                    c._push(s.t + hold_step, WAKE, m.idx)
                else:
                    c._dispatch(cm, a)
                c.run(until=s.t + H)
                tot += c.area - a0
            if bc is None or tot < bc:
                best, bc = a, tot
        if best is None:
            m.forced_until = s.t + hold_step
            return None, s.t + hold_step
        return best, None
    return f


def past_future(arrivals, sched_pool, H=8 * 3600, seed=0, L=0):
    """과거 도착으로 미래를 상상한다. 무작위 과거 시점의 도착을 지금으로 옮겨 쓴다.

    L>0이면 앞으로 L초 동안의 도착은 실제 값을 쓰고 그 뒤만 상상한다. 앞 공정에 있는 lot을 보고 곧 올 도착을 아는 상황을 흉내 낸다.
    """
    import bisect
    times = [a[0] for a in arrivals]
    rng = random.Random(seed)

    def f(s, i):
        lo = max(0.0, s.t - 30 * 86400)
        hi = s.t - H - 1
        o = rng.uniform(lo, hi)  # 과거 구간 시작점
        a, b = bisect.bisect_left(times, o + L), bisect.bisect_left(times, o + H)
        d = s.t - o
        fut = [(arrivals[j][0] + d, arrivals[j][1], arrivals[j][2], arrivals[j][3]) for j in range(a, b)]
        if L > 0:
            i0, i1 = bisect.bisect_right(times, s.t), bisect.bisect_right(times, s.t + L)
            fut = list(arrivals[i0:i1]) + fut
        return fut, sched_pool[rng.randrange(len(sched_pool))]
    return f


def summarize(sim, days=240):
    """지표 묶음 (2026-11-13 평가 수정, docs/EXPERIMENT_PLAN.md).

    측정 대상 = 60~220일(sim.warmup ~ sim.meas_end)에 도착한 lot. 시뮬레이션은 그 뒤로도 돌아서 대상 lot이 거의 다 끝난다.
    - delay_h  : 주 지표 D. lot당 (대기 + 그 lot이 일으킨 교체 시간), 처리를 시작한 측정 대상 lot 평균
    - unstarted: 측정 대상인데 끝까지 처리를 시작 못 한 lot 수 (0이어야 D에 빠진 lot이 없음)
    - wip      : 측정 기간 시간 평균 대기 lot 수 (재공)
    - little_wait_h: 리틀 법칙 대기 = wip ÷ 도착률. wait_h와 거의 같아야 굶기기가 없다는 뜻
    - weekly   : 도착 주별 평균 지연(초), 짝 비교·블록 부트스트랩용
    """
    rec = [r for r in sim.rec if r[0] < sim.meas_end]
    n = len(rec)
    if n == 0:
        return {}
    wait = sum(r[1] for r in rec) / n / 3600
    delay = sum(r[1] + r[2] for r in rec) / n / 3600
    d = sorted(r[1] + r[2] for r in rec)
    hot = [r[1] + r[2] for r in rec if r[3]]
    span = max(sim.t - sim.warmup, 1)
    win = max(min(sim.t, sim.meas_end) - sim.warmup, 1)
    lam = sim.n_arr_win / win
    wip = sim.qarea_win / win
    wk = {}
    for r in rec:
        k = int((r[0] - sim.warmup) // (7 * 86400))
        a = wk.setdefault(k, [0.0, 0])
        a[0] += r[1] + r[2]
        a[1] += 1
    return dict(n=n, wait_h=wait, delay_h=delay, p95_h=d[int(.95 * n)] / 3600,
                hot_delay_h=(sum(hot) / len(hot) / 3600 if hot else float("nan")),
                setup_ratio=sim.after_setup_s / (len(sim.ms) * span),
                switch_per_day=sum(1 for r in sim.rec if r[2] > 0) / (span / 86400),
                unstarted=sim.n_arr_win - n, wip=wip, little_wait_h=(wip / lam / 3600 if lam > 0 else float("nan")),
                throughput_per_day=n / (win / 86400),
                weekly={k: v[0] / v[1] for k, v in sorted(wk.items())})


def weekly(sim):
    """주 단위 평균 지연(초). 짝 비교용."""
    w = {}
    for r in sim.rec:
        k = int((r[0] - sim.warmup) // (7 * 86400))
        a = w.setdefault(k, [0.0, 0])
        a[0] += r[1] + r[2]
        a[1] += 1
    return {k: v[0] / v[1] for k, v in w.items() if v[1] > 0}
