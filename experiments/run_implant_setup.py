"""implant 가스 셋업 처리 방식만 바꿔서 PySCFabSim-revised greedy 시뮬레이션을 돌린다 (EDA 04).

가스 셋업이 있는 장비군 Implant_128 / Implant_132 / Implant_91 에만 적용한다. 나머지 장비군은 기본 그대로다.

모드:
  lssu       기본값. lot 우선순위에 '셋업 시간'이 들어가 가스가 맞는 lot을 먼저 고른다(dispatcher.py),
             가스가 맞는 다른 빈 장비가 있으면 그쪽으로 넘긴다(greedy.find_alternative_machine).
  nosetup    위 두 가지를 끈다. 즉 가스를 보지 않고 FIFO로 고른다. 최소 연속 처리(7 lot)는 SMT2020 필수 규칙이라 유지.
  dedicated  장비마다 가스 하나를 고정한다. lot은 자기 가스 전용 장비의 대기 목록에만 들어간다.
             대수 배정: EDA 02의 가스별 필요 대수(results/eda02_setup_tools_needed_<dataset>.csv)를 올림한 뒤,
             남는 장비를 (필요량 / 배정 대수)가 가장 큰 가스에 하나씩 더 준다.
             PM 뒤에는 시뮬레이터가 current_setup을 ''로 지우므로 전용이어도 셋업이 0은 아니다.
  longest    lssu와 같되, 가스를 바꿔야 할 때 가장 오래 기다린 lot 대신 대기열이 가장 긴 가스를 고른다.
             hot lot 우선은 유지한다(우선순위가 대기열 길이보다 앞선다).
  hold       lssu와 같되, 교체가 필요해지면 --hold_min 분 동안 교체하지 않고 기다린다.
             그사이 지금 가스의 lot이 오면 교체 없이 처리한다. 예외: hot lot, PM 뒤(어차피 셋업 필요).
  cov        EDA 06의 규칙. 가스별 (대기 lot 수 / 그 가스를 맡은 장비 수 + 1)을 계산해, 가장 큰 가스의 값이 Q 이상이면
             그 가스로 교체한다. 아니면 Tmax까지 기다린 뒤 가장 오래 기다린 lot의 가스로 교체한다. 예외는 hold와 같다.
             파라미터는 시드 1 도착 기록 재생에서 튜닝한 값(COV)이다. --cov_fb면 재방문 연결을 고친 재생 환경의
             튜닝값(COV_FB, results/eda06_fb/tune.json, RL 비교 기준과 같은 값)을 쓴다. 출력 이름은 covfb.
  rl         (2026-11-13) 학습한 PPO/DQN 정책. 결정 시점·예외는 재생 환경(src/implant_gym.py)과 맞춘다:
             교체가 필요한 결정에서 hot lot이 없고 장비에 가스가 있으면 정책이 {기다리기, 다른 가스 1, 2} 중 고른다.
             기다리면 같은 장비군의 다음 lot 도착이나 다른 장비 작업 끝에 다시 묻는다. 그 밖은 cov(COV_FB)가 처리한다.


사용:
    python experiments/run_implant_setup.py --mode dedicated --dataset SMT2020_HVLM --dispatcher fifo \
        --strategy Demand --days 730 --seed 1
출력: results/implant_setup/<dataset>_<dispatcher>_<mode>_<strategy>_<days>d_s<seed>/
시뮬레이터 코드는 수정하지 않는다(이 프로세스 안에서만 함수를 바꿔 끼운다).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "external" / "PySCFabSim-revised"
GAS_FAMS = ("Implant_128", "Implant_132", "Implant_91")
SETUP_S = {"Implant_128": 72 * 60, "Implant_132": 60 * 60, "Implant_91": 80 * 60}  # setup.txt STIME
COV = {"Implant_128": (5, 240), "Implant_132": (3, 240), "Implant_91": (5, 480)}  # (Q, Tmax 분), results/eda06/tune.json
COV_FB = {"Implant_128": (5, 120), "Implant_132": (3, 240), "Implant_91": (7, 240)}  # results/eda06_fb/tune.json = implant_gym.TUNED_COV


def allocate(dataset, fam_sizes):
    """가스별 전용 장비 대수. {family: {setup: n}}"""
    need = {}
    with open(ROOT / "results" / f"eda02_setup_tools_needed_{dataset}.csv") as f:
        for r in csv.DictReader(f):
            if r["STNFAM"] in GAS_FAMS:
                need.setdefault(r["STNFAM"], {})[r["SETUP"]] = float(r["tools_needed"])
    alloc = {}
    for fam, nd in need.items():
        a = {s: max(1, math.ceil(v)) for s, v in nd.items()}
        assert sum(a.values()) <= fam_sizes[fam], (fam, a, fam_sizes[fam])
        while sum(a.values()) < fam_sizes[fam]:
            s = max(a, key=lambda k: nd[k] / a[k])
            a[s] += 1
        alloc[fam] = a
    return alloc, need


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["lssu", "nosetup", "dedicated", "longest", "hold", "cov", "rl"])
    p.add_argument("--cov_fb", action="store_true", help="cov 파라미터로 고친 재생 환경의 튜닝값을 쓴다")
    p.add_argument("--policy", help="rl 모드: 정책 파일(.pt)")
    p.add_argument("--kind", choices=["ppo", "dqn"], help="rl 모드: 정책 종류")
    p.add_argument("--no_upstream", action="store_true", help="rl 모드: 앞 공정 lot 수를 0으로 (학습 때와 같게)")
    p.add_argument("--sample", action="store_true", help="rl 모드: argmax 대신 확률 정책에서 뽑는다(PPO, 작은 환경 시험과 같게)")
    p.add_argument("--hold_min", type=float, default=30, help="hold 모드의 교체 전 대기(분)")
    p.add_argument("--dataset", default="SMT2020_HVLM", choices=["SMT2020_HVLM", "SMT2020_LVHM"])
    p.add_argument("--dispatcher", default="fifo")
    p.add_argument("--strategy", default="Demand", choices=["Max", "Min", "RoundRobin", "Demand"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_events", action="store_true", help="implant 가스 장비군의 도착/투입 기록을 events.csv로 저장")
    p.add_argument("--log_prev", action="store_true",
                   help="implant 도착마다 route상 바로 앞 1, 2번째 공정에 들어온 시각을 prev_times.csv로 저장 (재생 환경의 앞 공정 lot 수용)")
    p.add_argument("--log_state", action="store_true",
                   help="cov 결정 시점의 RL 상태(src/implant_state.py)를 state_log.csv로 저장. 시뮬레이터 난수는 건드리지 않는다")
    a = p.parse_args()

    mode_name = f"hold{a.hold_min:g}" if a.mode == "hold" else a.mode
    if a.mode == "cov" and a.cov_fb:
        mode_name = "covfb"
    if a.mode == "rl":
        mode_name = f"rl_{Path(a.policy).stem}"
    cov_params = COV_FB if (a.cov_fb or a.mode == "rl") else COV
    out = ROOT / "results" / "implant_setup" / f"{a.dataset}_{a.dispatcher}_{mode_name}_{a.strategy}_{a.days}d_s{a.seed}"
    (out / "greedy").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SIM / "simulation"))
    import greedy as G  # noqa: E402
    from dispatching.dispatcher import dispatcher_map  # noqa: E402
    from dispatching.dm_lot_for_machine import LotForMachineDispatchManager as DM  # noqa: E402
    from events import ResetEvent  # noqa: E402
    from file_instance import FileInstance  # noqa: E402
    from plugins.cost_plugin import CostPlugin  # noqa: E402
    from randomizer import Randomizer  # noqa: E402
    from read import read_all  # noqa: E402
    from stats import print_statistics  # noqa: E402

    G.round_robin = False
    files = read_all(str(SIM / "datasets" / a.dataset))
    run_to = 3600 * 24 * a.days
    Randomizer().random.seed(a.seed)
    instance = FileInstance(files, run_to, True, [CostPlugin()], None, a.strategy)
    if a.days > 365:
        instance.add_event(ResetEvent(31536000))  # 원본과 같이 1년 뒤 통계 리셋
    base_disp = dispatcher_map[a.dispatcher]
    dispatcher = base_disp
    info = {"mode": mode_name}

    if a.mode == "nosetup":
        def dispatcher(lot, t, machine=None, setups=None):
            pt = base_disp(lot, t, machine, setups)
            if machine is not None and machine.family in GAS_FAMS:
                pt = (pt[0], 0) + tuple(pt[2:])  # 셋업 시간 항목만 0으로
                lot.ptuple = pt
            return pt

        orig_alt = G.find_alternative_machine

        def no_alt(instance_, lots, machine):
            if machine.family in GAS_FAMS:
                return machine
            return orig_alt(instance_, lots, machine)
        G.find_alternative_machine = no_alt

    elif a.mode == "dedicated":
        fam_machines = {f: list(instance.family_machines[f]) for f in GAS_FAMS}
        alloc, need = allocate(a.dataset, {f: len(m) for f, m in fam_machines.items()})
        gas_of = {}  # machine.idx -> setup
        for fam, ms in fam_machines.items():
            gases = [s for s in sorted(alloc[fam]) for _ in range(alloc[fam][s])]
            for m, s in zip(ms, gases):
                gas_of[m.idx] = s
                m.current_setup = s  # 시작부터 자기 가스로 맞춰 둔다
        info["allocation"] = alloc
        info["tools_needed"] = need
        # 시작 시점에 이미 대기 목록에 들어간 lot(초기 WIP)을 전용 규칙에 맞게 정리
        for fam, ms in fam_machines.items():
            for m in ms:
                for lot in list(m.waiting_lots):
                    sn = lot.actual_step.setup_needed
                    if sn != "" and sn != gas_of[m.idx]:
                        m.waiting_lots.remove(lot)
                        lot.waiting_machines.remove(m)
                if not m.waiting_lots and m in instance.usable_machines:
                    instance.usable_machines.remove(m)

        def free_up_lots(self, lot):
            fam = lot.actual_step.family
            sn = lot.actual_step.setup_needed
            for machine in self.family_machines[fam]:
                if fam in GAS_FAMS and sn != "" and gas_of[machine.idx] != sn:
                    continue
                di = lot.actual_step.order
                if di not in lot.dedications or machine.idx == lot.dedications[di]:
                    machine.waiting_lots.append(lot)
                    lot.waiting_machines.append(machine)
                    if self.free_machines[machine.idx]:
                        self.usable_machines.add(machine)
        DM.free_up_lots = staticmethod(free_up_lots)

    elif a.mode == "longest":
        cache = {}

        def dispatcher(lot, t, machine=None, setups=None):
            pt = base_disp(lot, t, machine, setups)
            if machine is None or machine.family not in GAS_FAMS:
                return pt
            key = (machine.idx, t, len(machine.waiting_lots))
            if cache.get("key") != key:
                q = {}
                for l in machine.waiting_lots:
                    sn = l.actual_step.setup_needed
                    q[sn] = q.get(sn, 0) + 1
                cache["key"], cache["q"] = key, q
            # base: (최소 연속 처리, 셋업 시간, -우선순위, 도착 시각, 납기). 교체가 필요한 lot만 대기열 길이를 앞세운다.
            qlen = cache["q"].get(lot.actual_step.setup_needed, 0) if pt[1] > 0 else 0
            lot.ptuple = (pt[0], pt[1], pt[2], -qlen) + tuple(pt[3:])
            return lot.ptuple

    state_rows, arr_rows = [], []
    if a.log_state or a.mode == "rl":
        assert a.mode in ("cov", "rl"), "--log_state는 cov 모드에서만 쓴다"
        import random as _random
        from collections import deque as _deque
        sys.path.insert(0, str(ROOT / "src"))
        import implant_state as S  # noqa: E402
        st_rng = _random.Random(12345)  # 표본 추출 전용. 시뮬레이터 난수와 분리
        st_gases = {f: sorted({s.setup_needed for r in instance.routes.values() for s in r.steps if s.family == f})
                    for f in GAS_FAMS}
        st_recent = {f: {g: _deque() for g in st_gases[f]} for f in GAS_FAMS}
        inner_free_s = DM.free_up_lots

        def state_free(self, lot):
            f = lot.actual_step.family
            if f in GAS_FAMS:
                st_recent[f][lot.actual_step.setup_needed].append(self.current_time)
                arr_rows.append((self.current_time, f, lot.actual_step.setup_needed))
            inner_free_s(self, lot)
        DM.free_up_lots = staticmethod(state_free)

        def build_view(machine, waited):
            """재생 환경 observe()와 같은 상태. (View, 상대 가스 순서, 가스별 [앞 1공정, 앞 1·2공정] lot 수)."""
            f, now = machine.family, instance.current_time
            gases = st_gases[f]
            q = {g: [0, 0, None] for g in gases}
            for l in machine.waiting_lots:
                x = q[l.actual_step.setup_needed]
                x[1 if l.priority >= hot_priority else 0] += 1
                x[2] = l.free_since if x[2] is None else min(x[2], l.free_since)
            others = []
            for m in instance.family_machines[f]:
                if m is machine:
                    continue
                busy = not instance.free_machines[m.idx]
                end = None
                if busy:
                    ends = [e.timestamp for e in m.events if type(e).__name__ == "MachineDoneEvent"]
                    end = max(ends) if ends else None
                others.append((m.current_setup or None, busy, m.min_runs_left is not None, end))
            for g in gases:
                dq = st_recent[f][g]
                while dq and dq[0] < now - 3600:
                    dq.popleft()
            v = S.View(n=len(instance.family_machines[f]), setup_s=SETUP_S[f], now=now, gases=gases,
                       gas0=machine.current_setup, waited_s=waited,
                       pm_pieces=min(machine.pieces_until_maintenance) if machine.pieces_until_maintenance else 0,
                       others=others, queue={g: tuple(q[g]) for g in gases},
                       arr1h={g: len(st_recent[f][g]) for g in gases})
            order = S.order(v)
            up = {g: [0, 0] for g in gases}  # 앞 1공정, 앞 2공정 안에서 이 장비군 이 가스로 올 lot
            for l in instance.active_lots:
                for k, s in enumerate(l.remaining_steps[:2]):
                    if s.family == f:
                        if s.setup_needed in up:
                            if k == 0:
                                up[s.setup_needed][0] += 1
                            up[s.setup_needed][1] += 1
                        break
            return v, order, up

        def record_state(machine, waited):
            v, order, up = build_view(machine, waited)
            f, now = machine.family, instance.current_time
            state_rows.append([now, f, machine.idx, machine.current_setup, *order[1:], int(machine.min_runs_left is not None),
                               *S.encode(v), *S.extras(v),
                               *[up[g][0] / v.n for g in order], *[up[g][1] / v.n for g in order]])

    hold_state = None
    if a.mode in ("hold", "cov", "rl"):
        hold_state = {"until": {}, "since": {}, "holds": 0, "kept_gas": 0, "expired_switch": 0, "hot_skipped": 0,
                      "rl_calls": 0, "rl_wait": 0, "rl_switch": 0}
        hot_priority = max(lot.priority for lot in instance.active_lots + list(instance.dispatchable_lots))
        info["hold_min"] = a.hold_min if a.mode == "hold" else None
        info["hold"] = hold_state

        class WakeEvent:
            """대기가 끝나는 순간 장비를 다시 결정 대상에 넣는다."""

            def __init__(self, timestamp, machine):
                self.timestamp, self.machine = timestamp, machine
                self.machines, self.lots = [], []

            def handle(self, inst):
                m = self.machine
                if inst.free_machines[m.idx] and m.waiting_lots:
                    inst.usable_machines.add(m)

        inner_d = instance.dispatch

        def hold_dispatch(machine, lots):
            hold_state["since"].pop(machine.idx, None)
            if machine.idx in hold_state["until"]:
                del hold_state["until"][machine.idx]
                if lots[0].actual_step.setup_needed == machine.current_setup:
                    hold_state["kept_gas"] += 1  # 기다린 덕분에 교체를 피했다
                else:
                    hold_state["expired_switch"] += 1
            return inner_d(machine, lots)
        instance.dispatch = hold_dispatch

    def should_hold(machine, lots):
        """이 결정을 미룰지 판단하고, 처음 미룰 때 깨우는 사건을 건다."""
        if hold_state is None or lots is None or machine.family not in GAS_FAMS:
            return False
        sn = lots[0].actual_step.setup_needed
        if sn == "" or sn == machine.current_setup or machine.current_setup == "":
            return False
        if lots[0].priority >= hot_priority:
            hold_state["hot_skipped"] += 1
            return False
        now = instance.current_time
        until = hold_state["until"].get(machine.idx)
        if until is None:
            until = now + a.hold_min * 60
            hold_state["until"][machine.idx] = until
            hold_state["holds"] += 1
            instance.add_event(WakeEvent(until, machine))
        return now < until

    def cov_decide(machine, lots):
        """cov 규칙. (장비, lot 목록)을 돌려준다. lot 목록이 None이면 기다린다."""
        sn = lots[0].actual_step.setup_needed
        cur = machine.current_setup
        if (a.log_state and machine.family in GAS_FAMS and sn != "" and sn != cur and cur != ""
                and machine.idx not in hold_state["until"] and instance.current_time >= 60 * 86400 and st_rng.random() < 0.3):
            record_state(machine, 0.0)
        if machine.family not in GAS_FAMS or sn == "" or sn == cur or cur == "" or lots[0].priority >= hot_priority:
            return machine, lots
        tools, waiting = {}, {}
        for m in instance.family_machines[machine.family]:
            tools[m.current_setup] = tools.get(m.current_setup, 0) + 1
        for l in machine.waiting_lots:
            g = l.actual_step.setup_needed
            waiting[g] = waiting.get(g, 0) + 1
        score = lambda g: waiting[g] / (tools.get(g, 0) + 1)  # noqa: E731
        best = max(waiting, key=score)
        q, tmax = cov_params[machine.family]
        if score(best) >= q:
            lot = min((l for l in machine.waiting_lots if l.actual_step.setup_needed == best),
                      key=lambda l: (-l.priority, l.free_since))
            return G.find_alternative_machine(instance, [lot], machine), [lot]
        now = instance.current_time
        until = hold_state["until"].get(machine.idx)
        if until is None:
            until = now + tmax * 60
            hold_state["until"][machine.idx] = until
            hold_state["holds"] += 1
            instance.add_event(WakeEvent(until, machine))
        return machine, (None if now < until else lots)

    if a.mode == "rl":
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT / "experiments"))
        import rl_common as RC  # noqa: E402
        torch.set_num_threads(1)
        net = RC.NETS[a.kind]()
        net.load_state_dict(torch.load(a.policy))
        net.eval()
        act_fn = RC.sampler(net, 1000 + a.seed) if a.sample else RC.greedy(net)
        info["policy"] = {"path": a.policy, "kind": a.kind, "use_upstream": not a.no_upstream, "sample": a.sample}

    def rl_decide(machine, lots):
        """rl 모드. 재생 환경의 rl_turn과 같은 조건에서만 정책을 부르고, 나머지는 cov가 처리한다."""
        sn = lots[0].actual_step.setup_needed
        cur = machine.current_setup
        if machine.family not in GAS_FAMS or sn == "" or sn == cur:
            return machine, lots
        hot = any(l.priority >= hot_priority for l in machine.waiting_lots)
        if cur == "" or hot:
            return cov_decide(machine, lots)
        now = instance.current_time
        since = hold_state["since"].setdefault(machine.idx, now)
        v, order, up = build_view(machine, now - since)
        if a.no_upstream:
            u = [0.0] * 6
        else:
            u = [float(np.log1p(up[g][k] / v.n)) for k in (0, 1) for g in order]
        obs = np.array(S.transform(S.encode(v)) + u, dtype=np.float32)
        mask = np.array([True] + [v.queue[g][0] + v.queue[g][1] > 0 for g in order[1:]])
        act = act_fn(obs, mask)
        hold_state["rl_calls"] += 1
        if act == 0:
            hold_state["rl_wait"] += 1
            # 재생 환경은 같은 장비군의 어떤 사건(도착, 다른 장비 작업 끝)에도 다시 묻는다.
            # 도착은 시뮬레이터가 장비를 다시 깨우고, 작업 끝은 가장 이른 끝 시각에 깨우는 사건을 건다.
            ends = [e.timestamp for m in instance.family_machines[machine.family] if m is not machine
                    for e in m.events if type(e).__name__ == "MachineDoneEvent" and e.timestamp > now]
            if ends:
                instance.add_event(WakeEvent(min(ends), machine))
            return machine, None
        hold_state["rl_switch"] += 1
        g = order[act]
        lot = min((l for l in machine.waiting_lots if l.actual_step.setup_needed == g),
                  key=lambda l: (-l.priority, l.free_since))
        return G.find_alternative_machine(instance, [lot], machine), [lot]

    events = []  # (t, kind, family, lot_idx, setup_needed, part, priority, machine_idx, machine_setup_before, n_lots)
    prev_rows = []
    if a.log_prev:
        from collections import deque as _dq
        lot_hist = {}
        inner_free_p = DM.free_up_lots

        def prev_free(self, lot):
            h = lot_hist.setdefault(lot.idx, _dq(maxlen=3))
            o = lot.actual_step.order
            if lot.actual_step.family in GAS_FAMS:
                e = {oo: tt for oo, tt in h}
                prev_rows.append((lot.idx, o, self.current_time, e.get(o - 1, ""), e.get(o - 2, "")))
            h.append((o, self.current_time))
            inner_free_p(self, lot)
        DM.free_up_lots = staticmethod(prev_free)

    if a.log_events:
        inner_free = DM.free_up_lots

        def logging_free(self, lot):
            f = lot.actual_step.family
            if f in GAS_FAMS:
                events.append((self.current_time, "arrive", f, lot.idx, lot.actual_step.setup_needed, lot.part_name,
                               lot.priority, -1, "", 1, lot.actual_step.order, lot.actual_step.cascading_time.avg()))
            inner_free(self, lot)
        DM.free_up_lots = staticmethod(logging_free)
        inner_dispatch = instance.dispatch

        def logging_dispatch(machine, lots):
            if machine.family in GAS_FAMS:
                for l in lots:
                    events.append((instance.current_time, "start", machine.family, l.idx, l.actual_step.setup_needed,
                                   l.part_name, l.priority, machine.idx, machine.current_setup, len(lots),
                                   l.actual_step.order, l.actual_step.cascading_time.avg()))
            return inner_dispatch(machine, lots)
        instance.dispatch = logging_dispatch

    t0 = time.time()
    next_report = 0
    while not instance.done:
        done = instance.next_decision_point()
        day = instance.current_time / 86400
        if day >= next_report:
            el = time.time() - t0
            eta = el / day * (a.days - day) if day > 0 else float("nan")
            print(f"[{a.dataset} {a.mode}] day {day:6.1f}/{a.days}  elapsed {el/60:6.1f} min  "
                  f"ETA {eta/60:6.1f} min", file=sys.stderr, flush=True)
            next_report += 10
        if done or instance.current_time > run_to:
            break
        machine, lots = G.get_lots_to_dispatch_by_machine(instance, dispatcher)
        if a.mode == "cov" and lots is not None:
            machine, lots = cov_decide(machine, lots)
        elif a.mode == "rl" and lots is not None:
            machine, lots = rl_decide(machine, lots)
        if lots is None or (a.mode == "hold" and should_hold(machine, lots)):
            instance.usable_machines.remove(machine)
        else:
            instance.dispatch(machine, lots)
    instance.finalize()
    wall = time.time() - t0

    # implant 장비군별 대기: (dispatch된 lot 수, 누적 대기 초). 1년 차 리셋 이후 값인지는 리셋 구현에 따른다.
    info["lot_waiting_at_machine"] = {f: list(instance.lot_waiting_at_machine.get(f, (0, 0))) for f in GAS_FAMS}
    fam_stat = {}
    for f in GAS_FAMS:
        ms = instance.family_machines[f]
        fam_stat[f] = dict(n=len(ms), utilized_s=sum(m.utilized_time for m in ms),
                           setup_s=sum(m.setuped_time for m in ms), pm_s=sum(m.pmed_time for m in ms),
                           br_s=sum(m.bred_time for m in ms))
    info["family_time"] = fam_stat  # 1년 차 리셋 이후 누적(ResetEvent가 machine 시간을 0으로 돌림)
    if hold_state is not None:
        hold_state.pop("until")
        hold_state.pop("since")
    info["done_lots"] = len(instance.done_lots)
    info["active_lots"] = len(instance.active_lots)
    if a.log_prev:
        with open(out / "prev_times.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["lot", "step", "t", "enter_prev1", "enter_prev2"])
            w.writerows(prev_rows)
    if a.log_events:
        with open(out / "events_v2.csv", "w", newline="") as fh:  # step, machine_time 추가 (기존 events.csv는 보존)
            w = csv.writer(fh)
            w.writerow(["t", "kind", "family", "lot", "setup", "part", "priority", "machine", "machine_setup_before", "n",
                        "step", "machine_time"])
            w.writerows(events)
    if a.log_state:
        with open(out / "state_log.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "family", "machine", "gas0", "gas1", "gas2", "locked", *S.NAMES, *S.EXTRA_NAMES,
                        "U1_g0", "U1_g1", "U1_g2", "U2_g0", "U2_g1", "U2_g2"])
            w.writerows(state_rows)
        with open(out / "state_arrivals.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "family", "gas"])
            w.writerows(arr_rows)
    (out / "implant_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))

    os.chdir(out)
    if a.days < 365:  # 짧은 점검 실행은 제품별 완성 lot이 없어 print_statistics가 실패한다
        print(f"done {a.mode} (short, no stats)"); return
    print_statistics(instance, a.days, a.dataset, a.dispatcher, method=f"greedy_seed{a.seed}_{a.mode}_{a.strategy}",
                     dir="greedy", wip=True, seed=a.seed)
    (out / "run_info.txt").write_text(
        f"dataset={a.dataset}\ndispatcher={a.dispatcher}\nmode={a.mode}\nstrategy={a.strategy}\ndays={a.days}\n"
        f"seed={a.seed}\nwall_seconds={wall:.0f}\nsimulator=PySCFabSim-revised@ae3d55e\n")
    print(f"done {a.mode} in {wall/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
