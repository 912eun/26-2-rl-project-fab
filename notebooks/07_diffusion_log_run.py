"""확산로 도착·배치 시작 기록용 공장 실행 (2026-11-13, 확산로 설계 아티팩트용).
experiments/run_batch_strategies.py와 같은 루프에 기록 플러그인만 붙인다. 시뮬레이터 코드는 수정하지 않는다.
출력: <out>/diff_arrive.csv (t, lot, fam, step, part, prio)
      <out>/diff_start.csv  (t, machine, fam, step, part, n, machine_end, lot ids, 각 lot 대기 분, 시작 시점 같은 패밀리 대기 수, 장비군 전체 대기 수)
사용: python notebooks/07_diffusion_log_run.py --dataset SMT2020_LVHM --dispatcher cr --strategy Min --days 240 --seed 1 --out DIR
"""
import argparse, sys, time, csv
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "external" / "PySCFabSim-revised"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="SMT2020_LVHM"); p.add_argument("--dispatcher", default="cr")
    p.add_argument("--strategy", default="Min"); p.add_argument("--days", type=int, default=240)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SIM / "simulation"))
    import greedy as G
    from dispatching.dispatcher import dispatcher_map
    from file_instance import FileInstance
    from plugins.cost_plugin import CostPlugin
    from plugins.interface import IPlugin
    from randomizer import Randomizer
    from read import read_all

    fa = open(out / "diff_arrive.csv", "w", newline=""); wa = csv.writer(fa)
    wa.writerow(["t_min", "lot", "fam", "step", "part", "prio"])
    fs = open(out / "diff_start.csv", "w", newline=""); ws = csv.writer(fs)
    ws.writerow(["t_min", "machine", "fam", "step", "part", "n", "bmin", "bmax", "busy_min", "lots", "waits_min", "q_same", "q_all", "free_tools"])

    class Log(IPlugin):
        def on_step_done(self, inst, lot, step):
            s = lot.actual_step
            if s.family.startswith("Diffusion"):
                wa.writerow([round(inst.current_time / 60, 2), lot.idx, s.family, s.step_name, lot.part_name, lot.priority])
        def on_dispatch(self, inst, machine, lots, m_end, l_end):
            if not machine.family.startswith("Diffusion"):
                return
            s = lots[0].actual_step; key = (s.step_name, lots[0].part_name)
            wl = machine.waiting_lots
            q_same = sum(1 for l in wl if (l.actual_step.step_name, l.part_name) == key)
            free = sum(1 for m in inst.family_machines[machine.family] if m in inst.usable_machines)
            t = inst.current_time
            ws.writerow([round(t / 60, 2), machine.idx, machine.family, s.step_name, lots[0].part_name, len(lots), s.batch_min, s.batch_max,
                         round((m_end - t) / 60, 1), ";".join(str(l.idx) for l in lots),
                         ";".join(str(round((t - l.free_since) / 60, 1)) for l in lots), q_same, len(wl), free])

    G.round_robin = False
    files = read_all(str(SIM / "datasets" / a.dataset))
    run_to = 3600 * 24 * a.days
    Randomizer().random.seed(a.seed)
    inst = FileInstance(files, run_to, True, [CostPlugin(), Log()], None, a.strategy)
    disp = dispatcher_map[a.dispatcher]
    t0 = time.time(); nxt = 0
    while not inst.done:
        done = inst.next_decision_point()
        day = inst.current_time / 86400
        if day >= nxt:
            print(f"[{a.dataset}] day {day:6.1f}/{a.days} elapsed {(time.time()-t0)/60:5.1f} min", file=sys.stderr, flush=True); nxt += 10
        if done or inst.current_time > run_to:
            break
        m, lots = G.get_lots_to_dispatch_by_machine(inst, disp)
        if lots is None:
            inst.usable_machines.remove(m)
        else:
            inst.dispatch(m, lots)
    fa.close(); fs.close()
    print("done", time.time() - t0, file=sys.stderr)

if __name__ == "__main__":
    main()
