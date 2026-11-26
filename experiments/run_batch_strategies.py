"""배치 전략만 바꿔서 PySCFabSim-revised greedy 시뮬레이션을 돌린다.

다른 조건(데이터셋, 디스패처, 기간, 시드, 초기 WIP, 1년 후 리셋)은 고정하고
batch_strat ∈ {Max, Min, RoundRobin, Demand}만 바꾼다.

사용:
    python experiments/run_batch_strategies.py --dataset SMT2020_HVLM --dispatcher fifo \
        --strategy Demand --days 730 --seed 1
출력: results/batch_strat/<dataset>_<dispatcher>_<strategy>_<days>d_s<seed>/ 아래 JSON과 로그

원본 run_greedy()(simulation/greedy.py)는 인자를 코드 안에서 덮어쓰고 wandb를 켜므로,
같은 루프를 여기서 인자만 바꿔 다시 쓴다. 시뮬레이터 코드는 수정하지 않는다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "external" / "PySCFabSim-revised"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="SMT2020_HVLM", choices=["SMT2020_HVLM", "SMT2020_LVHM"])
    p.add_argument("--dispatcher", default="fifo")
    p.add_argument("--strategy", required=True, choices=["Max", "Min", "RoundRobin", "Demand"])
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    out = ROOT / "results" / "batch_strat" / f"{a.dataset}_{a.dispatcher}_{a.strategy}_{a.days}d_s{a.seed}"
    (out / "greedy").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SIM / "simulation"))
    import greedy as G  # noqa: E402
    from dispatching.dispatcher import dispatcher_map  # noqa: E402
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
    dispatcher = dispatcher_map[a.dispatcher]

    t0 = time.time()
    next_report = 0
    while not instance.done:
        done = instance.next_decision_point()
        day = instance.current_time / 86400
        if day >= next_report:  # 10 시뮬레이션일마다 진행 표시
            el = time.time() - t0
            eta = el / day * (a.days - day) if day > 0 else float("nan")
            print(f"[{a.dataset} {a.strategy}] day {day:6.1f}/{a.days}  elapsed {el/60:6.1f} min  "
                  f"ETA {eta/60:6.1f} min", file=sys.stderr, flush=True)
            next_report += 10
        if done or instance.current_time > run_to:
            break
        machine, lots = G.get_lots_to_dispatch_by_machine(instance, dispatcher)
        if lots is None:
            instance.usable_machines.remove(machine)
        else:
            instance.dispatch(machine, lots)
    instance.finalize()
    wall = time.time() - t0

    os.chdir(out)
    print_statistics(instance, a.days, a.dataset, a.dispatcher, method=f"greedy_seed{a.seed}_{a.strategy}",
                     dir="greedy", wip=True, seed=a.seed)
    (out / "run_info.txt").write_text(
        f"dataset={a.dataset}\ndispatcher={a.dispatcher}\nstrategy={a.strategy}\ndays={a.days}\n"
        f"seed={a.seed}\nwall_seconds={wall:.0f}\nsimulator=PySCFabSim-revised@ae3d55e\n")
    print(f"done {a.strategy} in {wall/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
