import time
import uuid
import multiprocessing as mp
from itertools import product

from perseus.client import init, get_power_state_schedule, report_profiling_result
from perseus.models import PipeInstruction


dp = 2
pp = 1
tp = 3
job_id = uuid.uuid4().hex

def worker(args) -> None:
    dp_rank, pp_rank, tp_rank = args
    rank = dp_rank * pp * tp + pp_rank * tp + tp_rank
    world_size = dp * pp * tp

    init(
        job_id=job_id,
        rank=rank,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        world_size=world_size,
        pipe_schedule=[PipeInstruction.FORWARD, PipeInstruction.BACKWARD],
        server_url="http://127.0.0.1:7787",
        power_state_range=[1650, 1613, 1601],
    )

    # Check if get_power_state_schedule properly acts as a barrier.
    print(f"[{dp_rank},{pp_rank},{tp_rank}] Ask for next schedule")
    print(f"[{dp_rank},{pp_rank},{tp_rank}] {get_power_state_schedule()}")

    report_profiling_result([123.0 * rank] * world_size, [456.0 * rank] * world_size, {}, {})

    # Sleep a different amount of time.
    time.sleep(rank)

    # Check if get_power_state_schedule properly acts as a barrier.
    print(f"[{dp_rank},{pp_rank},{tp_rank}] Ask for next schedule")
    print(f"[{dp_rank},{pp_rank},{tp_rank}] {get_power_state_schedule()}")


def main():
    with mp.Pool(processes=dp * pp * tp) as pool:
        pool.map(worker, product(range(dp), range(pp), range(tp)))

if __name__ == "__main__":
    main()
