"""Profile the power consumption of the GPU while waiting on P2P communication."""

import os
import time
import multiprocessing as mp

import torch
import torch.distributed as dist
from zeus.monitor import ZeusMonitor


def main() -> None:
    """Run the main routine."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    worker0 = mp.Process(target=worker, args=(0,))
    worker1 = mp.Process(target=worker, args=(1,))

    worker0.start()
    worker1.start()

    worker0.join()
    worker1.join()


def worker(rank: int) -> None:
    """Run the worker routine."""
    if rank not in [0, 1]:
        raise ValueError(f"Invalid rank: {rank}")

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=2, rank=rank
    )

    # Allocate large tensor and run some computation to warm up the GPU.
    tensor = torch.rand(10000, 10000, device="cuda")
    tensor = (
        tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
        @ tensor
    )

    if rank == 0:
        monitor = ZeusMonitor(gpu_indices=[rank])

        # Communication warmup
        for _ in range(5):
            dist.recv(tensor, src=1 - rank)
            dist.send(tensor, dst=1 - rank)
            torch.cuda.synchronize()

        # Measure while the GPU is blocking on P2P communication.
        # Rank 1 is just sleeping.
        monitor.begin_window("p2p")
        dist.recv(tensor, src=1 - rank)
        measurement = monitor.end_window("p2p")

        torch.cuda.synchronize()

        print(f"Time (s): {measurement.time}")
        print(f"Energy (J): {measurement.total_energy}")
        print(f"Power (W): {measurement.total_energy / measurement.time}")

    else:
        # Communication warmup
        for _ in range(5):
            dist.send(tensor, dst=1 - rank)
            dist.recv(tensor, src=1 - rank)
            torch.cuda.synchronize()

        print("Sleeping for 60 seconds")
        time.sleep(60)
        dist.send(tensor, dst=1 - rank)

        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
