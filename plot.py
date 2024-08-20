from __future__ import annotations

from pathlib import Path
from glob import glob
from typing import Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, validator

from perseus.utils.state import load_prof
from perseus.utils.analysis import total_time_and_energy

# Data root directory
DATA = "data"


class Workload(BaseModel):
    """Dataclass that represents a single workload."""
    framework: Literal["merak", "alpa"] = "merak"
    model_name: str
    partition_method: str
    dp: int
    tp: int
    pp: int
    gpu_type: Literal["A40", "V100", "A100"] = "A40"
    microbatch_size: Union[int, Literal["optimal"]]
    num_microbatches: Optional[Union[int, Literal["optimal"]]] = None
    point_solution_path: Optional[str] = None  # Allows overriding of PD path (PointSolution result).
        
    @validator("microbatch_size")
    def _set_mbs(cls, microbatch_size, values):
        if microbatch_size == "optimal":
            parallel = f"dp{values['dp']}+pp{values['pp']}+tp{values['tp']}"
            workloads_df = pd.read_csv(f"{DATA}/workloads/{parallel}/models_{values['gpu_type']}.csv")
            workload_row = workloads_df[(workloads_df["model"] == values["model_name"]) \
                                        & (workloads_df["partition_method"] == values["partition_method"]) \
                                        & (workloads_df["dp"] == values["dp"]) & (workloads_df["tp"] == values["tp"]) \
                                        & (workloads_df["pp"] == values["pp"])]
            if workload_row.empty:
                raise ValueError(f"Entry for {values} not found in {DATA}/workloads/{parallel}/models_{values['gpu_type']}.csv")
            microbatch_size = workload_row["microbatch_size"].item()
        return microbatch_size
    
    @validator("num_microbatches")
    def _set_nmb(cls, num_microbatches, values):
        if num_microbatches == "optimal":
            parallel = f"dp{values['dp']}+pp{values['pp']}+tp{values['tp']}"
            workloads_df = pd.read_csv(f"{DATA}/workloads/{parallel}/models_{values['gpu_type']}.csv")
            workload_row = workloads_df[(workloads_df["model"] == values["model_name"]) \
                                        & (workloads_df["partition_method"] == values["partition_method"]) \
                                        & (workloads_df["dp"] == values["dp"]) & (workloads_df["tp"] == values["tp"]) \
                                        & (workloads_df["pp"] == values["pp"])]
            if workload_row.empty:
                raise ValueError(f"Entry for {values} not found in {DATA}/workloads/{parallel}/models_{values['gpu_type']}.csv")
            num_microbatches = workload_row["num_microbatches"].item()
        return num_microbatches

    @validator("point_solution_path")
    def _check_point_solution_path(cls, point_solution_path):
        # Make sure the path exists.
        if point_solution_path is not None and not Path(point_solution_path).exists():
            raise ValueError(f"{point_solution_path=} doesn't exist.")

        return point_solution_path

    def to_path(self, add_gpu: bool = True) -> str:        
        path = (
            f"{self.framework}+{self.model_name}+{self.partition_method}"
            f"+dp{self.dp}+pp{self.pp}+tp{self.tp}+mbs{self.microbatch_size}"
        )
        if self.num_microbatches is not None:
            path += f"+nmb{self.num_microbatches}"
        if add_gpu:
            path += f"+{self.gpu_type}"
        return path
    
    def get_zeus_df(self, scheduler: Literal["global", "local"]) -> pd.DataFrame:
        assert self.num_microbatches is not None
        path = f"{DATA}/zeus/csv/{self.gpu_type}/zeus_{scheduler}.csv"
        df = pd.read_csv(path)
        return df[(df["model_name"] == self.model_name) & (df["partition_method"] == self.partition_method) \
                  & (df["dp"] == self.dp) & (df["tp"] == self.tp) & (df["pp"] == self.pp) \
                  & (df["microbatch_size"] == self.microbatch_size)
                  & (df["num_microbatches"] == self.num_microbatches)
        ]

    def get_perseus_df(self, warmup_steps: int = 2, warmup_iters: int = 2) -> pd.DataFrame:
        if self.point_solution_path is not None:
            profdir = self.point_solution_path
        else:
            parallel = f"dp{self.dp}+pp{self.pp}+tp{self.tp}"
            # Small hack to support for 3D and DP-only
            if self.pp == 1 or (self.dp == 2 and self.tp == 2):
                candidates = sorted(glob(f"{DATA}/perseus/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution3D"))
                if not candidates:
                    raise ValueError(f"'{DATA}/perseus/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution3D' doesn't exist")
            else:
                candidates = sorted(glob(f"{DATA}/perseus/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution"))
                if not candidates:
                    raise ValueError(f"'{DATA}/perseus/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution' doesn't exist")
            profdir = candidates[-1]
        print(f"Reading Perseus results from {profdir}")
        num_prof_files = len(glob(f"{profdir}/*.prof.json"))
        profs = [load_prof(profdir, i) for i in range(warmup_iters, num_prof_files)]
        perseus_results = []
        for prof in profs:
            t, e = total_time_and_energy(prof, warmup_steps=warmup_steps)
            perseus_results.append([t, e])
        if perseus_results[0] < perseus_results[-1]:
            perseus_results = perseus_results[::-1]
        return pd.DataFrame(perseus_results, columns=['time', 'energy'])

    def get_envpipe_time_energy(self, warmup_steps: int = 2, warmup_iters: int = 2) -> tuple[float, float]:
        """Return the time and energy consumption of the frequency reconfiguration of EnvPipe."""
        parallel = f"dp{self.dp}+pp{self.pp}+tp{self.tp}"
        candidates = sorted(glob(f"{DATA}/envpipe/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution"))
        if not candidates:
            raise ValueError(f"'{DATA}/envpipe/{self.gpu_type}/{parallel}/*{self.to_path(add_gpu=False)}+PointSolution' doesn't exist")
        profdir = candidates[-1]
        print(f"Reading EnvPipe reconfig results from {profdir}")
        num_prof_files = len(glob(f"{profdir}/*.prof.json"))
        profs = [load_prof(profdir, i) for i in range(warmup_iters, num_prof_files)]
        if len(profs) != 1:
            raise ValueError("EnvPipe reconfig should only have one profile.")
        return total_time_and_energy(profs[0], warmup_steps=warmup_steps)
