"""GPU validation against a live zeusd on Windows.

Runs against zeusd serving on http://127.0.0.1:4938 with `gpu-control,gpu-read` enabled.

What this script verifies:
  1. `zeus.device.gpu` can introspect the local NVIDIA GPU (power-limit
     constraints and supported clocks).
  2. zeusd `/gpu/get_power` and `/gpu/get_cumulative_energy` return sane
     values at idle.
  3. A sustained PyTorch matmul drives the GPU power up; zeusd's polling
     observes the rise and the cumulative-energy counter is monotonic.
  4. zeusd's `POST /gpu/set_power_limit` round-trips for valid values
     (readback via NVML matches) and returns HTTP 400 for out-of-range
     values (the macro-status-code fix).
  5. zeusd's `POST /gpu/set_gpu_locked_clocks` / `reset_gpu_locked_clocks`
     round-trip for several frequency targets.

Exits non-zero on any failed assertion.
"""
import json
import statistics
import sys
import threading
import time

import requests
import torch

ZEUSD = "http://127.0.0.1:4938"
GPU_ID = 0


def banner(s):
    print(f"\n========== {s} ==========", flush=True)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------- 1. zeus.device.gpu introspection ----------
banner("zeus.device.gpu introspection")
from zeus.device import gpu as zgpu

gpus = zgpu.get_gpus()
if len(gpus.gpus) < 1:
    fail("zeus.device.gpu reports zero GPUs")
g = gpus.gpus[GPU_ID]
print(f"GPU 0: {type(g).__name__}  name={g.get_name()}")

pmin, pmax = g.get_power_management_limit_constraints()
current_pl = g.get_power_management_limit()
mem_clocks = g.get_supported_memory_clocks()
gfx_clocks_per_mem = {mc: g.get_supported_graphics_clocks(mc) for mc in mem_clocks[:3]}
print(json.dumps({
    "power_limit_mw_min": pmin,
    "power_limit_mw_max": pmax,
    "power_limit_mw_current": current_pl,
    "mem_clocks_mhz": mem_clocks,
    "gfx_clocks_per_mem_mhz_first3": {
        k: f"len={len(v)} min={min(v)} max={max(v)}"
        for k, v in gfx_clocks_per_mem.items()
    },
}, indent=2))

if not (isinstance(pmin, int) and isinstance(pmax, int) and pmin > 0 and pmax >= pmin):
    fail(f"power-limit constraints look wrong: ({pmin}, {pmax})")
if not mem_clocks:
    fail("zero supported memory clocks")

# ---------- 2. zeusd /discover ----------
banner("zeusd /discover")
disc = requests.get(f"{ZEUSD}/discover", timeout=5).json()
print(json.dumps(disc, indent=2))
if GPU_ID not in disc["gpu_ids"]:
    fail(f"GPU {GPU_ID} not listed in zeusd discovery")


def poll_once():
    rp = requests.get(f"{ZEUSD}/gpu/get_power", timeout=3).json()
    re = requests.get(f"{ZEUSD}/gpu/get_cumulative_energy", timeout=3).json()
    return rp["power_mw"][str(GPU_ID)], re[str(GPU_ID)]["energy_mj"]


# ---------- 3. idle baseline ----------
banner("idle baseline (5 s)")
idle = []
for _ in range(10):
    pw, en = poll_once()
    idle.append((time.time(), pw, en))
    time.sleep(0.5)
idle_pws = [s[1] for s in idle]
idle_ens = [s[2] for s in idle]
idle_p_med = statistics.median(idle_pws)
print(f"idle power_mw median={idle_p_med:.0f} min={min(idle_pws)} max={max(idle_pws)}")
print(f"idle energy_mj first={idle_ens[0]} last={idle_ens[-1]} delta={idle_ens[-1]-idle_ens[0]}")
if not all(idle_ens[i] <= idle_ens[i+1] for i in range(len(idle_ens)-1)):
    fail("idle energy counter is NOT monotonically non-decreasing")

# ---------- 4. matmul load + polling ----------
banner("matmul load (10 s) + polling")
device = torch.device("cuda", GPU_ID)
A = torch.randn(8192, 8192, device=device, dtype=torch.float32)
B = torch.randn(8192, 8192, device=device, dtype=torch.float32)
torch.cuda.synchronize()

stop = threading.Event()
load_samples = []


def poll_thread():
    while not stop.is_set():
        try:
            load_samples.append((time.time(), *poll_once()))
        except Exception:
            pass
        time.sleep(0.2)


t = threading.Thread(target=poll_thread, daemon=True)
t.start()
iters = 0
end = time.time() + 10
while time.time() < end:
    C = A @ B
    torch.cuda.synchronize()
    iters += 1
stop.set()
t.join()
del C

load_pws = [s[1] for s in load_samples]
load_ens = [s[2] for s in load_samples]
load_p_med = statistics.median(load_pws)
load_p_max = max(load_pws)
print(f"matmul iters: {iters}   polls: {len(load_samples)}")
print(f"load power_mw median={load_p_med:.0f} max={load_p_max}")
print(f"load energy_mj first={load_ens[0]} last={load_ens[-1]} delta_J={(load_ens[-1]-load_ens[0])/1000:.1f}")

if not all(load_ens[i] <= load_ens[i+1] for i in range(len(load_ens)-1)):
    fail("load energy counter is NOT monotonically non-decreasing")
# Load median should be visibly higher than idle median. T4 idle ~14W, load ~70W;
# require at least 3x of idle median as a sanity check.
if load_p_med < 3 * idle_p_med:
    fail(f"load median power ({load_p_med:.0f} mW) not >= 3x idle median ({idle_p_med:.0f} mW)")
# Load max should approach the power cap (within 5% headroom)
if load_p_max < int(0.9 * pmax):
    print(f"WARN: load max {load_p_max} mW is well below cap {pmax} mW; workload may not be saturating GPU")

# ---------- 5. power-limit round-trips ----------
banner("set_power_limit round-trip via zeusd")


def set_pl(mw, expected_status):
    r = requests.post(
        f"{ZEUSD}/gpu/set_power_limit",
        params={"gpu_ids": "0", "power_limit_mw": mw, "block": "true"},
        timeout=5,
    )
    rb = g.get_power_management_limit()
    print(f"  set {mw} mW -> HTTP {r.status_code} readback={rb}")
    if r.status_code != expected_status:
        fail(f"expected HTTP {expected_status}, got {r.status_code} for set {mw}")
    return rb


mid = (pmin + pmax) // 2
for v in (pmin, mid, pmax):
    rb = set_pl(v, 200)
    if rb != v:
        fail(f"readback {rb} != requested {v}")
set_pl(pmin - 1000, 400)
set_pl(pmax + 50000, 400)
set_pl(current_pl, 200)  # restore

# ---------- 6. locked-clocks round-trips ----------
banner("set/reset_gpu_locked_clocks via zeusd")
all_gfx = sorted({c for clocks in gfx_clocks_per_mem.values() for c in clocks})
for f in (all_gfx[0], all_gfx[len(all_gfx) // 2], all_gfx[-1]):
    r = requests.post(
        f"{ZEUSD}/gpu/set_gpu_locked_clocks",
        params={"gpu_ids": "0", "min_clock_mhz": f, "max_clock_mhz": f, "block": "true"},
        timeout=5,
    )
    print(f"  set [{f},{f}] MHz -> HTTP {r.status_code}")
    if r.status_code != 200:
        fail(f"expected 200 for locked-clocks {f}, got {r.status_code}")
r = requests.post(
    f"{ZEUSD}/gpu/reset_gpu_locked_clocks",
    params={"gpu_ids": "0", "block": "true"},
    timeout=5,
)
if r.status_code != 200:
    fail(f"expected 200 for reset_gpu_locked_clocks, got {r.status_code}")
print("  reset -> HTTP 200")

print("\nLOAD_TEST_OK", flush=True)
