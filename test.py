
from zeus.monitor import ZeusMonitor
from zeus.optimizer import GlobalPowerLimitOptimizer

if __name__ == '__main__':
    monitor = ZeusMonitor(gpu_indices=[0,1,2,3])

    monitor.begin_window()

    measurement = monitor.end_window("heavy computation")


    print(f"Energy: {measurement.total_energy} J")
    print(f"Time  : {measurement.time} s")


    plo = GlobalPowerLimitOptimizer(monitor)

    # training loop

    plo.on_step_begin()

    


