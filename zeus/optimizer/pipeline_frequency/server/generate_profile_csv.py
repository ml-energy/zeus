import os
import csv
import numpy as np

class PiecewiseLinearModel:
    """A simple model that interpolates energy measurements over time."""
    
    def __init__(self, time_measurements: np.ndarray, energy_measurements: np.ndarray) -> None:
        """
        Args:
            time_measurements: 1D array of timestamps.
            energy_measurements: 1D array of energy readings corresponding to the timestamps.
        """
        self.times = time_measurements
        self.energies = energy_measurements
        # Ensure measurements are sorted.
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("Time measurements must be sorted in ascending order.")
        if not np.all(np.diff(self.energies) >= 0):
            raise ValueError("Energy measurements must be sorted in ascending order.")

    def __call__(self, t: float) -> float:
        """
        Return the interpolated energy reading at time t.
        Raises ValueError if t is out of the measurement range.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(f"Time {t} is out of range [{self.times[0]}, {self.times[-1]}].")
        return np.interp(t, self.times, self.energies).item()


def generate_profile_csv(
    job_id: str,
    timing_data: dict[int, dict[str, list[tuple[float, float]]]],
    energy_data: dict[int, list[tuple[float, float]]],
    dump_dir: str,
    num_microbatches: int,
    num_prof_steps: int,
    warmup_iters: int,
    frequency_schedule: dict[int, list[int]] = None,
) -> str:
    """
    Generate a CSV profile that combines timing and energy data.
    
    For each rank and for each instruction type (e.g. 'forward', 'backward'),
    this function skips an initial number of warmup iterations (warmup_iters × batch_size)
    and then aggregates measurements in batches (batch_size = num_microbatches × num_prof_steps).
    The energy consumption during an instruction is estimated by constructing a piecewise
    linear model from the energy measurements and computing the difference between the energy
    readings at the end and at the start of the instruction.
    
    Optionally, if a frequency schedule is provided (as a list of frequency values per rank),
    the function will record the frequency applied for each batch.
    
    Args:
        job_id: The unique job identifier.
        timing_data: Dictionary mapping rank to a dict of instruction names to lists of
                     (start_time, end_time) tuples.
        energy_data: Dictionary mapping rank to a list of (time, energy) measurement tuples.
        dump_dir: Directory where the CSV file will be saved.
        num_microbatches: Number of microbatches per iteration.
        num_prof_steps: Number of profiling steps per iteration.
        warmup_iters: Number of warmup iterations to skip.
        frequency_schedule: Optional dict mapping rank to a list of frequency values for each batch.
    
    Returns:
        The file path of the generated CSV.
    """
    os.makedirs(dump_dir, exist_ok=True)
    output_path = os.path.join(dump_dir, f"{job_id}_profile.csv")
    
    # List the ranks from the timing data.
    ranks = list(timing_data.keys())
    
    # Build energy interpolation models per rank.
    models = {}
    for rank in ranks:
        # Expect energy_data[rank] to be a list of (time, energy) tuples.
        arr = np.array(energy_data[rank])
        times = arr[:, 0]
        energies = arr[:, 1]
        models[rank] = PiecewiseLinearModel(times, energies)
    
    # Generate the CSV file.
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rank", "instruction", "frequency", "avg_time", "avg_energy"])
        
        # Define the batch size for aggregating measurements.
        batch_size = num_microbatches * num_prof_steps
        
        for rank in ranks:
            rank_timing = timing_data[rank]
            # Get frequency iterator if a schedule is provided; otherwise, frequency is marked None.
            freq_iter = iter(frequency_schedule[rank]) if (frequency_schedule and rank in frequency_schedule) else None
            
            for inst, timings in rank_timing.items():
                if not timings:
                    continue  # No measurements for this instruction.
                
                inst_times_batch = []
                inst_energy_batch = []
                count = 0
                # Process each (start, end) measurement.
                for start, end in timings:
                    count += 1
                    # Skip warmup iterations.
                    if count <= warmup_iters * batch_size:
                        continue
                    inst_times_batch.append(end - start)
                    try:
                        # Compute energy consumption during this instruction via interpolation.
                        energy_delta = models[rank](end) - models[rank](start)
                    except ValueError:
                        # If the instruction timing is out of range of the energy measurements, skip it.
                        continue
                    inst_energy_batch.append(energy_delta)
                    
                    # When a full batch is accumulated, compute averages and write a row.
                    if (count - warmup_iters * batch_size) % batch_size == 0 and inst_times_batch:
                        avg_time = np.mean(inst_times_batch)
                        avg_energy = np.mean(inst_energy_batch)
                        freq = next(freq_iter) if freq_iter is not None else "N/A"
                        writer.writerow([rank, inst, freq, avg_time, avg_energy])
                        inst_times_batch = []
                        inst_energy_batch = []
    return output_path