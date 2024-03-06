from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio.session import AsyncSession
from zeus.optimizer.batch_size.common import ZeusBSOValueError
from zeus.optimizer.batch_size.server.database.dbapi import DBapi
from zeus.optimizer.batch_size.server.database.schema import BatchSize, ExplorationState
from zeus.optimizer.batch_size.server.job.models import Stage


class PruningExploreManager(object):
    """Helper class that generates batch sizes to explore and prune."""

    @staticmethod
    async def next_batch_size(
        db: AsyncSession,
        job_id: UUID,
        batch_sizes: list[BatchSize],
        num_pruning_rounds: int,
        default_batch_size: int,
    ) -> int | Stage:
        """Return the next batch size to explore.
        batch_sizes is already sorted from query (batch_size ASC)
        """
        trial = 0
        idx = -1
        for i, v in enumerate(batch_sizes):
            if v.batch_size == default_batch_size:
                idx = i
            latest_trial_of_bs = max(
                v.explorations, key=lambda exp: exp.trial_number, default=None
            )
            if (
                latest_trial_of_bs is not None
                and latest_trial_of_bs.trial_number > trial
            ):
                trial = latest_trial_of_bs.trial_number

        if idx == -1:
            raise ZeusBSOValueError(
                f"Default bs({default_batch_size}) is not in batch_size list({[bs.batch_size for bs in batch_sizes]})"
            )

        exp_bs: list[BatchSize] = []

        if trial > 1:
            for bs in batch_sizes:
                last_trial_exp = next(
                    (exp for exp in bs.explorations if exp.trial_number == trial - 1),
                    None,
                )
                if (
                    last_trial_exp is not None
                    and last_trial_exp.state == ExplorationState.State.Converged
                ):
                    exp_bs.append(bs)
            idx = next(
                (
                    i
                    for i, bs in enumerate(exp_bs)
                    if bs.batch_size == default_batch_size
                ),
                None,
            )
            if idx == None:
                raise ZeusBSOValueError(
                    f"Default bs({default_batch_size}) is not in converged batch_size list from last round({[bs.batch_size for bs in exp_bs]})"
                )
        else:
            exp_bs = batch_sizes

        down = exp_bs[: idx + 1]
        down.reverse()
        up = exp_bs[idx + 1 :]

        PruningExploreManager._log(
            f"Exploration space: {[[bs.batch_size for bs in down],[bs.batch_size for bs in up]]}"
        )
        """
        Trial => MAX TRIAL_NUMBER from all explorations , 
        (a) If round is over, update default_bs and round 
            How to update the default_bs? -> From explorationState.cost (from that trial_number)!
        (b) If round is not over, reconstruct default bs and keep going!

        How to know if round is over? -> Assume max trial_number is the current round.
        Construct a [down,up] and iterate. If we find Unconverge, break -> If we don't have any batch size to report,
        this means we should go to next round 
        """
        next_batch_size = -1
        if trial == 0:
            # No trials has been done, update exp state and return default bs
            next_batch_size = default_batch_size
            trial = 1
        else:
            # trial > 0
            best_bs = default_batch_size  # best_bs of current trial
            min_cost = np.inf
            for bs_list in [down, up]:
                for bs in bs_list:
                    cur_exp = next(
                        (exp for exp in bs.explorations if exp.trial_number == trial),
                        None,
                    )
                    if cur_exp == None:
                        # not explored this bs during this trial, return!
                        next_batch_size = bs.batch_size
                        break
                    elif cur_exp.state == ExplorationState.State.Unconverged:
                        break
                    elif cur_exp.state == ExplorationState.State.Exploring:
                        # Concurrent job submission, give the best bs known so far (skip updating exploration states).
                        PruningExploreManager._log(
                            f"Concurrent job submission. Waiting for {bs.batch_size}"
                        )
                        return Stage.Pruning
                    else:
                        if min_cost > cur_exp.cost:
                            min_cost = cur_exp.cost
                            best_bs = cur_exp.batch_size

                if next_batch_size != -1:
                    break

            if next_batch_size == -1:
                # this trial is over. Need to go to next round!
                trial += 1
                if trial > num_pruning_rounds:
                    # Exceeded pruning rounds, go to MAB stage
                    PruningExploreManager._log(f"Pruning over. go to MAB")
                    return Stage.MAB
                PruningExploreManager._log(f"Going to next trial({trial})")
                if best_bs != default_batch_size:
                    PruningExploreManager._log(
                        f"Update default_bs to {best_bs} from {default_batch_size}"
                    )
                    await DBapi.update_exp_default_bs(db, job_id, best_bs)
                next_batch_size = best_bs

        DBapi.add_exploration(
            db,
            job_id,
            next_batch_size,
            trial,
            ExplorationState.State.Exploring,
        )
        return next_batch_size

    # Add measurement before calling this function and also should know this is for reporting to exploration!
    @staticmethod
    async def report_batch_size_result(
        db: AsyncSession,
        bs: BatchSize,
        converged: bool,
        cost: float,
        within_cost_range: bool,
    ) -> None:
        """Report whether the previous batch size reached the target metric.

        Need to change the default_bs if the round was over.

        Args:
            batch_size: The batch size which this cost observation is from.
            cost: The energy-time cost of running the job with this batch size.
            reached: Whether the job reached the target metric.
        """

        trial_number = -1

        for exp in bs.explorations:  # sorted on trial_number ASC from query
            if exp.state == ExplorationState.State.Exploring:
                # Get most recent unreported exploration
                trial_number = exp.trial_number
                break

        if trial_number == -1:
            PruningExploreManager._log(
                f"Couldn't find issuing {bs} for exploration. Should be a concurrent job."
            )
            return

        state = (
            ExplorationState.State.Converged
            if converged and within_cost_range
            else ExplorationState.State.Unconverged
        )
        PruningExploreManager._log(f"Update exploration for {bs.batch_size}.")
        await DBapi.update_exploration(
            db, bs.job_id, bs.batch_size, trial_number, state, cost
        )

    @staticmethod
    def _log(message: str) -> None:
        """Log message with object name."""
        print(f"[Pruning Explore Manager] {message}")
