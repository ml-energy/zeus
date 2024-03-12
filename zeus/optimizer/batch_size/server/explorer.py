from uuid import UUID

import numpy as np
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpdateExploration,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    MeasurementOfBs,
)
from zeus.optimizer.batch_size.server.database.schema import State
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError
from zeus.optimizer.batch_size.server.job.commands import UpdateExpDefaultBs
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.services.service import ZeusService


class PruningExploreManager:
    """Helper class that generates batch sizes to explore and prune."""

    def __init__(self, service: ZeusService):
        self.service = service

    def next_batch_size(
        self,
        job_id: UUID,
        exploration_history: ExplorationsPerJob,
        num_pruning_rounds: int,
        default_exp_bs: int,
        batch_sizes: list[int],
    ) -> int | Stage:
        """Return the next batch size to explore.
        batch_sizes is already sorted from query (batch_size ASC)
        """
        # Explore round.
        round_number = 0
        bs_list = batch_sizes

        # Get which round we are in now by finding the max round: it is possible that exploring current round is over
        # We can detect this case by iterating batch_sizes and check all possible batch sizes already have that round number (will be done after)
        for bs, exps in exploration_history.explorations_per_bs.items():
            latest_round_of_bs = max(
                exps.explorations, key=lambda exp: exp.round_number, default=None
            )
            if (
                latest_round_of_bs is not None
                and latest_round_of_bs.round_number > round_number
            ):
                round_number = latest_round_of_bs.round_number

        if round_number > 1:
            bs_list.clear()
            # Filter the converged batch_sizes from the previous round
            for bs, exps in exploration_history.explorations_per_bs.items():
                converged_from_last_round = next(
                    (
                        exp.batch_size
                        for exp in exps.explorations
                        if exp.round_number == round_number - 1
                        and exp.state == State.Converged
                    ),
                    None,
                )
                if converged_from_last_round != None:
                    bs_list.append(bs)

        """
        (a) If round is over, update default_bs and round 
            How to update the default_bs? -> From explorationState.cost (from that round)!
        (b) If round is not over, reconstruct default bs and keep going!
        """

        next_batch_size = -1
        if round_number == 0:
            # No trials has been done, update exp state and return default bs
            next_batch_size = default_exp_bs
            round_number = 1
        else:
            # trial > 0
            if default_exp_bs not in bs_list:
                raise ZeusBSOValueError(
                    f"{default_exp_bs} is not in the exploration space({[bs_list]})"
                )

            idx = bs_list.index(default_exp_bs)
            down = sorted(bs_list[: idx + 1], reverse=True)
            up = sorted(bs_list[idx + 1 :])

            self._log(f"Exploration space: {[down, up]}")

            best_bs = default_exp_bs  # best_bs of current round
            min_cost = np.inf
            for bs_list in [down, up]:
                for bs in bs_list:
                    current_round_exp = None
                    if bs in exploration_history.explorations_per_bs.keys():
                        for exp in exploration_history.explorations_per_bs[
                            bs
                        ].explorations:
                            if exp.round_number == round_number:
                                current_round_exp = exp

                    if current_round_exp == None:
                        # not explored this bs during this trial, return!
                        next_batch_size = bs
                        break
                    elif current_round_exp.state == State.Unconverged:
                        # Unconverged batch size. Go to next list or next round
                        break
                    elif current_round_exp.state == State.Exploring:
                        # Concurrent job submission, give the best bs known so far (skip updating exploration states).
                        self._log(f"Concurrent job submission. Waiting for {bs}")
                        return Stage.Pruning
                    else:
                        if min_cost > current_round_exp.cost:
                            min_cost = current_round_exp.cost
                            best_bs = current_round_exp.batch_size

                if next_batch_size != -1:
                    break

            if next_batch_size == -1:
                # If we couldn't find a suitable batch size from this round -> this round is over

                round_number += 1
                if round_number > num_pruning_rounds:
                    # Exceeded pruning rounds, go to MAB stage
                    self._log(f"Pruning over. go to MAB")
                    return Stage.MAB

                self._log(f"Going to next round({round_number})")
                if best_bs != default_exp_bs:
                    self._log(f"Update default_bs to {best_bs} from {default_exp_bs}")
                next_batch_size = best_bs

                self.service.update_exp_default_bs(
                    UpdateExpDefaultBs(job_id=job_id, exp_default_batch_size=best_bs),
                )
        # Add exploration at the end
        self.service.add_exploration(
            CreateExploration(
                job_id=job_id,
                batch_size=next_batch_size,
                round_number=round_number,
                state=State.Exploring,
            )
        )

        return next_batch_size

    # Add measurement before calling this function and also should know this is for reporting to exploration!
    async def report_batch_size_result(
        self,
        current_meausurement: MeasurementOfBs,
        within_cost_range: bool,
        cost: float,
    ) -> None:
        """Report whether the previous batch size reached the target metric.

        Need to change the default_bs if the round was over.

        Args:
            batch_size: The batch size which this cost observation is from.
            cost: The energy-time cost of running the job with this batch size.
            reached: Whether the job reached the target metric.
        """

        explorations_per_bs = await self.service.get_explorations_of_bs(
            BatchSizeBase(
                job_id=current_meausurement.job_id,
                batch_size=current_meausurement.batch_size,
            )
        )

        if len(explorations_per_bs.explorations) == 0:
            raise ZeusBSOValueError(
                f"Current batch_size({current_meausurement.batch_size}) is not in the batch_size list({explorations_per_bs.explorations})"
            )
        self._log(f"Explorations per bs: {explorations_per_bs}")
        round_number = -1

        for exp in explorations_per_bs.explorations:  # round_number DESC
            if exp.state == State.Exploring:
                # Get most recent unreported exploration
                round_number = exp.round_number
                break

        if round_number == -1:
            self._log(
                f"Couldn't find issuing {current_meausurement.batch_size} for exploration. Should be a concurrent job."
            )
            self.service.report_concurrent_job(current_meausurement)
        else:
            state = (
                State.Converged
                if current_meausurement.converged and within_cost_range
                else State.Unconverged
            )
            self._log(
                f"Update exploration for {current_meausurement.batch_size} with state({state})."
            )
            await self.service.update_exploration(
                current_meausurement,
                UpdateExploration(
                    job_id=current_meausurement.job_id,
                    batch_size=current_meausurement.batch_size,
                    round_number=round_number,
                    state=state,
                    cost=cost,
                ),
            )

    def _log(self, message: str) -> None:
        """Log message with object name."""
        print(f"[Pruning Explore Manager] {message}")
