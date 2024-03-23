"""Provides report/next_batch_size during pruning stage."""

from __future__ import annotations
from uuid import UUID

import numpy as np
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateExploration,
    UpdateExploration,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import (
    BatchSizeBase,
    ExplorationsPerJob,
    Measurement,
)
from zeus.optimizer.batch_size.server.database.schema import State
from zeus.optimizer.batch_size.server.exceptions import ZeusBSOValueError
from zeus.optimizer.batch_size.server.job.commands import UpdateExpDefaultBs
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.util.logging import get_logger

logger = get_logger(__name__)


class PruningExploreManager:
    """Pruning manager that manges the batch size states in pruning stage."""

    def __init__(self, service: ZeusService):
        """Set up zeus service."""
        self.service = service

    def next_batch_size(
        self,
        job_id: UUID,
        exploration_history: ExplorationsPerJob,
        num_pruning_rounds: int,
        default_exp_bs: int,
        batch_sizes: list[int],
    ) -> int | Stage:
        """Find the next batch size to explore.

        Args:
            job_id: job id
            exploration_history: all explorations that we have done for that job
            num_pruning_rounds: maximum number of iteration over batch sizes for pruning
            default_exp_bs: default batch size for exploration
            batch_sizes: list of batch sizes to explore

        Returns:
            If there is a batch size to explore, then return the integer indicating the batch size.
            If it is a concurrent job submission, return Stage.pruning indicating that we are still waiting for the report for previously submitted batch size
            If pruning stage was over, return Stage.MAB to indicate that we should create arms.

        Raises:
            `ZeusBSOValueError`: If the value is invalid. EX) default batch size is not in the converged batch size list.
        """
        # Explore round.
        round_number = 0
        bs_list = batch_sizes

        # Get which round we are in now by finding the max round: it is possible that exploring current round is over
        # We can detect this case by iterating batch_sizes and check all possible batch sizes already have that round number (will be done after)
        for exps in exploration_history.explorations_per_bs.values():
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
                if converged_from_last_round is not None:
                    bs_list.append(bs)

        # Two cases below
        # (a) If round is over, update default_bs to min cost batch size and round
        # (b) If round is not over, reconstruct default bs and keep going

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

            logger.info("Exploration space: %s", str([down, up]))

            best_bs = default_exp_bs  # best_bs of current round
            min_cost = np.inf
            for bs_list in [down, up]:
                for bs in bs_list:
                    current_round_exp = None
                    if bs in exploration_history.explorations_per_bs:
                        for exp in exploration_history.explorations_per_bs[
                            bs
                        ].explorations:
                            if exp.round_number == round_number:
                                current_round_exp = exp

                    if current_round_exp is None:
                        # not explored this bs during this trial, should return this bs
                        next_batch_size = bs
                        break
                    elif current_round_exp.state == State.Unconverged:
                        # Unconverged batch size. Go to next list or next round
                        break
                    elif current_round_exp.state == State.Exploring:
                        # Concurrent job submission, give the best bs known so far (skip updating exploration states).
                        logger.info("Concurrent job submission. Waiting for %d", bs)
                        return Stage.Pruning
                    elif min_cost > current_round_exp.cost:
                        min_cost = current_round_exp.cost
                        best_bs = current_round_exp.batch_size

                if next_batch_size != -1:
                    break

            if next_batch_size == -1:
                # If we couldn't find a suitable batch size from this round -> this round is over

                round_number += 1
                if round_number > num_pruning_rounds:
                    # Exceeded pruning rounds, go to MAB stage
                    logger.info("Pruning over. go to MAB")
                    return Stage.MAB

                # Go to next round
                logger.info("Going to next round(%d)", round_number)
                if best_bs != default_exp_bs:
                    # Should update the exploration default bs to min_cost batch size
                    logger.info(
                        "Update default_bs to %d from %d", best_bs, default_exp_bs
                    )
                    self.service.update_exp_default_bs(
                        UpdateExpDefaultBs(
                            job_id=job_id, exp_default_batch_size=best_bs
                        ),
                    )
                # New round started. Always starting from exp_default_batch_size
                next_batch_size = best_bs

        # Add exploration to database
        self.service.add_exploration(
            CreateExploration(
                job_id=job_id, batch_size=next_batch_size, round_number=round_number
            )
        )

        return next_batch_size

    async def report_batch_size_result(
        self,
        current_meausurement: Measurement,
        cost: float,
    ) -> None:
        """Report whether the previous batch size reached the target metric.

        Should be called only when we are in Pruining stage.

        Args:
            current_meausurement: result of training, including time, energy and converged.
            cost: cost of training

        Raises:
            `ZeusBSOValueError`: When we couldn't find any explorations for that batch size.
        """
        explorations_per_bs = await self.service.get_explorations_of_bs(
            BatchSizeBase(
                job_id=current_meausurement.job_id,
                batch_size=current_meausurement.batch_size,
            )
        )

        if len(explorations_per_bs.explorations) == 0:
            raise ZeusBSOValueError(
                "Current batch_size(%d) is not in the batch_size list(%s)",
                current_meausurement.batch_size,
                str(explorations_per_bs.explorations),
            )
        logger.info("Explorations per bs: %s", str(explorations_per_bs))
        round_number = -1

        for exp in explorations_per_bs.explorations:  # round_number DESC
            if exp.state == State.Exploring:
                # Get most recent unreported exploration
                round_number = exp.round_number
                break

        if round_number == -1:
            logger.info(
                "Couldn't find issuing %d for exploration. Should be a concurrent job.",
                current_meausurement.batch_size,
            )
            self.service.report_concurrent_job(current_meausurement)
        else:
            state = (
                State.Converged if current_meausurement.converged else State.Unconverged
            )
            logger.info(
                "Update exploration for %d with state(%s).",
                current_meausurement.batch_size,
                str(state),
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
