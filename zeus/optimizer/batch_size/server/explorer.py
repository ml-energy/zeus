"""Provides report/next_batch_size during pruning stage."""

from __future__ import annotations

from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateConcurrentTrial,
    CreateExplorationTrial,
    ReadTrial,
)
from zeus.optimizer.batch_size.server.batch_size_state.models import ExplorationsPerJob
from zeus.optimizer.batch_size.server.database.schema import TrialStatus
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOServerRuntimeError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.job.models import JobState
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.logging import get_logger
from zeus.utils.metric import zeus_cost

logger = get_logger(__name__)


class PruningExploreManager:
    """Pruning manager that manges the batch size states in pruning stage."""

    def __init__(self, service: ZeusService):
        """Set up zeus service."""
        self.service = service

    async def next_batch_size(
        self,
        job: JobState,
        exploration_history: ExplorationsPerJob,
    ) -> ReadTrial | list[int]:
        """Find the next batch size to explore.

        Three cases possible.
        1. Pruninig Stage : There is a batch size that has not explored during the round.
        2. Concurrent job : There is an exploration with "Dispatched" state.
        3. Mab stage : All batch sizes have been explored and round is over.

        Args:
            job: state of the job
            exploration_history: all "succeeded" explorations that we have done for that job

        Returns:
            Return the batch size to use during Pruning stage.
            If Pruning stage was over, return None.

        Raises:
            `ZeusBSOValueError`: If the value is invalid. EX) default batch size is not in the converged batch size list.
        """
        batch_sizes = job.batch_sizes
        exp_default_bs = job.default_batch_size

        for round in range(job.num_pruning_rounds):
            converged_bs_list = []

            min_cost_of_round = float("inf")
            min_batch_size_of_round = 0

            batch_sizes.sort()
            idx = batch_sizes.index(exp_default_bs)
            down = sorted(batch_sizes[: idx + 1], reverse=True)
            up = sorted(batch_sizes[idx + 1 :])

            for bs_list in [down, up]:
                for bs in bs_list:
                    if (
                        bs in exploration_history.explorations_per_bs
                        and len(exploration_history.explorations_per_bs[bs]) > round
                    ):
                        # Already explored at this round
                        if (
                            exploration_history.explorations_per_bs[bs][round].status
                            == TrialStatus.Dispatched
                        ):
                            # We are waiting for the result of this exploration -> Concurrent job!
                            return await self.service.create_trial(
                                CreateConcurrentTrial(
                                    job_id=job.job_id,
                                    batch_size=job.min_cost_batch_size,
                                )
                            )

                        if not exploration_history.explorations_per_bs[bs][
                            round
                        ].converged:
                            # Failed to converge -> Go to next list or round
                            break
                        else:
                            # Training converged.
                            converged_bs_list.append(bs)

                            m = exploration_history.explorations_per_bs[bs][round]
                            if m.energy is None or m.time is None:
                                raise ZeusBSOValueError(
                                    "Energy or time is not available for the exploration."
                                )
                            cost = zeus_cost(
                                m.energy, m.time, job.eta_knob, job.max_power
                            )
                            if cost < min_cost_of_round:
                                min_cost_of_round = cost
                                min_batch_size_of_round = bs

                    else:
                        # Did not explore this round. Should explore!
                        return await self.service.create_trial(
                            CreateExplorationTrial(
                                job_id=job.job_id,
                                batch_size=bs,
                            )
                        )

            # We should go to next round. Update exp_default_bs and batch sizes!
            exp_default_bs = min_batch_size_of_round
            batch_sizes = converged_bs_list

            logger.info(
                "[PruningExploreManager] go to next round(%d) new default bs = %d converged bs list = %s",
                round,
                exp_default_bs,
                batch_sizes,
            )

            if len(batch_sizes) == 0:
                raise ZeusBSOServerRuntimeError(
                    "No converged batch sizes has observed. Reconfigure batch_sizes and re-launch the job."
                )
        # After going through pruning rounds, we couldn't find the bs. Should go to MAB stage, so return good batch_sizes.
        return sorted(batch_sizes)
