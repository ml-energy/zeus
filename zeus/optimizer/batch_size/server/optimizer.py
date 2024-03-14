from __future__ import annotations

from uuid import UUID

import numpy as np
from zeus.optimizer.batch_size.common import JobSpec, ReportResponse, TrainingResult
from zeus.optimizer.batch_size.server.batch_size_state.models import MeasurementOfBs
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOJobSpecMismatchError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.optimizer.batch_size.server.job.commands import CreateJob
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.services.commands import CreateArms
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.util.logging import get_logger
from zeus.util.metric import zeus_cost

logger = get_logger(__name__)


class ZeusBatchSizeOptimizer:
    def __init__(self, service: ZeusService) -> None:
        """Initialize the server."""
        self.service = service
        self.pruning_manager = PruningExploreManager(service)
        self.mab = GaussianTS(service)

    @property
    def name(self) -> str:
        """Name of the batch size optimizer."""
        return "ZeusBatchSizeOptimizer"

    async def register_job(self, job: JobSpec) -> bool:
        """Register a user-submitted job. Return number of newly created job. Return the number of job that is registered"""
        registered_job = await self.service.get_job(job.job_id)

        if registered_job is not None:
            # Job exists
            logger.info(f"Job({job.job_id}) already exists")
            registerd_job_spec = JobSpec.parse_obj(registered_job.dict())

            if registerd_job_spec != job:
                raise ZeusBSOJobSpecMismatchError(
                    "JobSpec doesn't match with existing jobSpec. Use a new job_id for different configuration"
                )
            return False

        self.service.create_job(CreateJob.from_jobSpec(job))
        logger.info(f"Registered {job.job_id}")

        return True

    async def predict(self, job_id: UUID) -> int:
        """Return a batch size to use."""
        job = await self.service.get_job(job_id)

        if job == None:
            raise ZeusBSOValueError(
                f"Unknown job({job_id}). Please register the job first"
            )

        if job.stage == Stage.MAB:
            arms = await self.service.get_arms(job_id)
            return self.mab.predict(
                job_id, job.mab_prior_precision, job.mab_num_exploration, arms
            )
        else:
            # Pruning stage
            explorations = await self.service.get_explorations_of_job(job_id)
            # First check if pruning explorer can give us any batch size
            next_batch_size = self.pruning_manager.next_batch_size(
                job_id,
                explorations,
                job.num_pruning_rounds,
                job.exp_default_batch_size,
                job.batch_sizes,
            )

            if isinstance(next_batch_size, Stage) and next_batch_size == Stage.Pruning:
                # Concurrent job submission during pruning stage
                return job.min_batch_size
            elif isinstance(next_batch_size, Stage):
                # MAB stage: construct MAB if we haven't done yet and return the batch size from MAB
                logger.info(
                    f"Constructing a MAB {explorations.explorations_per_bs == None}"
                )
                arms = await self.mab.construct_mab(
                    job,
                    CreateArms(
                        job_id=job_id,
                        explorations_per_bs=explorations.explorations_per_bs,
                    ),
                )
                return self.mab.predict(
                    job_id, job.mab_prior_precision, job.mab_num_exploration, arms
                )
            else:
                # Exploration stage and got the next available batch size to explore from the explore manager
                return next_batch_size

    async def report(self, result: TrainingResult) -> ReportResponse:
        """
        Report the training result
        3 cases.
        1) MAB: Update arm state for reported batch size*
        2) Pruning
            2-a) concurrent job -> skip
            2-b) update exploration states for corresponding batch sizes
        3) Keep training
        """
        cost_ub = np.inf
        job = await self.service.get_job(result.job_id)
        if job.beta_knob > 0 and job.min_cost != None:  # Early stop enabled
            cost_ub = job.beta_knob * job.min_cost

        reported_cost = zeus_cost(
            result.energy,
            result.time,
            job.eta_knob,
            job.max_power,
        )

        converged = (
            job.high_is_better_metric and job.target_metric <= result.metric
        ) or (not job.high_is_better_metric and job.target_metric >= result.metric)

        if (
            cost_ub >= reported_cost
            and result.current_epoch < job.max_epochs
            and converged == False
        ):
            # If it's not converged but below cost upper bound and haven't reached max_epochs, keep training

            return ReportResponse(
                stop_train=False,
                converged=False,
                message="Stop condition not met, keep training",
            )

        # Two cases below here (training ended)
        # 1. Converged == true
        # 2. reached max_epoch OR excceded upper bound cost (error case)
        message = (
            "Train succeeded"
            if converged
            else f"Train failed to converge within max_epoch({job.max_epochs})"
        )
        current_meausurement = MeasurementOfBs(
            job_id=result.job_id,
            batch_size=result.batch_size,
            time=result.time,
            energy=result.energy,
            converged=converged,
        )

        if job.stage == Stage.MAB:
            # We are in MAB stage
            # Since we're learning the reward precision, we need to
            # 1. re-compute the precision of this arm based on the reward history,
            # 2. update the arm's reward precision
            # 3. and `fit` the new MAB instance on all the reward history.
            # Note that `arm_rewards` always has more than one entry (and hence a
            # non-zero variance) because we've been through pruning exploration.

            await self.mab.report(job, current_meausurement)
        else:
            logger.info(
                f"{result.job_id} in pruning stage, Current BS {result.batch_size} that did {'not ' * (not converged)}converge."
            )

            await self.pruning_manager.report_batch_size_result(
                current_meausurement, reported_cost < cost_ub, reported_cost
            )

        if cost_ub < reported_cost:
            message = f"""Batch Size({result.batch_size}) exceeded the cost upper bound: current cost({reported_cost}) >
                    beta_knob({job.beta_knob})*min_cost({job.min_cost})"""

        return ReportResponse(stop_train=True, converged=converged, message=message)


## End of class ZeusBatchSizeOptimizer
