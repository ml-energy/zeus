from __future__ import annotations

from uuid import UUID

import numpy as np
from zeus.optimizer.batch_size.common import JobConfig, ReportResponse, TrainingResult
from zeus.optimizer.batch_size.server.batch_size_state.models import MeasurementOfBs
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOJobConfigMismatchError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.optimizer.batch_size.server.job.commands import CreateJob
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.services.commands import CompletedExplorations
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.util.logging import get_logger
from zeus.util.metric import zeus_cost

logger = get_logger(__name__)


class ZeusBatchSizeOptimizer:
    """Batch size optimizer server. Manages which stage the job is in and call corresponding manager (pruning or mab)."""

    def __init__(self, service: ZeusService) -> None:
        """Initialize the server. Set the service, pruning manager, and mab

        Args:
            service: ZeusService for interacting with database
        """
        self.service = service
        self.pruning_manager = PruningExploreManager(service)
        self.mab = GaussianTS(service)
        self.name = "ZeusBatchSizeOptimizer"

    async def register_job(self, job: JobConfig) -> bool:
        """Register a job that user submitted. If the job id already exists, check if it is identical with previously registered configuration

        Args:
            job: job configuration

        Return:
            True if a job is regiested, False if a job already exists and identical with previous configuration

        Raises:
            `ZeusBSOJobConfigMismatchError`: In the case of existing job, if job configuration doesn't match with previously registered config
        """
        registered_job = await self.service.get_job(job.job_id)

        if registered_job is not None:
            # Job exists
            logger.info(f"Job({job.job_id}) already exists")
            registerd_job_spec = JobConfig.parse_obj(registered_job.dict())

            # check if it is identical
            if registerd_job_spec != job:
                raise ZeusBSOJobConfigMismatchError(
                    "JobSpec doesn't match with existing jobSpec. Use a new job_id for different configuration"
                )
            return False

        self.service.create_job(CreateJob.from_jobSpec(job))
        logger.info(f"Registered {job.job_id}")

        return True

    async def predict(self, job_id: UUID) -> int:
        """Return a batch size to use.

        Args:
            job_id: Id of job

        Return:
            batch size to use

        Raises:
            `ZeusBSOValueError`: If the job id is unknown, or creating a mab failed due to no converged batch size
        """
        job = await self.service.get_job(job_id)

        if job == None:
            raise ZeusBSOValueError(
                f"Unknown job({job_id}). Please register the job first"
            )

        if job.stage == Stage.MAB:
            # If we are in MAB stage, use mab to get the next batch size
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
                # Concurrent job submission during pruning stage. Return the best known batch size
                return job.min_batch_size
            elif isinstance(next_batch_size, Stage):
                # MAB stage: construct MAB and update the job stage to MAB. Return the batch size from MAB
                logger.info(
                    f"Constructing a MAB {explorations.explorations_per_bs == None}"
                )
                arms = await self.mab.construct_mab(
                    job,
                    CompletedExplorations(
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
        Report the training result. Stop train if the train is converged or reached max epochs or reached early stop threshold.
        Otherwise, keep training.

        Args:
            result: result of training [`TrainingResult`][zeus.optimizer.batch_size.common.TrainingResult].

        Return:
            Decision on training [`ReportResponse`][zeus.optimizer.batch_size.common.ReportResponse].
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

        within_cost_range = cost_ub >= reported_cost
        converged = (
            job.high_is_better_metric and job.target_metric <= result.metric
        ) or (not job.high_is_better_metric and job.target_metric >= result.metric)

        if (
            within_cost_range
            and result.current_epoch < job.max_epochs
            and not converged
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
        if converged and within_cost_range:
            message = "Train succeeded"
        elif not within_cost_range:
            message = f"""Batch Size({result.batch_size}) exceeded the cost upper bound: current cost({reported_cost}) >
                    beta_knob({job.beta_knob})*min_cost({job.min_cost})"""
        else:
            # not converged
            message = f"Train failed to converge within max_epoch({job.max_epochs})"

        current_meausurement = MeasurementOfBs(
            job_id=result.job_id,
            batch_size=result.batch_size,
            time=result.time,
            energy=result.energy,
            converged=converged and within_cost_range,
        )

        if job.stage == Stage.MAB:
            await self.mab.report(job, current_meausurement)
        else:
            # Pruning stage
            logger.info(
                f"{result.job_id} in pruning stage, Current BS {result.batch_size} that did {'not ' * (not converged)}converge."
            )

            await self.pruning_manager.report_batch_size_result(
                current_meausurement, reported_cost
            )

        return ReportResponse(
            stop_train=True, converged=current_meausurement.converged, message=message
        )


## End of class ZeusBatchSizeOptimizer
