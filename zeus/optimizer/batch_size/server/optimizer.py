"""Batch size optimizer top-most layer that provides register/report/predict."""

from __future__ import annotations
import hashlib
import time

import numpy as np
from zeus.optimizer.batch_size.common import (
    JobSpecFromClient,
    TrialId,
    ReportResponse,
    TrainingResult,
)
from zeus.optimizer.batch_size.server.batch_size_state.commands import (
    CreateMabTrial,
    ReadTrial,
    UpdateTrial,
)
from zeus.optimizer.batch_size.server.database.schema import TrialStatus
from zeus.optimizer.batch_size.server.exceptions import (
    ZeusBSOJobConfigMismatchError,
    ZeusBSOServerNotFoundError,
    ZeusBSOServiceBadOperationError,
    ZeusBSOValueError,
)
from zeus.optimizer.batch_size.server.explorer import PruningExploreManager
from zeus.optimizer.batch_size.server.job.commands import CreateJob
from zeus.optimizer.batch_size.server.job.models import Stage
from zeus.optimizer.batch_size.server.mab import GaussianTS
from zeus.optimizer.batch_size.server.services.service import ZeusService
from zeus.utils.logging import get_logger
from zeus.utils.metric import zeus_cost

logger = get_logger(__name__)


class ZeusBatchSizeOptimizer:
    """Batch size optimizer server. Manages which stage the job is in and call corresponding manager (pruning or mab)."""

    def __init__(self, service: ZeusService) -> None:
        """Initialize the server. Set the service, pruning manager, and mab.

        Args:
            service: ZeusService for interacting with database
        """
        self.service = service
        self.pruning_manager = PruningExploreManager(service)
        self.mab = GaussianTS(service)

    async def register_job(self, job: JobSpecFromClient) -> bool:
        """Register a job that user submitted. If the job id already exists, check if it is identical with previously registered configuration.

        Args:
            job: job configuration

        Returns:
            True if a job is regiested, False if a job already exists and identical with previous configuration

        Raises:
            [`ZeusBSOJobConfigMismatchError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOJobConfigMismatchError]: In the case of existing job, if job configuration doesn't match with previously registered config
        """
        registered_job = None

        if job.job_id is None:
            while True:
                job.job_id = f"{job.job_id_prefix}-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
                if (await self.service.get_job(job.job_id)) is None:
                    break
        else:
            registered_job = await self.service.get_job(job.job_id)

        if registered_job is not None:
            # Job exists
            logger.info("Job(%s) already exists", job.job_id)
            registerd_job_config = JobSpecFromClient.parse_obj(registered_job.dict())

            # check if it is identical
            if registerd_job_config != job:
                raise ZeusBSOJobConfigMismatchError(
                    "JobSpec doesn't match with existing jobSpec. Use a new job_id for different configuration"
                )
            return False

        self.service.create_job(CreateJob.from_job_config(job))
        logger.info("Registered %s", job.job_id)

        return True

    async def predict(self, job_id: str) -> TrialId:
        """Return a batch size to use.

        Args:
            job_id: Id of job

        Returns:
            batch size to use

        Raises:
            [`ZeusBSOValueError`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOValueError]: If the job id is unknown, or creating a mab failed due to no converged batch size
        """
        job = await self.service.get_job(job_id)

        if job is None:
            raise ZeusBSOValueError(
                f"Unknown job({job_id}). Please register the job first"
            )

        if job.stage == Stage.MAB:
            # If we are in MAB stage, use mab to get the next batch size
            arms = await self.service.get_arms(job_id)
            next_trial = await self.service.create_trial(
                CreateMabTrial(
                    job_id=job_id,
                    batch_size=self.mab.predict(
                        job_id, job.mab_prior_precision, job.mab_num_explorations, arms
                    ),
                )
            )
        else:
            # Pruning stage
            explorations = await self.service.get_explorations_of_job(job_id)
            # First check if pruning explorer can give us any batch size. Returns batch_size or MAB to indicate going to MAB stage
            res = await self.pruning_manager.next_batch_size(job, explorations)

            if isinstance(res, list):
                # MAB stage: construct MAB and update the job stage to MAB. Return the batch size from MAB
                logger.info("Constructing a MAB")
                arms = await self.mab.construct_mab(job, explorations, res)
                next_trial = await self.service.create_trial(
                    CreateMabTrial(
                        job_id=job_id,
                        batch_size=self.mab.predict(
                            job_id,
                            job.mab_prior_precision,
                            job.mab_num_explorations,
                            arms,
                        ),
                    )
                )
            else:
                next_trial = res

        return TrialId(
            job_id=next_trial.job_id,
            batch_size=next_trial.batch_size,
            trial_number=next_trial.trial_number,
        )

    async def report(self, result: TrainingResult) -> ReportResponse:
        """Report the training result. Stop train if the train is converged or reached max epochs or reached early stop threshold. Otherwise, keep training.

        Args:
            result: result of training [`TrainingResult`][zeus.optimizer.batch_size.common.TrainingResult].

        Returns:
            Decision on training [`ReportResponse`][zeus.optimizer.batch_size.common.ReportResponse].
        """
        cost_ub = np.inf
        job = await self.service.get_job(result.job_id)
        if job is None:
            raise ZeusBSOServiceBadOperationError(f"Unknown job {result.job_id}")

        trial = await self.service.get_trial(
            ReadTrial(
                job_id=result.job_id,
                batch_size=result.batch_size,
                trial_number=result.trial_number,
            )
        )
        if trial is None:
            raise ZeusBSOServiceBadOperationError(f"Unknown trial {result}")

        if trial.status != TrialStatus.Dispatched:
            # result is already reported
            if trial.converged is None:
                raise ZeusBSOValueError(
                    f"Trial({trial.trial_number}) is already reported but converged is not set."
                )
            return ReportResponse(
                stop_train=True,
                converged=trial.converged,
                message=f"Result for this trial({trial.trial_number}) is already reported.",
            )

        if job.beta_knob is not None and job.min_cost is not None:  # Early stop enabled
            cost_ub = job.beta_knob * job.min_cost

        reported_cost = zeus_cost(
            result.energy,
            result.time,
            job.eta_knob,
            job.max_power,
        )

        within_cost_range = cost_ub >= reported_cost
        converged = (
            job.higher_is_better_metric and job.target_metric <= result.metric
        ) or (not job.higher_is_better_metric and job.target_metric >= result.metric)

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

        trial_result = UpdateTrial(
            job_id=result.job_id,
            batch_size=result.batch_size,
            status=TrialStatus.Succeeded,
            trial_number=result.trial_number,
            time=result.time,
            energy=result.energy,
            converged=converged and within_cost_range,
        )

        if job.stage == Stage.MAB:
            await self.mab.report(job, trial_result)
        else:
            # Pruning stage
            logger.info(
                "%s in pruning stage, Current BS %s that did %s converge.",
                result.job_id,
                result.batch_size,
                "not" * (not converged),
            )
            # update trial
            self.service.update_trial(trial_result)

        assert trial_result.converged is not None, "This just set to boolean above."
        return ReportResponse(
            stop_train=True, converged=trial_result.converged, message=message
        )

    async def end_trial(self, trial_id: TrialId) -> None:
        """Mark the trial as finished. If status is still `Dispatched` make the trial as `Failed`.

        Args:
            trial_id: Unique identifier of trial

        Raises:
            [`ZeusBSOServerNotFound`][zeus.optimizer.batch_size.server.exceptions.ZeusBSOServerNotFound]: If there is no corresponding trial.
        """
        trial = await self.service.get_trial(ReadTrial(**trial_id.dict()))

        if trial is not None:
            if trial.status == TrialStatus.Dispatched:
                self.service.update_trial(
                    UpdateTrial(
                        job_id=trial_id.job_id,
                        batch_size=trial_id.batch_size,
                        trial_number=trial_id.trial_number,
                        status=TrialStatus.Failed,
                    )
                )
        else:
            raise ZeusBSOServerNotFoundError(f"Could not find the trial: {trial_id}")

    async def delete_job(self, job_id: str) -> None:
        """Delete a job.

        Args:
            job_id: ID of a job.

        Returns:
            True if the job is deleted. False if none was deleted
        """
        if not (await self.service.delete_job(job_id)):
            raise ZeusBSOServerNotFoundError("No job was deleted.")
