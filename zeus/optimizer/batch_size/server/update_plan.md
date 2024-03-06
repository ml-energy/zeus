Pydantic model -> DB operation -> result pydantic model.

class BatchSize(Base):
    job_id: Mapped[UUID] = mapped_column(ForeignKey("Job.job_id"), primary_key=True)
    batch_size: Mapped[int] = mapped_column(Integer, primary_key=True)
    explorations: Mapped[list["ExplorationState"]] = relationship(
        order_by="ExplorationState.trial_number.asc()",
    )
    measurements: Mapped[list["Measurement"]] = relationship(
        order_by="Measurement.timestamp.asc()"
    )
    arm_state: Mapped[Optional["GaussianTsArmState"]] = relationship(
        backref="BatchSize",
    )

get_measurements_of_bs
get_job_with_explorations
add_exploration
update_exploration

add_arms
update_arm_state

add_measurement

Consistency:
For each batch_size:
    explorations,
    measurements,
    arm_state
Should be consistent

Unit of Work

Updating Exploration:

1. Fetch All Explorations
    -> Figure out which stage we are In
2. Update exploration

- Create

1. Fetch job with batch_sizes (list[int]) *
    1.a) if job exists -> check equality
    1.b) if job doesn't exist -> create*

- Predict

1. Fetch Job & all explorations *
    1.a) Concurrent job
        RETURN best_bs from explorations
    1.b) Pruning ongoing
        1.b.1) Going to next round
            -> Update exp_default_bs*
        1.b.2) Stay in current round
        -> ADD exploration *
        RETURN
    1.c) Pruning over (MAB stage)
        1.c.a) MAB haven't constructed (Switch to Pruning to MAB)
            -> Fetch windowed measurements per bs*
            -> add arms *
        1.c.b) MAB already constructed
            -> Fetch arm states*
            RETURN

- Report

1. Fetch Job & all explorations, and min_cost (might be all measurements) *
    1-a) MAB
        -> Fetch windowed measurments*
        -> Update arm state for reported batch size *
    1-b) Pruning
        -> update_exploration*
    1-c) concurrent job (no exp.state == Exploring but Mab is not constructed)
        NONE

    ADD MEASUREMENT *
    Update min_cost*

If we add stage, Now do we detect transition? -> when we call predict()

======

- Create

1. Fetch job with batch_sizes (list[int]) *
    1.a) if job exists -> check equality
    1.b) if job doesn't exist -> create*

- Predict

1. Fetch Job *
    1.a) Pruning Stage
        -> Fetch all explorations*
        1.b.1) Going to next round
            1.b.1.1) Exceed num_pruning, Go to MAB STAGE # STAGE CHANGE
                -> Fetch windowed measurements per bs *
                -> add arms*
                -> UPDATE STAGE TO MAB from Pruning *
            1.b.1.2) Proceed next round
                -> MIGHT Update exp_default_bs*
                -> ADD exploration *
        1.b.2) Explorting in current round
            -> ADD exploration*
        1.b.3) Pending exploration exists -> Concurrent job
            return min_bs

    1.b) MAB
        -> Fetch arm states *
        -> Update mab_rng if it is needed*

- Report

1. Fetch Job *(including min_cost)
    1-a) MAB
        -> Fetch windowed measurments PER BS*
        -> ADD MEASUREMENT *
        -> Update arm state for reported batch size*
    1-b) Pruning
        -> Fetch all explorations *
        1-b.1) Have a bs that is in "Exploring" state
            -> ADD MEASUREMENT*
            -> update_exploration *
        1-b.2) concurrent job (no exp.state == Exploring but Mab is not constructed)
            -> ADD MEASUREMENT*

    Update min_cost *

I have been thinking and this is the update plan.
Our use case:

- Fetching explorations always happen in the job level (for this job_id fetch all explorations)
- Fetching measurements is per batch size level since we have to fetch window_size amount of measurements per bs
- Updating/adding Exploration state, GaussianTs state, measurements is in Batch Size level -> For that (job_id, bs), do these operations
- Always, updating exploration states or GaussianTs states should come with adding measurement
  => For each batch size, exp states, measurements, and GaussianTs state should be consistent.

Modification in schema:
    Add Stage & min_cost to Job table
        Stage = Pruning OR MAB
            If we record stage transition, no need to fetch explorations all the time to figure out which stage we are in.
        min_cost observed so far
            No need to fetch all measurements everytime we call Report.

At the end two repos

1. JobRepository
    -> Just related to jobSpec stuff. Will fetch batchSizes together but not their states only the number itself.
    This repo will support
        1. fetching explorations of that job. (was debating if this should be in BatchSizeRepo, but since the arguemtn is job_id, seems like more suitable here)
        2. Fetch Job instance
        3. create job
2. BatchSizeRepository
    -> Manage State of each batch size(job_id, batch_size pair)
    Model will look like

    BatchSizeModel:
        job_id: UUID
        batch_size: int

        -- states of this bs --
        explorations: list[Measurement]
        measurements: list[ExplorationState]
        mab_state: GaussianTsState

    This repo will support
        1. add exploration
        2. update exploration
        3. add_arm
        4. update arm state
        5. add measurement
        6. fetch measurements of bs
