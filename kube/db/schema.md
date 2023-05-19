# DB Schema

## Jobs

- Records all the jobs submitted by users
- Append-only
  - A job is inserted by ZeusServer after a user submits it to ZeusServer.
  - The "status" of a job is changed from "Running" to "Completed" after terminated by the user.

| job_id | user_id | seed | default_batch_size | min_batch_size | max_batch_size | eta_knob | beta_knob | target_metric | max_epochs | num_recurrence | max_retries |   phase   |
| ------ | ------- | ---- | ---- | ------ | ------ | -------- | --------- | ------------- | ---------- | ------------ | ----------- | --------- |
| 000001 | 000000a |   1  | 1024 |    8   |  4096  |    0.5   |    2.0    |      0.50     |     100    |     100     |     20      |  Running  |

## Trials

- Records all trials created
- Append-only
  - A trial is inserted by ZeusServer after its completion.
- This table contains the same information as `train_json`.

| job_id | rec_i | try_i |  batch_size  | time | energy | cost | num_epochs | reached | phase |
| ------ | ----- | ----- | ---- | ---- | ------ | ---- | ---------- | ------- | ------ |
| 000001 |   1   |   1   | 1024 | 508.696199872531 | 117868.43460299837 | 135238.64728237884 | 28 | true | Running |


## Profiling

- For each job, records the following mappings:
  - Train (one record for each `power_limit`)
    - `power_limit` -> `train_avg_power`
    - `power_limit` -> `train_tput`
  - Eval (only one record for `opt_power_limit`)
    - `opt_power_limit` -> `eval_avg_power`
    - `opt_power_limit` -> `eval_tput`

- Append-only
  - For each job, inserted at the first *end of epoch* when profiling is done.

- This table contains the same information as `power_json`.


| job_id |  batch_size  | phase |   pl   | vtype |       value        |
| ------ | ------------ | ----- | ------ | ----- | ------------------ |
| 000001 |      32      | train | 300000 | power | 131.93493277891338 |
| 000001 |      32      | train | 275000 | power | 123.66380334160725 |
| 000001 |      32      | train | 300000 | tput  | 31.03646417467191  |
| 000001 |      32      | train | 275000 | tput  | 29.93935643421058  |
| 000001 |      32      | eval  | 175000 | power | 125.63629920513313 |
| 000001 |      32      | eval  | 175000 | tput  | 114.86617394848754 |

(Example job CIFAR100 with ShuffleNet)
