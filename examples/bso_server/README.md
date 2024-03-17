# Batch Size Optimizer in Zeus

## What it is?

Batch size optimizer(BSO) can choose the best batch size that minimizes the cost, where cost is defined as $cost = \eta \times \text{Energy consumption to accuracy} + (1-\eta) \times \text{Max power}\times \text{Time to accuracy}$.

## How does it work?

Core of BSO is a multi-arm-bandit based on **recurrent** trainings. After each training, we feed the result cost to MAB and after certain number of trainings, MAB can converge to the best batch size. In addition to MAB, we employed early-stopping and pruning to handle stragglers. For more detail, refer to [paper](https://www.usenix.org/conference/nsdi23/presentation/you).

## Should I use this?

The key of BSO is recurrent training. If you are training your model periodically or repeatedly, BSO can be a great choice to reduce energy or time consumption.

## Limitations

We currently doesn't support heterogeneous GPUs or different configurations. Number of GPUs, gpu models, and other configurations in JobSpec should be identical in recurrent trainings. If you are running your training in a various environment each time, then it might not desirable to use BSO.

## How to use?

Batch size optimzer is composed of two parts: server and client. Client will be running in your training script just like power limit optimizer or monitor. This client will send training result to BSO server and server will give the client the best batch size to use.

### Running a BSO server

1. Install zeus batch size optimizer dependencies

    ```shell
    pip install zeus-ml[bso]
    ```

2. Install additional dependencies for a chosen database
3. Install migration dependencies to use alembic for migration
4. Follow readme under migration to set up tables

5. Set up environment. Should specify, database_url, and database_password. For example,

    ```shell
    # Example 
    ZEUS_DATABASE_URL = "sqlite+aiosqlite:////home/Projects/zeus/test.db"
    ZEUS_DATABASE_PASSWORD = "SECRET"
    ZEUS_LOG_LEVEL = "INFO"
    ```

### Running

1. Run server

    ```shell
    # /zeus/zeus/optimizer/batch_size/server
    uvicorn router:app 
    ```

2. Run mnist example

    ```shell
    # /zeus/examples/bso_server
    python mnist.py
    ```

### Using a BSO client

In your training script, initialize batch size optimizer like below.

```python
bso = BatchSizeOptimizer(
    monitor=monitor,
    server_url="http://127.0.0.1:8000",
    job_in=JobSpecIn(job_id="3fa85f64-5717-4562-b3fc-2c963f66afa6"),
)
```
