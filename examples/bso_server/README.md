# Batch Size Optimizer in Zeus

Batch size optimzer is composed of two parts: server and client. Client will be running in your training script just like power limit optimizer or monitor. This client will send training result to BSO server and server will give the client the best batch size to use.

## Quick start (Server)

1. Clone the repository

    ```Shell
    git clone https://github.com/ml-energy/zeus/tree/master
    ```

2. Create `.env` under `/docker`. Example of `.env` is provided below.

    ```Shell
    ZEUS_BSO_DB_USER="me" 
    ZEUS_BSO_DB_PASSWORD="secret"
    ZEUS_BSO_ROOT_PASSWORD="secret*"
    ZEUS_BSO_SERVER_PORT=8000
    ```

3. Spawn containers. It will create database, perform migration, and spawn the server.

    ```Shell
    docker-compose -f ./docker/docker-compose.yaml up
    ```

Now server is good to go!

### Remark

Zeus Batch Size Optimizer server is using Sqlalchemy to support various type of database. However, you need to download the corresponding async connection driver.
As a default, we are using Mysql. You can add installation code to `Dockerfile.migration` and `Dockerfile.server`. Refer to those files for reference.

## Use BSO in your training script (Client)

1. Add `ZeusBatchSizeOptimizer` to your training script.

    ```Python
    # Initialization
    bso = BatchSizeOptimizer(
        monitor=monitor,
        server_url="<server url>", # http://127.0.0.1:8000
        job=JobParams(
            job_id_prefix="mnist-dev",
            default_batch_size=256,
            batch_sizes=[32, 64, 256, 512, 1024, 4096, 2048],
        ),
    )
    # ... other codes 

    # Get batch size to use from the server
    batch_size = bso.get_batch_size()

    # ... 

    # beginning of the train
    bso.on_train_begin()

    # ...

    # After evaluation
    bso.on_evaluate(metric)
    ```

### Remark

    TODO: ADD STUFF ABOUT JOB_ID
