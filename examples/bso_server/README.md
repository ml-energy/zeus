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

### Remark about server

Zeus Batch Size Optimizer server is using Sqlalchemy to support various type of database. However, you need to download the corresponding async connection driver.
As a default, we are using Mysql. You can add installation code to `Dockerfile.migration` and `Dockerfile.server`. Refer to those files for reference.

## Use BSO in your training script (Client)

1. Install Zeus package.

    ```Shell
    pip install zeus-ml[bso]
    ```

2. Add `ZeusBatchSizeOptimizer` to your training script.

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

### Remark about client

Training can fail if

1. It failed to converge within configured max_epochs
2. It exceeded the early stopping threshold which is configured by `beta_knob` in `JobSpec`

In that case, optimizer will raise `ZeusBSOTrainFailError`. This means that chosen batch size was not useful, and bso server will not give this batch size again. However, the user ***should re-lanuch the job*** so that the bso server can give another batch size. The server will learn which batch size is useful and will converge to the batch size that causes the least cost as you lanch the job multiple times.

## Kubernetes

You can convert `docker-compose` file into kubernetes yaml file using `kompose`.

```Shell
docker-compose config > docker-compose-resolved.yaml && kompose convert -f docker-compose-resolved.yaml && rm docker-compose-resolved.yaml
```

This command will resolve the `.env` file and create a resolved version of docker-compose yaml file. Then, you can convert it to kubernetes yaml file using `kompose convert`. Refer [Kompose](https://kompose.io/) and [Kompose labels](https://github.com/kubernetes/kompose/blob/main/docs/user-guide.md) for more information.

