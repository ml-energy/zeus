# Batch Size Optimizer in Zeus

## What is it?

Batch size optimizer(BSO) can choose the best batch size that minimizes the cost, where cost is defined as $cost = \eta \times \text{Energy consumption to accuracy} + (1-\eta) \times \text{Max power}\times \text{Time to accuracy}$.

## How does it work?

Core of BSO is a multi-arm-bandit based on **recurrent** trainings. After each training, we feed the result cost to MAB and after certain number of trainings, MAB can converge to the best batch size. In addition to MAB, we employed early-stopping and pruning to handle stragglers. For more detail, refer to [paper](https://www.usenix.org/conference/nsdi23/presentation/you).

## Should I use this?

The key of BSO is recurrent training. If you are training your model periodically or repeatedly, BSO can be a great choice to reduce energy or time consumption.

## Limitations

We currently doesn't support heterogeneous GPUs or different configurations. Number of GPUs, gpu models, and other configurations in JobSpec should be identical in recurrent trainings. If you are running your training in a various environment each time, then it might not desirable to use BSO.

## Quick start (Server)

1. Clone the repository

    ```Shell
    git clone https://github.com/ml-energy/zeus/tree/master
    ```

2. Create `.env` under `/docker`. Example of `.env` is provided below.

    For default, we are using the MySQL for the database.

    ```Shell
    ZEUS_BSO_DB_USER="me" 
    ZEUS_BSO_DB_PASSWORD="secret"
    ZEUS_BSO_ROOT_PASSWORD="secret*"
    ZEUS_BSO_SERVER_PORT=8000
    ZEUS_BSO_LOG_LEVEL = "INFO"
    ZEUS_BSO_ECHO_SQL = "True"
    ```

    If you want to use different databases, you need to add `ZEUS_BSO_DATABASE_URL` as an environment variable. See [Remark](#remark-about-server) for detail.
    Also, if you are running using docker-compose or kubernetes, you need to change the image name under `db` in docker-compose file.

3. Running a server
    - Using docker-compose

        ```Shell
        cd docker 
        docker-compose -f ./docker/docker-compose.yaml up
        ```

        This will build images for each container: db, migration and the server. Then, it will spin those containers.

    - Using Kubernetes.
        1. Build an image.

            ```Shell
            # From the root directory
            build -f ./docker/server.Dockerfile -t bso-server . 
            build -f ./docker/migration.Dockerfile -t bso-server .
            ```

        2. Create kubernetes yaml files using Kompose.

            ```Shell
            cd docker 
            docker-compose config > docker-compose-resolved.yaml && kompose convert -f docker-compose-resolved.yaml -o ./kube/ && rm docker-compose-resolved.yaml
            ```

            It first resolves env files using docker-compose, then create kubernetes yaml files under `docker/kube/`

        3. Run kubernetes.

            ```Shell
            cd kube
            kubectl apply -f .
            ```

    - Using uvicorn.

        If you are using the uvicorn to spin the server, you need to create database and perform migration before starting the server.

        1. Run the database of your choice.
        2. Set the environment variables in `.env`

            ```Shell
            ZEUS_BSO_DATABASE_URL="me" 
            ZEUS_BSO_LOG_LEVEL = "INFO"
            ZEUS_BSO_ECHO_SQL = "True"
            ```

        3. Run Alembic migration by following the guides in `zeus/optimizer/batch_size/migrations/README.md`

        4. Run server using uvicorn.

            ```Shell
            cd zeus/optimizer/batch_size/server
            uvicorn router:app --reload 
            ```

Now server is good to go!

### Remark about the server

Zeus Batch Size Optimizer server is using Sqlalchemy to support various type of database. However, you need to download the corresponding async connection driver.
As a default, we are using Mysql. You can add installation code to `migration.Dockerfile` and `server.Dockerfile`. Refer to those files for reference.

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
        server_url="<server url>", # ex) http://127.0.0.1:8000
        job=JobParams(
            job_id_prefix="mnist-dev",
            default_batch_size=256,
            batch_sizes=[32, 64, 256, 512, 1024, 4096, 2048],
            max_epochs=100
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

## Kompose references

Refer [Kompose](https://kompose.io/) and [Kompose labels](https://github.com/kubernetes/kompose/blob/main/docs/user-guide.md) for more information.
