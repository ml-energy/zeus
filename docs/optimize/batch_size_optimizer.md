# Batch Size Optimizer

The batch size optimizer (BSO) finds the optimal DNN training batch size that minimizes *cost*:

$$
\eta \cdot \textrm{ETA} + (1 - \eta) \cdot \textrm{MaxPower} \cdot \textrm{TTA}
$$

where ETA and TTA stands for Energy-to-Accuracy and Time-to-Accuracy, respectively, and accuracy is the target *validation* metric of training.
Users can trade-off between energy and time by setting the $\eta$ parameter to be between 0 and 1.


## Usage

In production environments, it is common for a single DNN to be trained and re-trained many times over time.
This is because the data distribution changes over time, and the model needs to be re-trained to adapt to the new data distribution.
The batch size optimizer uses these **recurring training jobs** as opportunities to find the optimal batch size for the DNN training job.
To do this, the batch size optimizer uses a Multi-Armed Bandit algorithm to explore the batch size space.
For more details, please refer to the [Zeus paper](../research_overview/zeus.md).

!!! Important "Constraints"
    Currently, the batch size optimizer only supports cases where the **number and type of GPUs** used for each recurrent training job is always the same.


## High-level architecture

```mermaid
sequenceDiagram;
    participant BSO server
    participant BSO client
    loop Every recurrent training
        BSO client->>BSO server: Register the training job and ask for the batch size to use
        BSO server->>BSO client: Return the batch size to use and an integer trial number
        loop Every epoch
            BSO client->>BSO server: At the end of each epoch, report time and energy consumption
            BSO server->>BSO client: Compute cost and determine whether the client should terminate training
        end
        BSO client->>BSO server: Notify the server that the trial finished
    end
```

In order to persist the state of the optimizer across all recurring training runs, we need to have a server that outlives individual training jobs.
Therefore, the batch size optimizer consists of two parts: the server and the client.
The server is a FastAPI server that manages the Multi-Armed Bandit algorithm and the database, while the client ([`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer]) is integrated into the training script.


## Deploying the server

Largely three steps: (1) starting the database, (2) running migration on the database, and (3) starting the BSO server.

### Clone the Zeus repository

To get example docker files and database migration scripts, clone the Zeus repository.

```sh
git clone https://github.com/ml-energy/zeus.git
```

### Decide on the database

By default, our examples use MySQL.
However, you can use any database [supported by SQLAlchemy](https://docs.sqlalchemy.org/en/latest/dialects/){.external}.
Please make sure the database's corresponding async connection driver (e.g., `asyncmy` for MySQL, `aiosqlite` for SQLite) is installed.
For instance, to adapt our examples, you can (1) add `pip install` to [`migration.Dockerfile`](https://github.com/ml-energy/zeus/blob/master/docker/batch_size_optimizer/migration.Dockerfile) and [`server.Dockerfile`](https://github.com/ml-energy/zeus/blob/master/docker/batch_size_optimizer/server.Dockerfile), and (2) change the `db` container specification in [`server-docker-compose.yaml`](https://github.com/ml-energy/zeus/blob/master/docker/batch_size_optimizer/server-docker-compose.yaml).

### Server configuration

You can configure the server using enviornment variables or the `.env` file.
Below are the complete list of environment variables you can set and example values.

```sh
ZEUS_BSO_DB_USER="me" 
ZEUS_BSO_DB_PASSWORD="secret"
ZEUS_BSO_DATABASE_URL="mysql+asyncmy://me:secret@localhost:3306/Zeus"
ZEUS_BSO_ROOT_PASSWORD="secret*"
ZEUS_BSO_SERVER_PORT=8000
ZEUS_BSO_LOG_LEVEL="INFO"
ZEUS_BSO_ECHO_SQL="True"
```

### With Docker Compose

```Shell
cd docker/batch_size_optimizer
docker-compose -f ./server-docker-compose.yaml up
```

Docker Compose will first build necessary images and spin up the containers.

### With Kubernetes

1. Build the Docker image.

    ```sh
    # From the repository root
    docker build -f ./docker/batch_size_optimizer/server.Dockerfile -t bso-server . 
    docker build -f ./docker/batch_size_optimizer/migration.Dockerfile -t bso-migration .
    ```

    Make sure Kubernetes has access to these build images.
    If you are locally using `minikube`, then the images are already available.
    However, if you are using the cloud such as AWS EKS, you should push the image to the registry and modify the image path in `server-docker-compose.yaml` to allow Kubernetes to pull the image.

2. Convert Docker Compose files to Kubernetes YAML files using [Kompose](https://kompose.io/){.external}.

    ```sh
    cd docker/batch_size_optimizer
    docker-compose -f server-docker-compose.yaml config > server-docker-compose-resolved.yaml
    kompose convert -f server-docker-compose-resolved.yaml -o ./kube/
    rm server-docker-compose-resolved.yaml
    ```

    This first resolves env files using `docker-compose`, then converts it into Kubernetes YAML files in `./kube/`.

3. Apply the Kubernetes YAML files.

    ```sh
    cd kube
    kubectl apply -f .
    ```

### With just Python

You can also run the server without Docker or Kubernetes.

1. Spin up the database of your choice.
2. Set server configuration environment variables described [here](#server-configuration).
3. Perform DB migration with Alembic.

    1. Install dependencies

        ```sh
        # From the repository root
        pip install asyncmy cryptography
        pip install '.[migration]'
        ```

    2. Create the migration script. This will create scripts in `./versions`.

        ```sh
        cd zeus/optimizer/batch_size
        alembic revision --autogenerate -m "Create tables" 
        ```

    3. Apply migration, either online or offline.

        ```sh
        # Online migration (applied directly to the database)
        alembic upgrade head 
        # Offline migration (just generate SQL)
        alembic upgrade head --sql
        ```

4. Spin up the server using `uvicorn`.

    ```sh
    uvicorn zeus.optimizer.batch_size.server.router:app
    ```

## Integrating [`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer]

In order for your recurring training job to communicate with the BSO server, you need to integrate the [`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer] class into your training script.

1. Install the Zeus package, including dependencies needed for the batch size optimizer.

    ```sh
    pip install zeus[bso]
    ```

2. Integrate [`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer] to your training script.

    ```python hl_lines="19-20 25-26 33-35 37-39"
    from zeus.monitor import ZeusMonitor
    from zeus.optimizer.batch_size import BatchSizeOptimizer, JobSpec

    monitor = ZeusMonitor()

    # On instantiation, the BSO will register the job to the BSO server
    bso = BatchSizeOptimizer(
        monitor=monitor,
        server_url="http://bso-server:8000",
        job=JobSpec(
            job_id=os.environ.get("ZEUS_JOB_ID"),
            job_id_prefix="mnist",
            default_batch_size=256,
            batch_sizes=[32, 64, 256, 512, 1024, 4096, 2048],
            max_epochs=100
        ),
    )

    # Get batch size to use from the server
    batch_size = bso.get_batch_size()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    # Start measuring the time and energy consumption of this training run
    bso.on_train_begin()

    for epoch in range(100):
        for batch in train_dataloader:
            # Training loop
            pass

        # The BSO server needs to know whether training has converged
        metric = evaluate(model, eval_dataloader)
        bso.on_evaluate(metric)

        # The BSO server will determine whether to stop training
        if bso.training_finished:
            break
    ```

### When does training stop?

The BSO server will determine whether to stop training and this will be reflected in the `training_finished` attribute of the [`BatchSizeOptimizer`][zeus.optimizer.batch_size.client.BatchSizeOptimizer] instance.

If the DNN reaches the target validation metric, training should stop.
However, training fails if

1. it failed to converge within the configured `JobSpec.max_epochs` ([reference][zeus.optimizer.batch_size.common.JobSpec]) epochs, or
2. its cost exceeded the early stopping threshold configured by `JobSpec.beta_knob`([reference][zeus.optimizer.batch_size.common.JobSpec]) .

In such failure cases, the optimizer will raise a [`ZeusBSOTrainFailError`][zeus.optimizer.batch_size.exceptions.ZeusBSOTrainFailError].
This means that the chosen batch size was not useful, and the BSO server will never try this batch size again.
The user should re-launch the training run in this case, and the BSO server will try another batch size.


## Integration examples

Two full examples are given for the batch size optimizer:

- [MNIST](https://github.com/ml-energy/zeus/tree/master/examples/batch_size_optimizer/mnist/){.external}: Single-GPU and data parallel training, with integration examples with Kubeflow
- [Sentiment Analysis](https://github.com/ml-energy/zeus/tree/master/examples/batch_size_optimizer/capriccio/){.external}: Full training example with HuggingFace transformers using the Capriccio dataset, a sentiment analysis dataset with data drift.
