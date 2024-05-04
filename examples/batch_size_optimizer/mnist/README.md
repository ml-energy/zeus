# MNIST training + BSO

With the simplest DNN training workload, we show:

- Single GPU training job integrated with the `GlobalPowerLimitOptimizer` and `BatchSizeOptimizer` (`train_single_gpu.py`).
- Multi-GPU data parallel training job integrated with the `GlobalPowerLimitOptimizer` and `BatchSizeOptimizer` (`train_dp.py`).
- How to integrate either of them with Kubeflow on Kubernetes.

## Training

First, set up the batch size optimizer server following the [docs](https://ml.energy/zeus/optimizers/batch_size_optimizer/).
On the first recurrence of the job, the batch size optimizer will register the job to the server and print out the job ID.
From the second recurrence, set the `ZEUS_JOB_ID` environment variable to allow the recurrence to be recognized as part of the recurring job.

## Kubeflow

Kubeflow is a tool to easily deploy your ML workflows to Kubernetes. We provide some examples of using Kubeflow with Zeus. In order to run your training in Kubeflow with Zeus, follow the `docs/batch_size_optimizer/server.md` to deploy batch size optimizer to Kubernetes. After then, you can deploy your training script using Kubeflow.

1. Set up Kubernetes and install Kubeflow training operator.

    Refer to [minikube](https://minikube.sigs.k8s.io/docs/start/) for local development of Kubernetes.
    Refer to [Kubeflow training operator](https://github.com/kubeflow/training-operator) to how to install Kubeflow.

2. Run server batch size optimizer server using Kubernetes.

    Refer docs to start the server [Quick start](../../docs/batch_size_optimizer/index.md).

3. Build MNIST example docker image.

    ```Shell
    # From project root directory
    docker build -f ./examples/batch_size_optimizer/mnist.Dockerfile -t mnist-example . 
    ```

    If you are using the cloud such as AWS, modify the `image` and `imagePullPolicy` in `train_dp.yaml` to pull it from the corresponding registry.

4. Deploy training script.

    ```Shell
    cd examples/batch_size_optimizer
    kubectl apply -f train_dp.yaml # For distributed training example
    kubectl apply -f train_single_gpu.yaml # For single gpu training example
    ```

## Data parallel training with Zeus

In the case of data parallel training, Batch size optimizer should be able to give the consistent batch size to all GPUs. Since there is no way for batch size to tell the differences between concurrent job submissions and multiple GPU training, we ask users to send a request from a single GPU and broadcast the result (batch size, trial number) to other GPUs. In the case of reporting the result to the batch size optimizer server and receiving the corresponding result (train fail or succeeded) can be dealt by the server since it has the `trial_number`. Thus, report doesn't require any broadcast or communications with other GPUs.
Refer to the `examples/batch_size_optimizer/mnist/train_dp.py` for the use case.
