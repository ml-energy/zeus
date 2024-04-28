# Batch Size Optimizer in Zeus

Batch size optimzer is composed of two parts: server and client. Client will be running in your training script just like power limit optimizer or monitor. This client will send training result to BSO server and server will give the client the best batch size to use. Refer to the `docs/batch_size_optimizer/server.md` for how to get started.

## Data parallel training with Zeus

In the case of data parallel training, Batch size optimizer should be able to give the consistent batch size to all gpus. Since there is no way for batch size to tell the differences between concurrent job submissions and multiple GPU training, we ask users to send a request from a single GPU and broadcast the result(batch size, trial number) to other GPUs. In the case of reporting the result to the batch size optimizer server and receiving the corresponding result (train fail or succeeded) can be dealt by the server since it has the `trial_number`. Thus, report doesn't require any broadcast or communications with other GPUs.
Refer to the `examples/batch_size_optimizer/mnist_dp.py` for the use case.

## Kubeflow

Kubeflow is a tool to easily deploy your ML workflows to kubernetes. We provides some examples of using kubeflow with Zeus. In order to run your training in Kubeflow with Zeus, follow the `docs/batch_size_optimizer/server.md` to deploy batch size optimizer to kubernetes. After then, you can deploy your training script using kubeflow.

1. Install kubeflow training operator.

    Refer [Kubeflow training operator](https://github.com/kubeflow/training-operator) to how to install kubeflow.

2. Build mnist example docker image.

    ```Shell
    # From project root directory
    docker build -f ./examples/batch_size_optimizer/mnist.Dockerfile -t mnist-example . 
    ```

3. Deploy training script.

    ```Shell
    kubectl apply -f mnist_dp.yaml # For distributed training example
    kubectl apply -f mnist_single_gpu.yaml # For single gpu training example
    ```
