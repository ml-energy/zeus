## Environment Setup
1. Git clone `SymbioticLab/Zeus`.
2. Get [k3s with `Docker` as the container runtime](https://docs.k3s.io/advanced#using-docker-as-the-container-runtime)
```shell
curl -sfL https://get.k3s.io | sh -s - --docker
```
3. Install [Kubeflow](https://github.com/kubeflow/manifests#installation) manually
4. Install FastAPI
```shell
pip install "fastapi[all]"
```

## Learning Materials
- FastAPI
  - [Tutorial: FastAPI and asynchronous programming](https://fastapi.tiangolo.com/tutorial/)
- Kubernetes
  - [Tutorial: Learn Kubernetes and try out minikube](https://www.youtube.com/watch?v=d6WC5n9G_sM)
  - [Docs: Understand Kubernetes](https://kubernetes.io/docs/home/)

## Progress
Please check out [RFC: Kubeflow Integration](https://github.com/SymbioticLab/Zeus/issues/11) for our design.
Following is the list of components and the latest progress:
- `ZeusServer`:
  - [x] APIs between `ZeusServer` and clients.
    - APIs in `kube/server/main.py`.
    - Model definitions (i.e. data structures communicated between server and clients) in `kube/server/models.py`.
  - [ ] `ZeusServer` class that manages the training jobs
    - In `kube/server/server.py`. This singleton class contains all the functions that do the "real" work on the server side.
- Database and ORM
  - [x] Table schema `kube/db/schema.md`.
  - [ ] Functions that `ZeusServer` will use to store and query states (see `kube/server/dbapis.py`). Basically, send DB query and return the result of `ZeusServer`.
- Extended `ZeusDataLoader`
  - [ ] Receive `job_id` from the user as a CLI argument. Note this is a job-specific parameter.
  - [ ] Extend `ZeusDataLoader`.
    - In `__init__`, register the job (i.e. POST to `ZeusServer`) and receive the batch size before initializing the `DataLoader`.
    - Report `ProfilingResult` when profiling is done.
    - Report `TrialResult` when training is done.

## Additional
- Kubeflow provides `PytorchJob`
  - [x] Example of configuring a trial as a `PytorchJob` with `.yaml` (see `kube/job/cifar100_try.yaml`), which includes:
    - Passing CLI parameters for this trial
    - Exporting Zeus-related environment parameters
    - Setting system capability for this trail
    - Setting resource limit (see `kube/job/gpu_job.yaml`)
  - [x] [Python client library for Kubernetes](https://github.com/kubernetes-client/python) can be used to configure a trial as well.
