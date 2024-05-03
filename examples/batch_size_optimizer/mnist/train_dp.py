"""Mnist example from https://github.com/kubeflow/training-operator/blob/c20422067e3ef81df39d03c6f285353344d8f77d/examples/pytorch/mnist/mnist.py"""

from __future__ import print_function

import argparse
import os
from time import sleep

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms

from zeus.callback import Callback, CallbackSet
from zeus.monitor import ZeusMonitor
from zeus.optimizer.batch_size import BatchSizeOptimizer, JobSpec
from zeus.optimizer.power_limit import GlobalPowerLimitOptimizer
from zeus.utils.lr_scaler import LinearScaler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer, callbacks: CallbackSet):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        ### Zeus usage: call callback for step begin
        callbacks.on_step_begin()

        # Attach tensors to the device.
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)

        ### Zeus usage: call callback for step end
        callbacks.on_step_end()


def test(model, device, test_loader, writer, epoch) -> float:
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Attach tensors to the device.
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar("accuracy", float(correct) / len(test_loader.dataset), epoch)
    return float(correct) / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.5,
        help="Target accuracy (default: 0.5)",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {rank}")

    dist.init_process_group(backend=args.backend, init_method="env://")

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)

    model = nn.parallel.DistributedDataParallel(model)

    # Get FashionMNIST train and test dataset.
    train_ds = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_ds = datasets.FashionMNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    ########################### ZEUS INIT BEGIN ###########################
    monitor = ZeusMonitor(gpu_indices=None)  # All visible GPUs.
    bso = BatchSizeOptimizer(
        monitor=monitor,
        server_url=os.environ.get("ZEUS_SERVER_URL", "http://localhost:30100"),
        job=JobSpec(
            job_id=os.environ.get("ZEUS_JOB_ID"),
            job_id_prefix="mnist-dev",
            default_batch_size=256,
            batch_sizes=[32, 64, 256, 512, 1024, 4096, 2048],
            max_epochs=args.epochs,
            target_metric=args.target_accuracy,
        ),
        rank=rank,
    )

    # The rank 0 process will monitor and optimize the power limit of all GPUs.
    if rank == 0:
        callback_set: list[Callback] = [GlobalPowerLimitOptimizer(monitor), bso]
        # Get batch size from bso
        batch_size = bso.get_batch_size()
        print("Rank", dist.get_rank())
        print("Chosen batach_size:", batch_size)
        # Need to broadcast 1. batch size 2. trial_number to identify current trial from other GPUs
        bs_trial_tensor = torch.tensor([batch_size, bso.trial_number], device="cuda")
    else:
        sleep(3)  # Wait for the initailization of rank=0 gpu
        callback_set = [bso]
        bs_trial_tensor = torch.tensor([0, 0], device="cuda")
        print("Rank", dist.get_rank())

    dist.broadcast(bs_trial_tensor, src=0)
    bso.current_batch_size = int(bs_trial_tensor[0].item())
    bso.trial_number = int(bs_trial_tensor[1].item())
    batch_size = bso.current_batch_size // world_size

    print(f"Batach_size to use for gpu[{rank}]: {batch_size}, trial number: {bso.trial_number}")
    callbacks = CallbackSet(callback_set)

    # Scale the learning rate accordingly.
    # Default was batch size 64 and learing rate 0.01, and we use the linear
    # scaling rule since the optimizer is SGD.
    args.lr = LinearScaler(bs=64, lr=0.01).compute_lr(new_bs=batch_size * world_size)
    ########################### ZEUS INIT END ###########################

    # Add train and test loaders.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=DistributedSampler(train_ds),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        sampler=DistributedSampler(test_ds),
    )

    ########################### ZEUS USAGE BEGIN ###########################
    callbacks.on_train_begin()
    for epoch in range(1, args.epochs + 1):
        callbacks.on_epoch_begin()
        train(args, model, device, train_loader, epoch, writer, callbacks)
        callbacks.on_epoch_end()
        acc = test(model, device, test_loader, writer, epoch)
        callbacks.on_evaluate(acc)

        if bso.training_finished:
            break
    ########################### ZEUS USAGE END ###########################

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
