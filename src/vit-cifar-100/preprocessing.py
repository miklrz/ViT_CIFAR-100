import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import os
import random


def load_CIFAR100_64x64(PATHS, DATA_CONFIG):
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=PATHS["path_to_data"], train=True, download=True, transform=transform
    )

    trainsubset = Subset(
        trainset, random.sample(range(len(trainset)), DATA_CONFIG["num_samples"])
    )

    testset = torchvision.datasets.CIFAR100(
        root=PATHS["path_to_data"], train=False, download=True, transform=transform
    )

    return trainsubset, testset


def get_loaders(DATA_CONFIG, trainset, testset):
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=True,
        num_workers=DATA_CONFIG["num_workers"],
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=False,
        num_workers=DATA_CONFIG["num_workers"],
    )

    return trainloader, testloader


def save_model(PATHS, DATA_CONFIG, net):
    save_dir = PATHS["path_to_saved_models"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_model = os.path.join(save_dir, DATA_CONFIG["saved_model_name"])

    torch.save(net.state_dict(), path_to_model)
    print(f"Model saved to {path_to_model}")
