import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from net import ViT
from preprocessing import load_CIFAR100_64x64, get_loaders, save_model
from config import (
    HYPERPARAMS,
    PATHS,
    WANDB_CONFIG,
    DATA_CONFIG,
)
from train import train, test


def main():
    wandb_run = wandb.init(
        project=WANDB_CONFIG["project"],
        config=HYPERPARAMS,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset, testset = load_CIFAR100_64x64(PATHS, DATA_CONFIG)
    trainloader, testloader = get_loaders(DATA_CONFIG, trainset, testset)

    net = ViT(
        img_size=HYPERPARAMS["img_size"],
        patch_size=HYPERPARAMS["patch_size"],
        in_chans=HYPERPARAMS["in_chans"],
        num_classes=HYPERPARAMS["num_classes"],
        embed_dim=HYPERPARAMS["embed_dim"],
        depth=HYPERPARAMS["depth"],
        num_heads=HYPERPARAMS["num_heads"],
        mlp_ratio=HYPERPARAMS["mlp_ratio"],
        qkv_bias=HYPERPARAMS["qkv_bias"],
        drop_rate=HYPERPARAMS["drop_rate"],
    )
    net.to(device)

    train(
        net,
        device,
        trainloader,
        testloader,
        HYPERPARAMS,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(
            net.parameters(),
            lr=HYPERPARAMS["learning_rate"],
            weight_decay=HYPERPARAMS["weight_decay"],
        ),
        wandb_run=wandb_run,
        log_interval=HYPERPARAMS["log_interval"],
    )
    test(
        net,
        testloader,
        device,
        wandb_run=wandb_run,
        criterion=nn.CrossEntropyLoss(),
    )

    wandb.finish()

    save_model(PATHS, DATA_CONFIG, net=net)


if __name__ == "__main__":
    main()
