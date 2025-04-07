HYPERPARAMS = {
    "epochs": 6,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "img_size": 64,
    "patch_size": 4,
    "in_chans": 3,
    "num_classes": 100,
    "embed_dim": 512,
    "depth": 8,
    "num_heads": 8,
    "mlp_ratio": 4,
    "qkv_bias": True,
    "drop_rate": 0.5,
    "log_interval": 200,
    "patience": 3,
}

PATHS = {
    "path_to_saved_models": "experiments",
    "path_to_data": "loader_data",
}

WANDB_CONFIG = {
    "project": "ViT_CIFAR-100",
}

DATA_CONFIG = {
    "batch_size": 64,
    "num_workers": 2,
    "saved_model_name": "ViT_classification.pth",
    "num_samples": 50000,
}
