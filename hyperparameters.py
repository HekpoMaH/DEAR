import torch


def get_hyperparameters():
    return {
        "dim_latent": 128,
        "num_bits": 8,
        "lr": 0.0003,
        "nee_warmup_steps": 4000,
        "dim_nodes_mst_prim": 1,
        "dim_target_mst_prim": 1,
        "device": "cuda",
        "batch_size": 32,
        "bias": True,
        "seed": 47,  # for dataset generation
        "calculate_termination_statistics": False,
    }
