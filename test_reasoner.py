"""
Script to test the TSP model and a combination of deterministic algorithms

Usage:
   test_reasoner.py (--load-model-from LFM) [options]

Options:
    -h --help              Show this screen.

    --load-model-from LFM  Path to the model to be loaded

    --seed S               Random seed to set. [default: 47]
"""

import time
import os
from docopt import docopt
import schema
from collections import defaultdict

import pytorch_lightning as pl

from models.algorithm_processor import LitAlgorithmProcessor
from hyperparameters import get_hyperparameters
from datasets.constants import _DATASET_ROOTS

name_mapper = {
    "bellman_ford": "Bellman-Ford",
    "bfs": "BFS",
    "dfs": "DFS",
    "dag_shortest_paths": "DAG Shortest Paths",
    "floyd_warshall": "Floyd-Warshall",
    "mst_prim": "MST Prim",
    "strongly_connected_components_local": "SCC",
    "binary_search": "Binary Search",
    "insertion_sort": "Insertion Sort",
    "minimum": "Minimum",
}

run_names_baselines = [
    "decent-bee-661",
    "glowing-valley-660",
    "super-terrain-659",
    "still-sky-644",
    "wise-pine-643",
    "misunderstood-morning-644",
    "earnest-haze-632",
    "rosy-pond-631",
    "scarlet-vortex-630",
    "comic-shadow-620",
    "firm-river-616",
    "wandering-armadillo-615",
    "chocolate-wind-596",
    "comfy-snowball-587",
    "decent-firefly-580",
    "earthy-microwave-579",
    "devout-rain-578",
    "trim-dragon-577",
    "divine-salad-574",
    "fearless-feather-571",
    "floral-glitter-570",
    "expert-water-569",
    "pious-deluge-566",
    "misty-glitter-565",
    "faithful-sky-534",
    "lilac-lion-529",
    "firm-pond-528",
    "unique-dream-522",
    "prime-river-521",
    "fluent-wind-516",
]  # baselines
run_names_deqs = [
    "woven-fire-666",
    "stoic-hill-665",
    "spring-lion-662",
    "stilted-plant-639",
    "comfy-serenity-637",
    "colorful-dust-636",
    "summer-thunder-621",
    "fearless-frog-619",
    "brisk-feather-618",
    "soft-oath-539",
    "northern-sunset-538",
    "wild-disco-537",
    "fallen-glade-448",
    "lilac-dew-446",
    "vivid-haze-445",
    "stellar-firebrand-429",
    "prime-pine-414",
    "brisk-pyramid-413",
    "hopeful-grass-475",
    "balmy-morning-341",
    "trim-thunder-332",
    "jumping-dew-362",
    "clean-pine-363",
    "dazzling-haze-361",
    "glad-plasma-344",
    "prime-capybara-343",
    "confused-feather-342",
    "unique-totem-609",
    "distinctive-vortex-604",
    "different-violet-603",
]  # unmodified DEARs


def get_df_for_run_names(run_names, whattoget="train/loss/average_loss_epoch"):
    runs = api.runs("clrs-cambridge/nardeq")
    runs = list(filter(lambda x: x.name in run_names, runs))
    hist_list_step = []
    hist_list_epoch = []
    for run in runs:
        name = run.config["algorithm_names"][0]
        if name == "binary_search":
            continue

        hist_epoch = run.history(keys=["_step", "epoch", eval(f'f"{whattoget}"')])
        hist_epoch["name"] = name_mapper[name]
        hist_list_epoch.append(hist_epoch)

    df = pd.concat(hist_list_epoch, ignore_index=True)
    return df


MODEL_PATHS = [
    "./serialised_models/pretrained/BS_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/BFS_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/BS_BL_seed3.ckpt",
    "./serialised_models/pretrained/BFS_BL_seed3.ckpt",
    "./serialised_models/pretrained/FW_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/SCC_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/mst_prim_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/BFS_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/FW_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/BS_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/mst_prim_BL_seed1.ckpt",
    "./serialised_models/pretrained/MIN_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/mst_prim_BL_seed2.ckpt",
    "./serialised_models/pretrained/DSP_BL_seed3.ckpt",
    "./serialised_models/pretrained/IS_BL_seed2.ckpt",
    "./serialised_models/pretrained/IS_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/BS_BL_seed2.ckpt",
    "./serialised_models/pretrained/DFS_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/BS_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/SCC_BL_seed1.ckpt",
    "./serialised_models/pretrained/FW_BL_seed1.ckpt",
    "./serialised_models/pretrained/BFS_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/DSP_BL_seed1.ckpt",
    "./serialised_models/pretrained/DSP_BL_seed2.ckpt",
    "./serialised_models/pretrained/SCC_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/FW_BL_seed3.ckpt",
    "./serialised_models/pretrained/MIN_BL_seed1.ckpt",
    "./serialised_models/pretrained/BFS_BL_seed2.ckpt",
    "./serialised_models/pretrained/SCC_BL_seed3.ckpt",
    "./serialised_models/pretrained/MIN_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/BF_BL_seed1.ckpt",
    "./serialised_models/pretrained/mst_prim_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/BF_BL_seed3.ckpt",
    "./serialised_models/pretrained/MIN_BL_seed3.ckpt",
    "./serialised_models/pretrained/DSP_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/MIN_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/IS_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/BF_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/BFS_BL_seed1.ckpt",
    "./serialised_models/pretrained/mst_prim_BL_seed3.ckpt",
    "./serialised_models/pretrained/DFS_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/DSP_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/SCC_BL_seed2.ckpt",
    "./serialised_models/pretrained/DFS_BL_seed1.ckpt",
    "./serialised_models/pretrained/IS_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/BF_BL_seed2.ckpt",
    "./serialised_models/pretrained/DFS_BL_seed2.ckpt",
    "./serialised_models/pretrained/SCC_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/DSP_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/FW_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/mst_prim_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/DFS_BL_seed3.ckpt",
    "./serialised_models/pretrained/DFS_DEQ_seed1.ckpt",
    "./serialised_models/pretrained/BS_BL_seed1.ckpt",
    "./serialised_models/pretrained/IS_BL_seed1.ckpt",
    "./serialised_models/pretrained/IS_BL_seed3.ckpt",
    "./serialised_models/pretrained/BF_DEQ_seed2.ckpt",
    "./serialised_models/pretrained/FW_BL_seed2.ckpt",
    "./serialised_models/pretrained/BF_DEQ_seed3.ckpt",
    "./serialised_models/pretrained/MIN_BL_seed2.ckpt",
]


def test_model(model_path):
    lit_processor = LitAlgorithmProcessor.load_from_checkpoint(
        model_path,
        dataset_root=_DATASET_ROOTS["mst_prim"],
        strict=False,
    )

    start_time = time.time()
    trainer = pl.Trainer(
        accelerator="cuda",  # Change to 'cpu' if you're not using GPU
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
    )
    trainer.test(model=lit_processor)
    end_time = time.time()

    return (end_time - start_time) / 100.0


if __name__ == "__main__":
    serialised_models_dir = os.path.abspath("./serialised_models/")
    hidden_dim = get_hyperparameters()["dim_latent"]
    schema = schema.Schema(
        {
            "--help": bool,
            "--load-model-from": schema.Or(None, os.path.exists),
            "--seed": schema.Use(int),
        }
    )
    args = docopt(__doc__)
    args = schema.validate(args)
    testing_times_dict = defaultdict(list)

    # Iterate over each model path
    for model_path in MODEL_PATHS:
        print("TESTING", model_path)
        # Extract model type and dataset name from the path
        model_type = model_path.split("/")[-1].split("_")[
            1
        ]  # Extracting model type (DEQ or BL)
        dataset_name = model_path.split("/")[-1].split("_")[
            0
        ]  # Extracting dataset name

        # Test the model and record testing time
        testing_time = test_model(model_path)

        # Store testing time for the corresponding model type and dataset
        testing_times_dict[(model_type, dataset_name)].append(testing_time)

        # Dictionary to store mean testing times for each combination of model type and dataset
        mean_testing_times = defaultdict(dict)
        std_testing_times = defaultdict(dict)

    for key, times in testing_times_dict.items():
        mean_time = torch.tensor(times).mean().item()
        std_time = torch.tensor(times).std().item()
        mean_testing_times[key[0]][key[1]] = mean_time
        std_testing_times[key[0]][key[1]] = std_time

    # Print mean testing times
    print("MEAN:")
    for model_type, dataset_times in mean_testing_times.items():
        print(f"{model_type}:")
    for dataset, mean_time in dataset_times.items():
        print(f"  {dataset}: {mean_time} seconds")

    # Print standard deviation of testing times
    print("\nSTD:")
    for model_type, dataset_times in std_testing_times.items():
        print(f"{model_type}:")
    for dataset, std_time in dataset_times.items():
        print(f"  {dataset}: {std_time} seconds")

    # lit_processor = LitAlgorithmProcessor.load_from_checkpoint(
    #    args['--load-model-from'],
    #    dataset_root=_DATASET_ROOTS['mst_prim'],
    #    strict=False,
    # )

    # trainer = pl.Trainer(
    #    accelerator='cuda',
    #    check_val_every_n_epoch=1,
    #    log_every_n_steps=100,
    # )
    # trainer.test(
    #    model=lit_processor,
    # )
