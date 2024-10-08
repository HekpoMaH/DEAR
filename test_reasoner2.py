import time
import os
import torch
import wandb
import pandas as pd
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

spr = ["snowy-snowball-1328", "lyric-bee-1324", "splendid-dream-1320"]
# run_names_baselines = ["decent-bee-661", "glowing-valley-660", "super-terrain-659", "still-sky-644", "wise-pine-643", "misunderstood-morning-644", "earnest-haze-632", "rosy-pond-631", "scarlet-vortex-630", "comic-shadow-620", "firm-river-616", "wandering-armadillo-615", "chocolate-wind-596", "comfy-snowball-587", "decent-firefly-580", "earthy-microwave-579", "devout-rain-578", "trim-dragon-577", "divine-salad-574", "fearless-feather-571", "floral-glitter-570", "expert-water-569", "pious-deluge-566", "misty-glitter-565", "faithful-sky-534", "lilac-lion-529", "firm-pond-528", "unique-dream-522", "prime-river-521", "fluent-wind-516"] # baselines
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


def get_df_for_run_names(run_names, whattoget="train/loss/average_loss/epoch"):
    runs = wandb.Api().runs("clrs-cambridge/nardeq")
    runs = list(filter(lambda x: x.name in run_names, runs))
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


runs = wandb.Api().runs("clrs-cambridge/nardeq")
names = [x.name for x in runs]


def download_model_for_run(run_name):
    print("downloading", run_name)

    run = runs[names.index(run_name)]
    arti = list(filter(lambda r: r.type == "model", run.logged_artifacts()))[0]
    arti.download()
    model_path = arti.file()
    print(model_path)
    name = run.config["algorithm_names"][0]
    return model_path, name


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
    suffix = ""
    metrics = trainer.test(
        model=lit_processor, dataloaders=lit_processor.test_dataloader(suffix=suffix)
    )
    end_time = time.time()

    return metrics, (end_time - start_time) / 100.0


if __name__ == "__main__":
    testing_times_dict = defaultdict(list)
    testing_metrics_dict = defaultdict(list)

    # Iterate over each run name
    for run_name in spr:
        model_path, name = download_model_for_run(run_name)
        # if name in ['floyd_warshall', 'strongly_connected_components_local']:
        #     continue
        print("Testing", run_name)
        metrics, testing_time = test_model(model_path)
        testing_times_dict[name].append(testing_time)
        testing_metrics_dict[name].append(metrics[0])

    from pprint import pprint

    for run_name in spr:
        model_path, name = download_model_for_run(run_name)
        # if name in ['floyd_warshall', 'strongly_connected_components_local']:
        #     continue
        dict_of_lists = {
            key: [d[key] for d in testing_metrics_dict[name]]
            for key in testing_metrics_dict[name][0]
        }
        pprint(
            {
                k: (torch.tensor(v).mean().item(), torch.tensor(v).std().item())
                for k, v in dict_of_lists.items()
            }
        )

    # Calculate mean testing times
    mean_testing_times = {
        k: torch.tensor(v).mean().item() for k, v in testing_times_dict.items()
    }

    # Print mean testing times
    print("MEAN:")
    for run_name, mean_time in mean_testing_times.items():
        print(f"{run_name}: {mean_time} seconds")
