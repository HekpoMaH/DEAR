import torch
from pytorch_lightning.utilities.seed import seed_everything
from hyperparameters import get_hyperparameters
from datasets.constants import _DATASET_SPECS
from datasets._configs import CONFIGS


def construct_by_num_nodes_and_splits(dataset_names, splits, do_plot=False):

    for dn in dataset_names:

        def _construct(split, nn):
            ns = CONFIGS[dn][split]["num_samples"]
            dataclass = _DATASET_SPECS[dn]["dataclass"]
            rd = _DATASET_SPECS[dn]["rootdir"]
            offset = nn * 3
            if split == "val":
                offset += 1
            if split == "test":
                offset += 2
            print(
                f'constructing algorithm {dn}/{rd}/seed: {get_hyperparameters()["seed"]+offset}'
            )
            f = dataclass(
                rd,
                algorithm=dn,
                split=split,
                num_nodes=nn,
                num_samples=ns,
                seed=(get_hyperparameters()["seed"] + offset),
            )
            return f

        for split in splits:
            nns = CONFIGS[dn][split]["num_nodes"]
            if isinstance(nns, int):
                nns = [nns]
            for nn in nns:
                f = _construct(split, nn)
            if do_plot:
                plot_histogram_boxplot(f.data.edge_index.cpu().numpy(), name=str(f))


def print_by_num_nodes_and_splits(dataset_names, splits, do_plot=False):

    for dn in dataset_names:
        # for nn in num_nodes:
        def _construct(split, nn):
            ns = CONFIGS[dn][split]["num_samples"]
            dataclass = _DATASET_SPECS[dn]["dataclass"]
            rd = _DATASET_SPECS[dn]["rootdir"]
            offset = nn * 3
            if split == "val":
                offset += 1
            if split == "test":
                offset += 2
            f = dataclass(
                rd,
                algorithm=dn,
                split=split,
                num_nodes=nn,
                num_samples=ns,
                seed=(get_hyperparameters()["seed"] + offset),
            )
            print(
                "class",
                dataclass,
                "nn",
                nn,
                "sum",
                f[0].edge_index.sum(),
                f[0].pos.sum(),
            )
            return f

        for split in splits:
            nns = CONFIGS[dn][split]["num_nodes"]
            if isinstance(nns, int):
                nns = [nns]
            for nn in nns:
                f = _construct(split, nn)
            # f = _construct(split)
            if do_plot:
                plot_histogram_boxplot(f.data.edge_index.cpu().numpy(), name=str(f))


if __name__ == "__main__":
    seed = get_hyperparameters()["seed"]
    seed_everything(seed)
    print(f"SEEDED with {seed}")

    construct_by_num_nodes_and_splits(
        [
            "bfs",
            "insertion_sort",
            "bellman_ford",
            "dag_shortest_paths",
            "floyd_warshall",
            "mst_prim",
            "strongly_connected_components_local",
            "dfs",
            "binary_search",
            "minimum",
            "search",
        ],
        ["train", "val", "test"],
    )  # NOTE Also available splits are 'test_clrs', 'test_128', 'test_256', 'test_512'

    print_by_num_nodes_and_splits(
        [
            "bfs",
            "insertion_sort",
            "bellman_ford",
            "dag_shortest_paths",
            "floyd_warshall",
            "mst_prim",
            "strongly_connected_components_local",
            "dfs",
            "binary_search",
            "minimum",
            "search",
        ],
        ["train", "val", "test"],
    )  # NOTE Also available splits are 'test_clrs', 'test_128', 'test_256', 'test_512'
