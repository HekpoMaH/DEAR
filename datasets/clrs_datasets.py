import os.path as osp
import shutil
import os
import requests
from absl import logging
import inspect
import clrs
from typing import Any, Callable, List, Optional, Tuple
from clrs._src.algorithms import (
    mst_prim,
    bellman_ford,
    graham_scan,
    heapsort,
    insertion_sort,
    strongly_connected_components,
    kmp_matcher,
    floyd_warshall,
    find_maximum_subarray,
    find_maximum_subarray_kadane,
    matrix_chain_order,
    lcs_length,
    optimal_bst,
    segments_intersect,
    jarvis_march,
    dfs,
    bfs,
    topological_sort,
    articulation_points,
    bridges,
    mst_kruskal,
    dijkstra,
    dag_shortest_paths,
    bipartite_matching,
    activity_selector,
    task_scheduling,
    minimum,
    binary_search,
    quickselect,
    bubble_sort,
    quicksort,
    naive_string_matcher,
)
from clrs._src.specs import Stage, Location, Type
from clrs._src.samplers import Sampler
from clrs._src import specs

import time
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric
from torch_geometric.data import Data, DataLoader
from utils_execution import check_edge_index_sorted, edge_one_hot_encode_pointers
from datasets.geometric_sampler import build_geometric_sampler
from datasets.specs import SPECS
from datasets.samplers import SAMPLERS
from datasets.local_algos import *

PRED_AS_INPUT_ALGOS = [
    "binary_search",
    "minimum",
    "find_maximum_subarray",
    "find_maximum_subarray_kadane",
    "matrix_chain_order",
    "lcs_length",
    "optimal_bst",
    "activity_selector",
    "task_scheduling",
    "naive_string_matcher",
    "kmp_matcher",
    "jarvis_march",
]

_algorithms = {
    "mst_prim": mst_prim,
    "bellman_ford": bellman_ford,
    "floyd_warshall": floyd_warshall,
    "graham_scan": graham_scan,
    "find_maximum_subarray": find_maximum_subarray,
    "find_maximum_subarray_kadane": find_maximum_subarray_kadane,
    "matrix_chain_order": matrix_chain_order,
    "lcs_length": lcs_length,
    "optimal_bst": optimal_bst,
    "segments_intersect": segments_intersect,
    "jarvis_march": jarvis_march,
    "dfs": dfs,
    "bfs": bfs,
    "kmp_matcher": kmp_matcher,
    "topological_sort": topological_sort,
    "articulation_points": articulation_points,
    "bridges": bridges,
    "mst_kruskal": mst_kruskal,
    "dijkstra": dijkstra,
    "dag_shortest_paths": dag_shortest_paths,
    "bipartite_matching": bipartite_matching,
    "activity_selector": activity_selector,
    "task_scheduling": task_scheduling,
    "minimum": minimum,
    "binary_search": binary_search,
    "search": search,
    "quickselect": quickselect,
    "bubble_sort": bubble_sort,
    "quicksort": quicksort,
    "naive_string_matcher": naive_string_matcher,
    "heapsort": heapsort,
    "heapsort_local": heapsort_local,
    "insertion_sort_local": insertion_sort_local,
    "insertion_sort": insertion_sort,
    "strongly_connected_components": strongly_connected_components,
    "strongly_connected_components_local": strongly_connected_components_local,
}

ALGOS_REQUIRING_UNDIRECTED = set(
    [
        "dfs",
        "strongly_connected_components",
        "dag_shortest_paths",
    ]
)


def _iterate_sampler(sampler, batch_size):
    while True:
        yield sampler.next(batch_size)


def _load_inputs(data, feedback, spec, i=0, requires_undirected=False):
    spec = copy.deepcopy(spec)
    attrs, adj, num_nodes = None, None, None
    for inp in feedback.features.inputs:
        if inp.name == "pos":
            num_nodes = inp.data.shape[1]

        if inp.name == "A":
            attrs = torch.tensor(inp.data[i], dtype=torch.float32)
            continue
        if inp.name == "adj":
            adj = torch.tensor(inp.data[i])
            continue
        new_name = inp.name
        if spec[inp.name][1:] == (Location.NODE, Type.POINTER):
            new_name = f"{new_name}_index"
            spec[new_name] = spec.pop(inp.name)
        setattr(data, new_name, torch.tensor(inp.data[i], dtype=torch.float32))

    if adj is None:
        adj = torch.ones((num_nodes, num_nodes))
    data.edge_index = adj.nonzero().T
    if requires_undirected:
        data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
    data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
        data.edge_index, num_nodes=num_nodes
    )
    data.edge_index = torch_geometric.utils.coalesce(data.edge_index)
    if attrs is not None:
        attrs = torch.tensor(attrs, dtype=torch.float32)
        data.A = attrs[data.edge_index[0], data.edge_index[1]]
    data.num_nodes = adj.shape[0]
    data.lengths = torch.tensor(feedback.features.lengths[i]).expand(1)

    return data, spec


def _load_hints_and_outputs(data, feedback, spec, i=0):
    spec = copy.deepcopy(spec)

    def _prep_probe(unpr_probe, name):
        probe = torch.tensor(np.array(unpr_probe), dtype=torch.float32)

        if (
            spec[name][0] == Stage.HINT
            and spec[name][1] == Location.NODE
            and spec[name][2] not in (Type.POINTER, Type.SHOULD_BE_PERMUTATION)
            and "stack_prev" not in name
        ):
            probe = probe.transpose(0, 1)
        elif spec[name][-2] == Location.EDGE:
            if spec[name][0] == Stage.HINT:
                if spec[name][2] not in (Type.POINTER, Type.SHOULD_BE_PERMUTATION):
                    probe = probe[:, data.edge_index[0], data.edge_index[1]].transpose(
                        0, 1
                    )
                else:
                    probe = probe[:, data.edge_index[0], data.edge_index[1]].long()
            if spec[name][0] == Stage.OUTPUT:
                probe = probe[data.edge_index[0], data.edge_index[1]]
        elif spec[name][-2] == Location.GRAPH:
            probe = probe.unsqueeze(0)
            assert probe.dtype == torch.float32
        elif spec[name][1] == Location.NODE and spec[name][2] in (
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ):  # i.e it's a predecessor-like
            probe = probe.long()
            if spec[name][2] == Type.PERMUTATION_POINTER:
                probe = probe.argmax(-1)
            if spec[name][2] == Type.SHOULD_BE_PERMUTATION:
                sl_at = (probe == torch.arange(probe.shape[0])).float().argmax()
                maximum_at = (
                    F.one_hot(probe, num_classes=(probe.shape[0])).sum(0).argmin()
                )
                probe[sl_at] = maximum_at

        return probe

    for hint in feedback.features.hints:
        probe = _prep_probe(hint.data[:, i], hint.name)
        new_name = hint.name
        if spec[hint.name][2] in (
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ):
            new_name = f"{new_name}_index"
        spec[f"{new_name}_temporal"] = spec.pop(hint.name)
        setattr(data, f"{new_name}_temporal", probe)

    for out in feedback.outputs:
        if not out.data.size:
            spec.pop(out.name)
            continue
        new_name = out.name
        if spec[out.name][2] in (
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ):
            new_name = f"{new_name}_index"
        spec[f"{new_name}"] = spec.pop(out.name)
        probe = _prep_probe(out.data[i], new_name)
        setattr(data, new_name, probe)

    return data, spec


def _pad_hints(hints, length):
    for i, hint in enumerate(hints):
        padder_arr = []
        assert length >= hint.data.shape[0], "please pad with a non-smaller length"
        old_dim = hint.data.shape[0]
        padder_arr.append((0, length - hint.data.shape[0]))
        padder_arr.extend([(0, 0) for _ in range(len(hint.data.shape) - 1)])
        hints[i].data = np.pad(hint.data, padder_arr, mode="constant")
        assert hints[i].data[old_dim:].sum() == 0

    return hints


def build_sampler(
    name: str,
    algorithm: str,
    num_samples: int,
    *args,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
    """Builds a sampler. See `Sampler` documentation."""

    if name not in SPECS or name not in SAMPLERS:
        raise NotImplementedError(f"No implementation of algorithm {name}.")
    spec = SPECS[name]
    sampler_class = SAMPLERS[name]
    # Ignore kwargs not accepted by the sampler.
    sampler_args = inspect.signature(
        sampler_class._sample_data
    ).parameters  # pylint:disable=protected-access
    clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}
    if set(clean_kwargs) != set(kwargs):
        logging.warning(
            "Ignoring kwargs %s when building sampler class %s",
            set(kwargs).difference(clean_kwargs),
            sampler_class,
        )
    sampler = sampler_class(
        algorithm, spec, num_samples, seed=seed, *args, **clean_kwargs
    )
    return sampler, spec


def _maybe_download_dataset(dataset_path):
    """Download CLRS30 dataset if needed."""
    dataset_folder = osp.join(dataset_path, clrs.get_clrs_folder())
    if osp.isdir(dataset_folder):
        logging.info("Dataset found at %s. Skipping download.", dataset_folder)
        return dataset_folder
    logging.info("Dataset not found in %s. Downloading...", dataset_folder)

    clrs_url = clrs.get_dataset_gcp_url()
    request = requests.get(clrs_url, allow_redirects=True)
    clrs_file = osp.join(dataset_path, osp.basename(clrs_url))
    os.makedirs(dataset_folder)
    open(clrs_file, "wb").write(request.content)
    shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
    os.remove(clrs_file)
    return dataset_folder


class CLRS(torch_geometric.data.Dataset):
    @property
    def processed_file_names(self):
        return [f"processed_{i}.pt" for i in range(self.num_samples)]

    @property
    def processed_dir(self):
        return osp.join(
            self.root,
            self.algorithm,
            f"num_nodes_{self.num_nodes}",
            f"randomise_pos_{self.randomise_pos}",
            f"sampler_type_{self.sampler_type}",
            "processed",
            self.split,
        )

    @property
    def spec_path(self):
        return osp.join(
            self.root,
            self.algorithm,
            f"num_nodes_{self.num_nodes}",
            f"randomise_pos_{self.randomise_pos}",
            f"sampler_type_{self.sampler_type}",
            "processed",
            self.split,
            "spec.pt",
        )

    def __init__(
        self,
        root,
        num_nodes,
        num_samples,
        algorithm="mst_prim",
        split="train",
        sampler_type="normal",
        randomise_pos=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        seed=0,
    ):
        self.split = split
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.seed = seed
        self.sampler_type = sampler_type
        self.algorithm = algorithm
        self._rng = np.random.RandomState(self.seed)
        self.randomise_pos = randomise_pos
        self.samplers = dict()
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.spec, self.attribute_list = torch.load(self.spec_path)

    def get_sampler(self, size):
        if self.split == "train" and self.algorithm not in [
            "kmp_matcher"
        ]:  # NOTE NOTE THE NOT
            size = self._rng.randint(size // 2, size + 1)
        if self.algorithm in ["kmp_matcher"]:
            orig_size = size
            size = size * 5 // 4
        if size not in self.samplers:
            spec = SPECS[self.algorithm]
            algo = _algorithms[self.algorithm]
            if self.split == "test_clrs":
                dataset_folder = _maybe_download_dataset("./data/tmp/")
                self.samplers[size], _, spec = clrs.create_dataset(
                    folder=dataset_folder,
                    algorithm=self.algorithm,
                    batch_size=1,
                    split="test",
                )
                self.samplers[size] = self.samplers[size].as_numpy_iterator()
                self.samplers[size] = (self.samplers[size], spec)
            else:
                self.samplers[size] = build_sampler(
                    self.algorithm,
                    algo,
                    -1,
                    length=size,
                    p=tuple([0.1 + 0.1 * i for i in range(9)]),
                    seed=self.seed,
                    min_length=self.num_nodes,
                    length_needle=-(size - 1),
                )

            smplr, spec = self.samplers[size]
            smplr.iterator = (
                _iterate_sampler(smplr, 1) if self.split != "test_clrs" else smplr
            )
            if self.randomise_pos:
                smplr.iterator = clrs.process_random_pos(smplr.iterator, self._rng)
            spec, smplr.iterator = clrs.process_permutations(spec, smplr.iterator, True)
            if self.algorithm in PRED_AS_INPUT_ALGOS:
                spec, smplr.iterator = clrs.process_pred_as_input(spec, smplr.iterator)
            self.samplers[size] = (smplr, spec)

        return self.samplers[size]

    def process(self):
        maxlen = 0
        for _ in tqdm(range(len(self.processed_paths) // 10)):
            generator, spec = self.get_sampler(self.num_nodes)

            fdb = next(generator.iterator)

            maxlen = max(maxlen, int(fdb.features.hints[0].data.shape[0]))

        maxlen = int(maxlen * 1.5)
        print("Max len is", maxlen)

        for i, proc_path in tqdm(enumerate(self.processed_paths)):
            generator, spec = self.get_sampler(self.num_nodes)

            fdb = next(generator.iterator)

            feedback = fdb
            feedback.features._replace(
                hints=_pad_hints(feedback.features.hints, maxlen)
            )
            data = Data()
            data, new_spec = _load_inputs(
                data,
                feedback,
                spec,
                requires_undirected=self.algorithm in ALGOS_REQUIRING_UNDIRECTED,
            )
            data, new_spec = _load_hints_and_outputs(data, feedback, new_spec)

            check_edge_index_sorted(data.edge_index)
            torch.save(data, proc_path)

        attribute_list = list(new_spec.keys())
        torch.save((new_spec, attribute_list), self.spec_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"processed_{idx}.pt"))
        return data


if __name__ == "__main__":
    scp = CLRS("./data/clrs/", 64, 100, algorithm="heapsort", split="test")
    ldr = DataLoader(scp, batch_size=2)
    items = list(iter(ldr))
    breakpoint()
