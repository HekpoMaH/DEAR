import os, glob
import torch
from collections import defaultdict
import torch.nn.functional as F
import torch_geometric
import torch_scatter
import pytorch_lightning as pl
import networkx as nx


def maybe_remove(path):  # path can be regex
    try:
        for f in glob.glob(path):
            os.remove(f)
    except Exception:
        pass


class ZeroerCallback(pl.callbacks.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.custom_logs = defaultdict(list)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.custom_logs = defaultdict(list)


class ProcessorZeroerCallback(pl.callbacks.Callback):
    @staticmethod
    def zero_it(pl_module):
        pl_module.custom_logs = defaultdict(list)
        for name, algorithm in pl_module.algorithms.items():
            algorithm.algorithm_module.zero_validation_stats()

    def on_validation_epoch_start(self, trainer, pl_module):
        ProcessorZeroerCallback.zero_it(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        ProcessorZeroerCallback.zero_it(pl_module)


class ReasonerZeroerCallback(pl.callbacks.Callback):
    @staticmethod
    def zero_it(pl_module):
        pl_module.custom_logs = defaultdict(list)
        pl_module.algorithm_module.zero_validation_stats()

    def on_validation_epoch_start(self, trainer, pl_module):
        ReasonerZeroerCallback.zero_it(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        ReasonerZeroerCallback.zero_it(pl_module)


def cross_entropy(pred, softmax_idx, truth_1h, num_nodes):
    lsm_pred = torch_scatter.scatter_log_softmax(
        pred, softmax_idx, dim=-1, dim_size=num_nodes, eps=1e-9
    )
    return -truth_1h * lsm_pred


def check_edge_index_sorted(ei):
    for i in range(ei.shape[1] - 1):
        assert ei[0][i] <= ei[0][i + 1]
        if ei[0][i] == ei[0][i + 1]:
            assert ei[1][i] < ei[1][i + 1]


def prepare_constants(batch):
    SIZE = batch.num_nodes
    STEPS_SIZE = batch.lengths.max() - 1
    return SIZE, STEPS_SIZE


def get_callbacks(
    name, serialised_models_dir, patience, monitor="val/loss/average_loss"
):
    best_checkpointing_cb = pl.callbacks.ModelCheckpoint(
        dirpath=serialised_models_dir,
        filename=f"best_{name}",
        save_top_k=1,
        monitor=monitor,
        mode="min",
    )
    all_cbs = [best_checkpointing_cb]  # , checkpoint_cb]
    if patience is not None:
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            verbose=True,
            mode="min",
        )
        all_cbs.append(early_stopping_cb)
    return all_cbs


def edge_one_hot_encode_pointers(pred, batch, inv=False):

    stacked = torch.stack(
        [torch.arange(pred.shape[0], device=pred.device).long(), pred.long()], dim=0
    )
    pi_dense = torch_geometric.utils.to_dense_adj(stacked, batch=batch.batch)
    ebatch = batch.edge_index_batch
    e1 = batch.edge_index[0] - batch.ptr[ebatch]
    e2 = batch.edge_index[1] - batch.ptr[ebatch]
    msk = pi_dense[ebatch, e1, e2] if not inv else pi_dense[ebatch, e2, e1]
    return msk


def edge_one_hot_encode_pointers_edge(ptrs, batch, max_nodes_in_graph):
    tns = torch.full((batch.edge_index.shape[1], max_nodes_in_graph), 0.0).to(
        batch.edge_index.device
    )

    tns[torch.arange(ptrs.shape[0]), ptrs] = 1.0
    return tns


def compute_tour_cost(tour, weights):
    src_t, dst_t = tour
    _, num_nodes = tour.shape

    W = weights.reshape(num_nodes, num_nodes)
    tour_cost = 0
    for u, v in zip(src_t, dst_t):
        tour_cost += W[u, v]

    return tour_cost


def merge_nested_dicts(dicts):
    result = {}

    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                # If the value is a nested dictionary, recursively merge it
                result[key] = merge_nested_dicts([result.get(key, {}), value])
            else:
                # If the value is not a dictionary, convert it to a list
                if key in result:
                    result[key].append(value)
                else:
                    result[key] = [value]

    return result
