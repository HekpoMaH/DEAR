import time
from pprint import pprint
from collections import defaultdict
import copy
from enum import Enum
import numpy
from sklearn.metrics import precision_score, recall_score
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_f1_score

import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch_scatter
from torch_sparse import SparseTensor
from torchdeq.loss import jac_reg
import pytorch_lightning as pl

from datasets._configs import CONFIGS
from layers.algorithm_reasoner import AlgorithmReasoner
from layers.DEQ_reasoner import DEQReasoner
from hyperparameters import get_hyperparameters
from utils_execution import cross_entropy, edge_one_hot_encode_pointers
from clrs import Type, Location, Stage
from utils.expander import build_cayley_bank, get_cayley_data

_REASONERS = defaultdict(lambda: AlgorithmReasoner)
_REASONERS["deq"] = DEQReasoner


class AlgoType(Enum):
    SORTING = 1
    ANY = -1


ALGO_TYPES = defaultdict(lambda: AlgoType.ANY) | {
    "heapsort": AlgoType.SORTING,
    "heapsort_local": AlgoType.SORTING,
    "insertion_sort_local": AlgoType.SORTING,
    "insertion_sort": AlgoType.SORTING,
}


class LitAlgorithmReasoner(pl.LightningModule):
    def __init__(
        self,
        hidden_dim,
        algo_processor,
        dataset_class,
        dataset_root,
        dataset_kwargs,
        algorithm="mst_prim",
        use_TF=False,
        use_sinkhorn=True,
        xavier_on_scalars=True,
        learning_rate=get_hyperparameters()["lr"],
        test_with_val=True,
        test_with_val_every_n_epoch=20,
        test_train_every_n_epoch=20,
        config=None,
        **algorithm_base_kwargs,
    ):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim
        if config["use_expander"]:
            self.cayley_bank = build_cayley_bank()
        self.algorithm_base_kwargs = algorithm_base_kwargs
        self.dataset_class = dataset_class
        self.dataset_root = dataset_root
        self.dataset_kwargs = dataset_kwargs
        self.learning_rate = learning_rate
        self.weight_decay = config["weight_decay"]
        self.timeit = False
        self.use_TF = use_TF
        self.use_sinkhorn = use_sinkhorn
        self.algorithm_base_kwargs = algorithm_base_kwargs
        self.algorithm = algorithm
        self.xavier_on_scalars = xavier_on_scalars
        self.test_with_val = test_with_val
        self.test_with_val_every_n_epoch = test_with_val_every_n_epoch
        self.test_train_every_n_epoch = test_train_every_n_epoch
        self._datasets = {}
        if self.test_with_val:
            self.val_dataloader = self.val_dataloader_alt
            self.validation_step = self.validation_step_alt
        self._current_epoch = 0
        self._global_step = 0
        self.load_dataset("train")

        algo_cls = _REASONERS[config["algo_cls"]]
        self.algorithm_module = algo_cls(
            self.dataset.spec,
            self.dataset[0],
            hidden_dim,
            algo_processor,
            use_TF=use_TF,
            use_sinkhorn=use_sinkhorn,
            timeit=self.timeit,
            xavier_on_scalars=xavier_on_scalars,
            config=config,
            **algorithm_base_kwargs,
        )
        if config["algo_cls"] == "deq" and config["deq"]["align"]:
            from models.algorithm_processor import LitAlgorithmProcessor

            pretrained_processor = LitAlgorithmProcessor.load_from_checkpoint(
                config["deq"]["align_target"], strict=False
            )
            self.normal_algorithm_module = pretrained_processor.algorithms[
                algorithm
            ].algorithm_module
            for p in self.normal_algorithm_module.parameters():
                p.requires_grad = False
        self.save_hyperparameters(ignore=["algo_processor"])

    @property
    def global_step(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        return self.trainer.global_step if self._trainer else self._global_step

    @global_step.setter
    def global_step(self, global_step) -> int:
        self._global_step = global_step

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        return self.trainer.current_epoch if self._trainer else self._current_epoch

    @current_epoch.setter
    def current_epoch(self, epoch) -> int:
        self._current_epoch = epoch

    def prepare_for_transfer(self):
        algo_processor = copy.deepcopy(self.algorithm_module.processor)
        self.algorithm_module = AlgorithmReasoner(
            self.hidden_dim,
            self.node_features,
            self.edge_features,
            self.output_features,
            algo_processor,
            use_TF=False,
            timeit=self.timeit,
            **self.algorithm_base_kwargs,
        )
        for p in self.algorithm_module.processor.parameters():
            p.requires_grad = False

    @staticmethod
    def pointer_loss(predecessor_pred, predecessor_gt_edge_1h, softmax_idx, num_nodes):
        loss_unreduced = cross_entropy(
            predecessor_pred, softmax_idx, predecessor_gt_edge_1h, num_nodes
        )
        sum_loss = loss_unreduced.flatten().sum()
        cnt_loss = predecessor_gt_edge_1h.count_nonzero()
        return sum_loss / cnt_loss

    def single_prediction_loss(
        self, name, pred, pred_gt, batch, graph_mask, node_mask, edge_mask
    ):
        loss = None
        stage, loc, data_type = self.dataset.spec[name]
        if loc == Location.GRAPH:
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[graph_mask], pred_gt[graph_mask].argmax(-1))
            if data_type == Type.SCALAR:
                loss = F.mse_loss(pred[graph_mask].squeeze(-1), pred_gt[graph_mask])
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                    pred[graph_mask].squeeze(-1), pred_gt[graph_mask]
                )
            if data_type == Type.POINTER:
                pred_gt_one_hot = torch.zeros_like(batch.pos)
                pred_gt_one_hot[pred_gt.long()] = 1
                loss = type(self).pointer_loss(
                    pred[node_mask].squeeze(-1),
                    pred_gt_one_hot[node_mask],
                    batch.batch[node_mask],
                    batch.num_nodes,
                )

        if loc == Location.NODE:
            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:
                pred_gt_one_hot = edge_one_hot_encode_pointers(pred_gt, batch)
                loss = type(self).pointer_loss(
                    pred[0][edge_mask],
                    pred_gt_one_hot[edge_mask],
                    batch.edge_index[0][edge_mask],
                    batch.num_nodes,
                )
                pos = (pred_gt_one_hot[edge_mask]).sum()
                neg = pred_gt_one_hot[edge_mask].shape[0]
                loss_inv = 0
                if self.config["inv_ptr"]:
                    loss_inv = F.binary_cross_entropy_with_logits(
                        pred[1][edge_mask], pred_gt_one_hot[edge_mask]
                    )
                loss = loss + loss_inv
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                    pred[node_mask].squeeze(-1), pred_gt[node_mask]
                )
            if data_type == Type.MASK_ONE:
                lsms = torch_scatter.scatter_log_softmax(
                    pred[node_mask], batch.batch[node_mask].unsqueeze(-1), dim=0
                )
                loss = (-lsms[(pred_gt[node_mask] == 1.0)]).mean()
            if data_type == Type.SCALAR:
                loss = F.mse_loss(pred[node_mask].squeeze(-1), pred_gt[node_mask])
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[node_mask], pred_gt[node_mask].argmax(-1))
        if loc == Location.EDGE:
            if data_type == Type.MASK:
                loss = F.binary_cross_entropy_with_logits(
                    pred[edge_mask].squeeze(-1), pred_gt[edge_mask]
                )
            if data_type == Type.CATEGORICAL:
                loss = F.cross_entropy(pred[edge_mask], pred_gt[edge_mask].argmax(-1))
            if data_type == Type.SCALAR:
                loss = F.mse_loss(pred[edge_mask].squeeze(-1), pred_gt[edge_mask])
            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                pred_gt = pred_gt.int() - starts_edge
                loss = F.cross_entropy(pred[edge_mask], pred_gt[edge_mask])
        assert loss is not None, f"{stage}/{name}/{loc}/{data_type}"
        return loss

    def get_prediction_loss(
        self,
        batch,
        all_hint_logits,
        output_logits,
        all_masks_graph,
        all_latent_states_pair,
        align=False,
        monotonise=False,
        loss_prefix="",
    ):

        losses_dict = defaultdict(list)
        if self.timeit:
            st = time.time()
        if self.config["algo_cls"] == "deq" and (align or monotonise):
            all_latent_states_DEQ, all_latent_states_normal = all_latent_states_pair

            # shapes are (num_iter, batch_size, max_num_nodes. embedding_dim)
            def compute_distance_matrix(
                batched_gnn_embeddings, batched_tgt_embeddings, num_nodes_per_graph
            ):
                # bring to (batch_size, mx_num_nodes, num_iter, embedding_dim))
                batched_gnn_embeddings = batched_gnn_embeddings.permute(1, 2, 0, 3)
                batched_tgt_embeddings = batched_tgt_embeddings.permute(1, 2, 0, 3)

                # compute distances
                dense_batch_ggn_embs = batched_gnn_embeddings.permute(
                    0, 2, 1, 3
                )  # shape (batch_size, num_gnn_terations, max_num_nodes, embedding_dim)
                dense_batch_tgt_embs = batched_tgt_embeddings.permute(
                    0, 2, 1, 3
                )  # shape (batch_size, num_algo_terations, max_num_nodes, embedding_dim)
                d = (
                    dense_batch_ggn_embs.unsqueeze(2)
                    - dense_batch_tgt_embs.unsqueeze(1)
                ).norm(
                    dim=-1
                )  # shape (batch_size, num_gnn_terations, num_DEQ_terations, max_num_nodes)

                # compute average distance across nodes manually to take into account that different graphs have different number of nodes
                d = d.sum(
                    dim=-1
                )  # shape (batch_size, num_gnn_terations, num_DEQ_terations)
                d = d / num_nodes_per_graph.unsqueeze(1).unsqueeze(1)
                return d

            num_nodes_per_graph = torch_scatter.scatter(
                torch.ones_like(batch.batch), batch.batch
            )
            batched_d_matrices = compute_distance_matrix(
                all_latent_states_DEQ, all_latent_states_normal, num_nodes_per_graph
            )
            batched_d_matrices_for_dp = compute_distance_matrix(
                all_latent_states_DEQ, all_latent_states_normal, num_nodes_per_graph
            )

            if align:

                def dynamic_programming_algo(batched_d_matrices):
                    batch_size = batched_d_matrices.shape[0]
                    max_value = batched_d_matrices.max().item()
                    dp = batched_d_matrices

                    def elementwise_softmin(tensor1, tensor2, temperature=0.5):
                        # Stack the tensors along a new dimension
                        stacked_tensors = torch.stack([tensor1, tensor2], dim=-1)
                        # Negate the stacked tensors and apply temperature
                        neg_stacked_tensors = -stacked_tensors / temperature
                        # Compute the softmax across the last dimension
                        softmax_weights = torch.nn.functional.softmax(
                            neg_stacked_tensors, dim=-1
                        )
                        # Compute the weighted sum along the last dimension
                        softmin_result = torch.sum(
                            softmax_weights * stacked_tensors, dim=-1
                        )
                        return softmin_result

                    def generate_masks(shape):
                        num_antidiagonals = shape[0] + shape[1] - 1
                        masks = []
                        for i in range(num_antidiagonals):
                            indices = torch.arange(max(shape))
                            valid_indices = (
                                (0 <= i - indices)
                                & (i - indices < shape[1])
                                & (indices < shape[0])
                            )
                            cur_mask = torch.zeros(shape)
                            cur_mask[
                                indices[valid_indices], i - indices[valid_indices]
                            ] = 1
                            masks.append(cur_mask)
                        return masks

                    masks = generate_masks(
                        (batched_d_matrices.shape[1], batched_d_matrices.shape[2])
                    )[1:]

                    leq_steps = dp.shape[1] <= dp.shape[2]
                    for dc, mask in enumerate(masks):
                        # add column with "infinity" at the beginning
                        # trim away last column
                        skips = torch.nn.functional.pad(dp, (1, 0, 0, 0), value=(1e9))[
                            :, :, :-1
                        ]  # dp i, j-1, dp i, 0 = inf
                        # add zero row at bottom
                        if leq_steps:
                            aligns_rhs = torch.nn.functional.pad(
                                dp, (1, 0, 0, 0)
                            )  # dp[i, j-1], dp[i, 0] = inf
                            aligns_rhs = torch.nn.functional.pad(
                                aligns_rhs, (0, 0, 1, 0)
                            )  # dp[i-1, j-1], dp[0, j] = 0, dp[0,0]=0
                            aligns_lhs = torch.nn.functional.pad(
                                batched_d_matrices, (0, 1, 0, 1)
                            )  # aligns dp and d
                            el1 = (aligns_lhs + aligns_rhs)[:, :-1, :-1]
                        else:
                            aligns_rhs = torch.nn.functional.pad(
                                dp, (0, 0, 1, 0)
                            )  # dp[i-1, j], dp[0, j] = 0
                            aligns_lhs = torch.nn.functional.pad(
                                batched_d_matrices, (0, 0, 0, 1)
                            )  # aligns dp and d
                            # sum and trim away the last row
                            el1 = (aligns_lhs + aligns_rhs)[:, :-1, :]

                        m = mask.unsqueeze(0).expand(batch_size, -1, -1)
                        dp[m.bool()] = torch.minimum(el1[m.bool()], skips[m.bool()])
                    return dp

                deq_steps = (self.algorithm_module.deq_info["nstep"] - 2).long()
                algo_steps = (batch.lengths - 2).long()
                one = torch.tensor(1, device=batch.edge_index.device)
                aranged = torch.arange(
                    batched_d_matrices.shape[0], device=batched_d_matrices.device
                )
                mones = torch.zeros(
                    (batched_d_matrices.shape[0], deq_steps.max() + one),
                    device=aranged.device,
                )
                mones[aranged, deq_steps] = 1
                mones = 1 - mones.cumsum(dim=-1)
                num_samples = ((deq_steps.min() - 1) // 10).maximum(one)
                rsidx, _ = torch.sort(torch.multinomial(mones, num_samples), dim=-1)
                rbatched_d_matrices = batched_d_matrices_for_dp[aranged[:, None], rsidx]
                rdeq_steps = torch.full_like(
                    deq_steps, rbatched_d_matrices.shape[1] - 1
                )
                rdp_solution = dynamic_programming_algo(rbatched_d_matrices)
                dpsolarr = rdp_solution[aranged, rdeq_steps, algo_steps]
                dpsolarr = dpsolarr / torch.maximum(
                    rdeq_steps + 1, torch.tensor(1.0, device=dpsolarr.device)
                )
                alignment_loss = dpsolarr.mean()
                alignment_loss = (
                    1 * alignment_loss
                    + batched_d_matrices[aranged, deq_steps, algo_steps].mean()
                )
                losses_dict["alignment"] = [
                    alignment_loss * self.config["deq"]["alignment_factor"]
                ]

            if monotonise:
                dtolast = batched_d_matrices[aranged, :, algo_steps]
                dtolastroll = torch.roll(dtolast, -1, 1)
                monoto = dtolastroll - dtolast
                monoto[:, -1] = 0
                monoto = monoto.maximum(torch.tensor(0, device=dtolast.device))
                mask = torch.zeros_like(monoto)
                mask[:, deq_steps] = 1
                mask = mask.cumsum(1) > 0
                monoto[mask] = torch.nan
                losses_dict["monotonicity"] = [
                    torch.nanmean(monoto) * self.config["deq"]["monoto_factor"]
                ]

        batch = self.algorithm_module.prepare_batch(batch)
        if self.config["supervise_hints"]:
            for i, (pred, graph_mask) in enumerate(
                zip(all_hint_logits, all_masks_graph)
            ):
                node_mask = graph_mask[batch.batch]
                edge_mask = node_mask[batch.edge_index[0]]
                assert graph_mask.any()
                for name in pred:
                    if "_inv" in name:  # NOTE Handled with the other
                        continue
                    pred_gt = batch[name][min(i + 1, batch[name].shape[0] - 1)]
                    losses_dict[name].append(
                        self.single_prediction_loss(
                            name,
                            pred[name],
                            pred_gt,
                            batch,
                            graph_mask,
                            node_mask,
                            edge_mask,
                        )
                    )
                    _, _, data_type = self.dataset.spec[name]

        def append_output_loss(ld, output_logits, graph_mask=None, prefix=""):
            for name in output_logits:
                if "_inv" in name:  # NOTE No need to convert to logits
                    continue
                if graph_mask is None:
                    graph_mask = torch.ones(
                        batch.num_graphs, dtype=torch.bool, device=self.device
                    )
                node_mask = graph_mask[batch.batch]
                edge_mask = node_mask[batch.edge_index[0]]
                ld[prefix + name].append(
                    self.single_prediction_loss(
                        name,
                        output_logits[name],
                        getattr(batch, name),
                        batch,
                        graph_mask,
                        node_mask,
                        edge_mask,
                    )
                )
            return ld

        losses_dict = append_output_loss(losses_dict, output_logits)

        for k, v in losses_dict.items():
            losses_dict[k] = torch.stack(v).mean()
        losses_dict = {(loss_prefix + k): v for k, v in losses_dict.items()}
        if self.timeit:
            print(f"loss calculation: {time.time()-st}")
            input()

        return losses_dict

    def single_prediction_acc(
        self, name, pred, pred_gt, batch, graph_mask, node_mask, edge_mask
    ):

        acc = None
        stage, loc, data_type = self.dataset.spec[name]
        if loc == Location.NODE:
            if data_type == Type.MASK_ONE:
                acc = (
                    (
                        pred[node_mask].squeeze(-1).nonzero()
                        == pred_gt[node_mask].nonzero()
                    )
                    .float()
                    .mean()
                )
            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:
                acc = (pred[node_mask].squeeze(-1) == pred_gt[node_mask]).float().mean()
            if data_type == Type.SCALAR:
                acc = ((pred[node_mask].squeeze(-1) - pred_gt[node_mask]) ** 2).mean()
            if data_type == Type.CATEGORICAL:
                acc = (
                    (pred[node_mask].argmax(-1) == pred_gt[node_mask].argmax(-1))
                    .float()
                    .mean()
                )
            if data_type == Type.MASK:
                acc = multiclass_f1_score(
                    pred[node_mask].squeeze(-1), pred_gt[node_mask]
                )

        if loc == Location.GRAPH:
            if data_type == Type.CATEGORICAL:
                acc = (
                    (pred[graph_mask].argmax(-1) == pred_gt[graph_mask].argmax(-1))
                    .float()
                    .mean()
                )
            if data_type == Type.SCALAR:
                acc = ((pred[graph_mask].squeeze(-1) - pred_gt[graph_mask]) ** 2).mean()
            if data_type == Type.MASK:
                acc = multiclass_f1_score(
                    pred[graph_mask].squeeze(-1), pred_gt[graph_mask]
                )
            if data_type == Type.POINTER:
                acc = (pred[graph_mask] == pred_gt[graph_mask]).float().mean()

        if loc == Location.EDGE:
            if data_type == Type.CATEGORICAL:
                acc = (
                    (pred[edge_mask].argmax(-1) == pred_gt[edge_mask].argmax(-1))
                    .float()
                    .mean()
                )
            if data_type == Type.MASK:
                acc = multiclass_f1_score(
                    pred[edge_mask].squeeze(-1), pred_gt[edge_mask]
                )
            if data_type == Type.SCALAR:
                acc = ((pred[edge_mask].squeeze(-1) - pred_gt[edge_mask]) ** 2).mean()
            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
                Type.MASK,
            ]:
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                pred_gt = pred_gt.int() - starts_edge
                acc = (pred[edge_mask] == pred_gt[edge_mask]).float().mean()
        assert acc is not None, f"Please implement {name}"
        return acc

    def get_metrics(
        self,
        batch,
        all_hint_logits,
        output_logits,
        all_masks_graph,
        all_continue_logits,
    ):

        batch = self.algorithm_module.prepare_batch(batch)
        accs_dict = defaultdict(list)

        for i, (pred, graph_mask) in enumerate(zip(all_hint_logits, all_masks_graph)):
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            outputs = type(self.algorithm_module).convert_logits_to_outputs(
                self.dataset.spec,
                {"hint": pred},
                batch.edge_index[0],
                batch.edge_index[1],
                batch.num_nodes,
                batch.batch,
                include_probabilities=False,
            )["hint"]

            for name in outputs:
                accs_dict[name].append(
                    self.single_prediction_acc(
                        name,
                        outputs[name],
                        batch[name][min(i + 1, batch[name].shape[0] - 1)],
                        batch,
                        graph_mask,
                        node_mask,
                        edge_mask,
                    )
                )

        outputs = type(self.algorithm_module).convert_logits_to_outputs(
            self.dataset.spec,
            output_logits,
            batch.edge_index[0],
            batch.edge_index[1],
            batch.num_nodes,
            batch.batch,
            include_probabilities=False,
        )["output"]
        for name in outputs:
            graph_mask = torch.ones(
                batch.num_graphs, dtype=torch.bool, device=self.device
            )
            node_mask = graph_mask[batch.batch]
            edge_mask = node_mask[batch.edge_index[0]]
            accs_dict[name].append(
                self.single_prediction_acc(
                    name,
                    outputs[name],
                    getattr(batch, name),
                    batch,
                    graph_mask,
                    node_mask,
                    edge_mask,
                )
            )

        for k, v in accs_dict.items():
            accs_dict[k] = torch.stack(v).mean()

        if self.config["learn_termination"]:
            termination_accs = []
            precision_accs = []
            recall_accs = []
            for step_idx, cls in enumerate(all_continue_logits):
                true_continuation = ~(step_idx + 1 >= batch.lengths - 1)
                already_terminated = step_idx >= batch.lengths - 1
                if already_terminated.all():
                    break
                acc = (
                    (
                        (cls[~already_terminated] > 0)
                        == true_continuation[~already_terminated]
                    )
                    .float()
                    .mean()
                )
                termination_accs.append(acc)
                precision = precision_score(
                    true_continuation[~already_terminated].cpu().numpy(),
                    (cls[~already_terminated] > 0).cpu().numpy(),
                    zero_division=numpy.nan,
                )
                precision_accs.append(precision)
                recall_accs.append(
                    recall_score(
                        true_continuation[~already_terminated].cpu().numpy(),
                        (cls[~already_terminated] > 0).cpu().numpy(),
                        zero_division=1.0,
                    )
                )
            accs_dict["continuation/accuracy"] = torch.tensor(termination_accs).mean()
            accs_dict["continuation/precision"] = (
                torch.tensor(precision_accs).nanmean()
                if not torch.isnan(torch.tensor(precision_accs)).all()
                else torch.tensor(0.0)
            )
            accs_dict["continuation/recall"] = torch.tensor(recall_accs).mean()

        return accs_dict

    def fwd_step(self, batch, batch_idx, hint_mode="encoded_decoded", algo_module=None):
        if self.timeit:
            st = time.time()
        if algo_module == None:
            algo_module = self.algorithm_module
        algo_module.epoch = self.current_epoch
        algo_module.global_step = self.global_step
        (
            all_hint_logits,
            output_logits,
            masks,
            all_continue_logits,
            all_latent_states,
        ) = algo_module.process(batch, hint_mode=hint_mode)
        if self.timeit:
            print(f"forward step: {time.time()-st}")
            input()
        return (
            algo_module.last_latent,
            all_hint_logits,
            output_logits,
            masks,
            all_continue_logits,
            all_latent_states,
        )

    def augment_data_obj(self, data, num_nodes_to_add=1):
        num_nodes_in_G = data.pos.shape[0]
        data = copy.deepcopy(data)
        if ALGO_TYPES[self.algorithm] == AlgoType.SORTING:

            def reverse_permutaion(perm):
                rv = torch.empty_like(perm)
                rv[perm] = torch.arange(perm.size(0), device=perm.device)
                return rv

            sk, ik = torch.sort(data.key, stable=True)
            rik = reverse_permutaion(ik)
            unqk = torch.unique(data.key)
            rands = torch.rand_like(data.key)
            while len(torch.unique(rands)) < len(unqk):
                rands = torch.rand_like(data.key)

            nk = torch.sort(rands)[0][rik]
            rorder1 = data.key[:, None] > data.key[None, :]
            rorder2 = nk[:, None] > nk[None, :]
            assert (rorder1 == rorder2).all(), (data.key, rands, nk)
            data.key = nk

        return data

    def augment_batch(self, batch):
        return torch_geometric.data.Batch.from_data_list(
            [self.augment_data_obj(dt) for dt in batch.to_data_list()],
            follow_batch=["edge_index"],
        )

    def augment_batch_cayley(self, batch):
        og_num_nodes = batch.num_nodes
        og_batch = batch.batch
        ogs = torch.bincount(og_batch)

        cayley_data = get_cayley_data(self.cayley_bank, ogs)

        cg_num_nodes = cayley_data.num_nodes
        cg_ptr = cayley_data.ptr.to(self.device)

        cg_ptr_ogs = cg_ptr[:-1] + ogs
        cg_ptr_ogs_mask = (
            cg_ptr_ogs[:-1] if cg_ptr_ogs[-1] == cg_num_nodes else cg_ptr_ogs
        )

        cayley_nodes = torch.zeros(cg_num_nodes).to(self.device)
        cayley_nodes[cg_ptr[:-1]] = -1
        cayley_nodes[cg_ptr_ogs_mask] += 1

        cayley_node_mask = (
            torch.ones(cg_num_nodes).to(self.device) + cayley_nodes.cumsum(-1)
        ).to(bool)

        og_nodes = torch.arange(og_num_nodes).to(self.device)
        new_cayley_nodes = torch.arange(og_num_nodes, cg_num_nodes).to(self.device)
        cayley_mask = torch.zeros(cg_num_nodes, dtype=torch.int64).to(self.device)
        cayley_mask[~cayley_node_mask] = og_nodes
        cayley_mask[cayley_node_mask] = new_cayley_nodes

        batch.cayley_num_nodes = cg_num_nodes
        batch.cayley_num_edges = cayley_data.num_edges
        batch.cayley_edge_index = torch.stack(
            (
                cayley_mask[cayley_data.edge_index[0]],
                cayley_mask[cayley_data.edge_index[1]],
            )
        )

    def get_contrastive_loss(self, hidden, hidden_aug, batch, batch_aug):
        hidden_dense, mask = to_dense_batch(hidden, batch=batch.batch)
        hidden_dense_aug, masks_aug = to_dense_batch(hidden_aug, batch=batch_aug.batch)
        sims_logits = torch.einsum("bid,bjd->bij", hidden_dense, hidden_dense_aug)
        sims_logits[~mask] = -1e9
        sims = torch.softmax(sims_logits, dim=-1)
        target = (
            torch.eye(sims.shape[1], sims.shape[2], device=sims.device)[None]
            .broadcast_to(sims.shape)
            .bool()
            & mask[..., None]
        )
        return (-torch.log(sims[target] + 1e-6)).mean()

    def get_termination_loss(self, all_continue_logits, batch):
        losses = []
        for step_idx, cls in enumerate(all_continue_logits):
            true_continuation = ~(step_idx + 1 >= batch.lengths - 1)
            already_terminated = step_idx >= batch.lengths - 1
            if already_terminated.all():
                break
            loss = F.binary_cross_entropy_with_logits(
                cls[~already_terminated], true_continuation[~already_terminated].float()
            )
            losses.append(loss)
        return torch.stack(losses).mean() if len(losses) else 0.0

    def logg(self, *args, **kwargs):
        if self._trainer:
            self.log(*args, **kwargs)

    def logg_dict(self, *args, **kwargs):
        if self._trainer:
            self.log_dict(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.config["contrastive_reg"]:
            batch_aug = self.augment_batch(batch)
            if self.config["use_expander"]:
                self.augment_batch_cayley(batch_aug)
            hidden_aug, all_hint_logits_aug, output_logits_aug, masks_aug, _, _ = (
                self.fwd_step(batch_aug, batch_idx, hint_mode="none")
            )
        if self.config["use_expander"]:
            self.augment_batch_cayley(batch)
        (
            hidden,
            all_hint_logits,
            output_logits,
            masks,
            all_continue_logits,
            all_latent_states,
        ) = self.fwd_step(batch, batch_idx, hint_mode=self.config["hint_mode"])
        all_latent_states_normal = None
        losses_dict = defaultdict(list)
        if self.config["algo_cls"] == "deq" and self.config["deq"]["align"]:
            (
                _,
                all_hint_logits_normal,
                output_logits_normal,
                masks_normal,
                _,
                all_latent_states_normal,
            ) = self.fwd_step(
                batch,
                batch_idx,
                hint_mode=self.config["hint_mode"],
                algo_module=self.normal_algorithm_module,
            )
            if self.config["deq"]["align_sv_on_normal"]:
                losses_dict.update(
                    self.get_prediction_loss(
                        batch,
                        all_hint_logits_normal,
                        output_logits_normal["output"],
                        masks_normal,
                        (all_latent_states_normal, all_latent_states_normal),
                        loss_prefix="normal_",
                    )
                )
        losses_dict.update(
            self.get_prediction_loss(
                batch,
                all_hint_logits,
                output_logits["output"],
                masks,
                (all_latent_states, all_latent_states_normal),
                align=self.config["deq"]["align"],
                monotonise=self.config["deq"]["monotonise"],
            )
        )
        if self.config["contrastive_reg"]:
            contrastive_loss = self.get_contrastive_loss(
                hidden, hidden_aug, batch, batch_aug
            )
            losses_dict["contrastive_reg"] = contrastive_loss
        if self.config["jac_reg"]:
            losses_dict["jac_loss"] = 0
        norms = {"abs": [], "rel": []}
        for i, (jro, jri) in enumerate(
            zip(self.algorithm_module.jac_reg_out, self.algorithm_module.jac_reg_in)
        ):
            if i == 1 and not self.config["update_edges_hidden"]:
                continue
            if self.config["jac_reg"]:
                losses_dict["jac_loss"] = losses_dict["jac_loss"] + self.config[
                    "jac_reg_tau"
                ] * jac_reg(jro, jri)
            norms["abs"].append((jro - jri).norm(1, dim=-1)[jro.sum(-1) != 0].mean())
            norms["rel"].append(
                ((jro - jri).norm(1, dim=-1) / (jro).norm(1, dim=-1)).nanmean()
            )
        if self.config["learn_termination"] and not self.config["algo_cls"] == "deq":
            losses_dict["termination"] = self.get_termination_loss(
                all_continue_logits, batch
            )
        self.logg_dict(
            dict((f"train/loss/{k}", v) for k, v in losses_dict.items()),
            batch_size=batch.num_graphs,
        )
        total_loss = sum(losses_dict.values())  # / len(losses_dict)
        self.logg(
            "train/loss/average_loss",
            total_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        accs_dict = {}
        accs_dict["L^1-norm"] = torch.tensor(norms["abs"]).mean()
        accs_dict["rel L^1-norm"] = torch.tensor(norms["rel"]).mean()
        accs_dict["avg. # steps real"] = batch.lengths.mean()
        accs_dict["max # steps real"] = batch.lengths.max()
        if hasattr(self.algorithm_module, "deq_info"):
            accs_dict["avg. # steps solver"] = self.algorithm_module.deq_info[
                "nstep"
            ].mean()
            accs_dict["max # steps solver"] = self.algorithm_module.deq_info[
                "nstep"
            ].max()
        if self.current_epoch % self.test_train_every_n_epoch == 0:
            accs_dict |= self.get_metrics(
                batch, all_hint_logits, output_logits, masks, all_continue_logits
            )
        self.logg_dict(
            dict((f"train/acc/{k}", v) for k, v in accs_dict.items()),
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
        )
        if sum(losses_dict.values()) > 1e3:
            breakpoint()
        return {"loss": total_loss, "losses_dict": losses_dict, "accuracies": accs_dict}

    def valtest_step(self, batch, batch_idx, mode):
        if self.config["contrastive_reg"]:
            batch_aug = self.augment_batch(batch)
            if self.config["use_expander"]:
                self.augment_batch_cayley(batch_aug)
            hidden_aug, all_hint_logits_aug, output_logits_aug, masks_aug, _, _ = (
                self.fwd_step(batch_aug, batch_idx, hint_mode="none")
            )
        if self.config["use_expander"]:
            self.augment_batch_cayley(batch)
        (
            hidden,
            all_hint_logits,
            output_logits,
            masks,
            all_continue_logits,
            all_latent_states,
        ) = self.fwd_step(batch, batch_idx, hint_mode=self.config["hint_mode"])
        all_latent_states_normal = None
        if self.config["algo_cls"] == "deq" and self.config["deq"]["align"]:
            _, _, _, masks_normal, _, all_latent_states_normal = self.fwd_step(
                batch,
                batch_idx,
                hint_mode=self.config["hint_mode"],
                algo_module=self.normal_algorithm_module,
            )
        losses_dict = self.get_prediction_loss(
            batch,
            all_hint_logits,
            output_logits["output"],
            masks,
            (all_latent_states, all_latent_states_normal),
        )
        if self.config["contrastive_reg"]:
            contrastive_loss = self.get_contrastive_loss(
                hidden, hidden_aug, batch, batch_aug
            )
            losses_dict["contrastive_reg"] = contrastive_loss
        if self.config["learn_termination"] and not self.config["algo_cls"] == "deq":
            losses_dict["termination"] = self.get_termination_loss(
                all_continue_logits, batch
            )
        self.logg_dict(
            dict((f"{mode}/loss/{k}", v) for k, v in losses_dict.items()),
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
        )
        if torch.isnan(sum(losses_dict.values())).any():
            breakpoint()
        self.logg(
            f"{mode}/loss/average_loss",
            sum(losses_dict.values()),
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
        )
        accs_dict = self.get_metrics(
            batch, all_hint_logits, output_logits, masks, all_continue_logits
        )
        accs_dict["avg. # steps real"] = batch.lengths.mean()
        accs_dict["max # steps real"] = batch.lengths.max()
        if hasattr(self.algorithm_module, "deq_info"):
            accs_dict["avg. # steps solver"] = self.algorithm_module.deq_info[
                "nstep"
            ].mean()
            accs_dict["max # steps solver"] = self.algorithm_module.deq_info[
                "nstep"
            ].max()
        self.logg_dict(
            dict((f"{mode}/acc/{k}", v) for k, v in accs_dict.items()),
            batch_size=batch.num_graphs,
            add_dataloader_idx=False,
        )
        return {"losses": losses_dict, "accuracies": accs_dict}

    def validation_step_alt(self, batch, batch_idx, dataloader_idx):
        if (
            dataloader_idx == 1
            and not self.trainer.state.stage == "sanity_check"
            and self.current_epoch % self.test_with_val_every_n_epoch == 0
        ):
            return self.valtest_step(batch, batch_idx, "periodic_test")
        if dataloader_idx == 0:
            return self.valtest_step(batch, batch_idx, "val")

    def validation_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.valtest_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.fwd_step(batch, batch_idx, hint_mode=self.config["hint_mode"])

    def load_dataset(self, split, suffix=""):
        split = split + suffix
        nn = CONFIGS[self.algorithm][split]["num_nodes"]
        self.dataset_kwargs["split"] = split
        if (split, nn) not in self._datasets:
            self._datasets[(split, nn)] = self.dataset_class(
                self.dataset_root,
                nn,
                CONFIGS[self.algorithm][split]["num_samples"],
                algorithm=self.algorithm,
                **self.dataset_kwargs,
            )
        self.dataset = self._datasets[(split, nn)]
        print(f"Loading {self.dataset=} (num nodes: {nn}) with kwargs")
        pprint(self.dataset_kwargs)
        print()

    def get_a_loader(self, split, suffix=""):
        self.load_dataset(split, suffix="")
        self.algorithm_module.dataset_spec = self.dataset.spec
        dl = DataLoader(
            self.dataset,
            batch_size=get_hyperparameters()["batch_size"],
            shuffle=True if split == "train" else False,
            drop_last=False,
            follow_batch=["edge_index"],
            num_workers=5,
            persistent_workers=True,
        )
        return dl

    def train_dataloader(self):
        return self.get_a_loader("train")

    def val_dataloader_alt(self):
        return [self.get_a_loader("val"), self.get_a_loader("test")]

    def val_dataloader(self):
        return self.get_a_loader("val")

    def test_dataloader(self, suffix=""):
        return self.get_a_loader("test" + suffix)

    def configure_optimizers(self):
        lr = self.learning_rate
        wd = self.weight_decay
        opter = optim.AdamW if wd != 0.0 else optim.Adam
        optimizer = opter(self.parameters(), weight_decay=wd, lr=lr)
        return optimizer


if __name__ == "__main__":
    ...
