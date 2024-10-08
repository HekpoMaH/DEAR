import time
import absl
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch_scatter
from torchdeq import get_deq

from utils_execution import (
    check_edge_index_sorted,
    prepare_constants,
    edge_one_hot_encode_pointers,
)
from clrs import Type, Location, Stage
from layers.predinet import PrediNet
from layers.algorithm_reasoner import AlgorithmReasoner


class DEQReasoner(AlgorithmReasoner):
    def __init__(
        self,
        spec,
        data,
        latent_features,
        *args,
        hint_annotation=None,
        bias=True,
        **kwargs,
    ):
        super().__init__(spec, data, latent_features, bias=bias, *args, **kwargs)
        self.deqs = []
        for slvr in self.config["deq"]["solvers"]:
            self.deqs.append(
                get_deq(
                    f_max_iter=self.config["deq"]["deq_max_iter"],
                    b_max_iter=self.config["deq"]["deq_max_iter"],
                    f_solver=slvr,
                    b_solver=slvr,
                    f_stop_mode=self.config["deq"]["stop_mode"],
                    b_stop_mode=self.config["deq"]["stop_mode"],
                    f_tol=self.config["deq"]["f_tol"],
                    b_tol=self.config["deq"]["b_tol"],
                    eval_factor=self.config["deq"]["eval_factor"],
                )
            )

    def old_process(self, *args, **kwargs):
        return super().process(*args, **kwargs, forward_fn=super().forward)

    def process(
        self,
        batch,
        EPSILON=0,
        enforced_mask=None,
        hardcode_outputs=False,
        debug=False,
        first_n_processors=1000,
        init_last_latent=None,
        hint_mode="none",
        **kwargs,
    ):

        SIZE, STEPS_SIZE = prepare_constants(batch)
        self.hardcode_outputs = hardcode_outputs

        # Pytorch Geometric batches along the node dimension, but we execute
        # along the temporal (step) dimension, hence we need to transpose
        # a few tensors. Done by `prepare_batch`.
        if self.assert_checks:
            check_edge_index_sorted(batch.edge_index)
        if self.epoch > self.debug_epoch_threshold:
            breakpoint()
        self.zero_steps()
        batch = type(self).prepare_batch(batch)
        # When we want to calculate last step metrics/accuracies
        # we need to take into account again different termination per graph
        # hence we save last step tensors (e.g. outputs) into their
        # corresponding tensor. The function below prepares these tensors
        # (all set to zeros, except masking for computation, which are ones)
        self.set_initial_states(batch, init_last_latent=init_last_latent)
        # Prepare masking tensors (each graph does at least 1 iteration of the algo)
        self.prepare_initial_masks(batch)
        # A flag if we had a wrong graph in the batch. Used for visualisation
        # of what went wrong
        self.wrong_flag = False
        assert self.mask_cp.all(), self.mask_cp
        if self.timeit:
            st = time.time()
        node_fts_inp, edge_fts_inp, graph_fts_inp = self.encode_inputs(batch)
        node_fts = node_fts_inp
        edge_fts = edge_fts_inp
        graph_fts = graph_fts_inp
        if self.timeit:
            print(f"encoding inputs: {time.time()-st}")

        _, mask = to_dense_batch(node_fts, batch.batch)  # B x NMAX
        mask_edges = to_dense_adj(batch.edge_index, batch=batch.batch).bool()
        if self.config["update_edges_hidden"]:

            def fntocall(hidden):
                hsparse = hidden[mask]
                hidden, edges_hidden = self.forward(
                    batch,
                    node_fts,
                    edge_fts,
                    graph_fts,
                    hsparse,
                    self.last_latent_edges,
                    first_n_processors=first_n_processors,
                )
                hidden = hidden + torch.randn_like(hidden) * 0.001
                hidden = to_dense_batch(hidden, batch.batch)[0]
                edges_hidden = to_dense_adj(
                    batch.edge_index, batch=batch.batch, edge_attr=edges_hidden
                )
                if "decay_hidden" in self.config:
                    hidden = hidden * self.config["decay_hidden"]
                    edges_hidden = edges_hidden * self.config["decay_hidden"]
                return (hidden,)

        else:

            def fntocall(hidden):
                hsparse = hidden[mask]
                hidden, edges_hidden = self.forward(
                    batch,
                    node_fts,
                    edge_fts,
                    graph_fts,
                    hsparse,
                    self.last_latent_edges,
                    first_n_processors=first_n_processors,
                )
                hidden = hidden + torch.randn_like(hidden) * 0.001
                hidden = to_dense_batch(hidden, batch.batch)[0]
                if "decay_hidden" in self.config:
                    hidden = hidden * self.config["decay_hidden"]
                return (hidden,)

        arg1 = to_dense_batch(self.last_latent, batch.batch)[0]
        arg2 = to_dense_adj(
            batch.edge_index, batch=batch.batch, edge_attr=self.last_latent_edges
        )
        arglist = [arg1]
        if self.training:
            chosen = torch.randint(len(self.deqs), size=(1,))
            deq = self.deqs[chosen]
        else:
            deq = self.deqs[0]
        z_out, info = deq(
            fntocall,
            tuple(arglist),
            solver_kwargs={"first_hit_under_tol": True, "return_all": True},
        )
        hidden = z_out[-1][0]
        hiddentoret = hidden.clone()
        trajectory = info["trajectory"]
        trajectory = [
            traj[:, : hidden.shape[1] * hidden.shape[2]].view(
                traj.shape[0], -1, self.latent_features
            )
            for traj in trajectory
        ]  # List[B x Nmax x H]
        trajectory = torch.stack(trajectory, dim=0)
        edge_fts_decode = edge_fts
        self.extra_outs = []
        if self.config["deq"].get("stochastic", False) and self.training:
            trajectory = F.pad(trajectory, (0, 0, 0, 0, 0, 0, 0, 9))
            aranged = torch.arange(trajectory.shape[1], device=trajectory.device)
            alpha = 0.998**self.global_step * self.config["deq"]["stochastic_alpha"]
            conts = torch.full((batch.num_graphs,), alpha, device=aranged.device)
            conts = torch.bernoulli(conts).bool()
            cnt = 0
            upped = torch.zeros(batch.num_graphs, device=trajectory.device)
            while conts.any() and cnt < 9:
                arglist = [hidden]
                z_star = fntocall(*arglist)
                hidden[conts] = z_star[0][conts]
                edges_hidden[conts[batch.edge_index_batch]] = (
                    z_star[1][conts[batch.edge_index_batch]]
                    if self.config["update_edges_hidden"]
                    else self.last_latent_edges[conts[batch.edge_index_batch]]
                )
                self.extra_outs.append(
                    (
                        self.decode(
                            batch,
                            node_fts,
                            hidden[mask],
                            edge_fts_decode,
                            graph_fts,
                            hint_mode=hint_mode,
                        )["output"],
                        conts,
                    )
                )
                trajectory[(info["nstep"].long() - 1 + cnt)[conts], aranged[conts]] = (
                    hidden[conts]
                )
                upped[conts] += 1
                conts_upds = torch.full(conts.shape, alpha, device=conts.device)
                conts_upds = torch.bernoulli(conts_upds)
                conts = conts & conts_upds.bool()
                cnt += 1
            info["nstep"] += upped
        if not self.config["update_edges_hidden"]:
            edges_hidden = self.last_latent_edges
        else:
            hsparse = hidden[mask]
            _, edges_hidden = self.forward(
                batch,
                node_fts,
                edge_fts,
                graph_fts,
                hsparse,
                self.last_latent_edges,
                first_n_processors=first_n_processors,
            )
            edges_hidden_arg_list = to_dense_adj(
                batch.edge_index, batch=batch.batch, edge_attr=edges_hidden
            )

        arglist = [hidden]
        z_star = fntocall(*arglist)
        hidden_star = z_star[0]
        hidden = hidden[mask]
        hidden_star = hidden_star[mask]
        if self.config["update_edges_hidden"]:
            edge_fts_decode = torch.cat((edge_fts, edges_hidden), -1)
        outs = self.decode(
            batch, node_fts, hidden, edge_fts_decode, graph_fts, hint_mode=hint_mode
        )
        self.update_states(
            batch,
            hidden,
            edges_hidden,
            outs,
            torch.ones_like(self.last_continue_logits),
        )
        self.jac_reg_in = (arglist[0],)
        self.jac_reg_out = (fntocall(*arglist)[0],)
        self.deq_info = info
        return [], self.last_logits, self.all_masks_graph, [], trajectory

    def forward(
        self,
        batch,
        node_fts,
        edge_fts,
        graph_fts,
        hidden,
        edges_hidden,
        first_n_processors=1000,
    ):
        if torch.isnan(node_fts).any():
            breakpoint()
        assert not torch.isnan(hidden).any()
        assert not torch.isnan(node_fts).any()
        if self.timeit:
            st = time.time()
        if self.timeit:
            print(f"projecting nodes: {time.time()-st}")

        if self.timeit:
            st = time.time()
        edge_index = batch.edge_index
        hidden, edges_hidden, _ = self.call_processor(
            node_fts,
            edge_fts,
            graph_fts,
            hidden,
            edges_hidden,
            batch,
            edge_index,
            first_n_processors=first_n_processors,
        )
        if self.timeit:
            print(f"message passing: {time.time()-st}")
        assert not torch.isnan(hidden).any()
        if self.timeit:
            st = time.time()
        return hidden, edges_hidden
