import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_geometric
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch_scatter

from utils_execution import (
    check_edge_index_sorted,
    prepare_constants,
    edge_one_hot_encode_pointers,
    edge_one_hot_encode_pointers_edge,
)
from clrs import Type, Location, Stage
from layers.predinet import PrediNet


def sinkhorn_normalize(batch, y, temperature, steps=10, add_noise=False):

    Inf = 1e6
    from_, to = batch.edge_index[0], batch.edge_index[1]

    if add_noise:
        eps = -torch.log(-torch.log(torch.rand_like(y) + 1e-12) + 1e-12)
        y = y + eps

    y = y / temperature
    y = y.masked_fill(from_ == to, -Inf)

    for _ in range(steps):
        y = torch_scatter.scatter_log_softmax(y, from_, dim_size=batch.num_nodes)
        y = torch_scatter.scatter_log_softmax(y, to, dim_size=batch.num_nodes)

    return y


class AlgorithmReasoner(nn.Module):
    @classmethod
    def prepare_batch(cls, batch):
        batch = batch.clone()
        for name, tensor in batch.items():
            if not torch.is_tensor(tensor):
                continue
            if name.endswith("_temporal") and "index" not in name:
                tensor = tensor.transpose(1, 0)
                batch[name] = tensor
        return batch

    @classmethod
    def get_masks(cls, train, batch, continue_logits, enforced_mask):
        mask = continue_logits[batch.batch] > 0
        mask_cp = (continue_logits > 0.0).bool()
        mask_edges = mask[batch.edge_index[0]]
        if not train and enforced_mask is not None:
            enforced_mask_ids = enforced_mask[batch.batch]
            mask &= enforced_mask_ids
            mask_cp &= enforced_mask
        return mask_cp, mask, mask_edges

    def add_encoder(
        self, stage, name, loc, data_type, data_sample, bias, drop_pos=False
    ):
        if name == "adj":  # we use edge indices
            return
        if name == "pos" and drop_pos:
            return
        if (
            data_type == Type.SCALAR
            or data_type == Type.MASK
            or data_type == Type.MASK_ONE
        ):
            self.encoders[stage][name] = nn.Linear(1, self.latent_features, bias=bias)

        if data_type == Type.CATEGORICAL:
            in_shape = data_sample.shape[-1]
            self.encoders[stage][name] = nn.Linear(
                in_shape, self.latent_features, bias=bias
            )

        if loc == Location.NODE and data_type in [
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ]:  # pointers are 1-hot encoded on the edges
            self.encoders[stage][name] = nn.Linear(1, self.latent_features, bias=bias)
            self.encoders[stage][name + "_inv"] = nn.Linear(
                1, self.latent_features, bias=bias
            )
        if loc == Location.EDGE and data_type in [
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ]:  # pointers are 1-hot encoded on the edges
            self.encoders[stage][name] = nn.ModuleList(
                [
                    nn.Linear(1, self.latent_features, bias=bias),
                    nn.Linear(1, self.latent_features, bias=bias),
                ]
            )

    def add_decoder(self, stage, name, loc, data_type, data_sample, bias):
        assert name != "adj", "Adjacency matrix should not be decoded"
        if loc == Location.NODE:
            if data_type in (Type.SCALAR, Type.MASK, Type.MASK_ONE):
                self.decoders[stage][name] = nn.Linear(
                    2 * self.latent_features, 1, bias=bias
                )

            if data_type == Type.CATEGORICAL:
                in_shape = data_sample.shape[-1]
                self.decoders[stage][name] = nn.Linear(
                    2 * self.latent_features, in_shape, bias=bias
                )

            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:  # pointers are decoded from both node and edge information
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            (
                                2 * self.latent_features
                                if self.config["update_edges_hidden"]
                                else self.latent_features
                            ),
                            self.latent_features,
                            bias=bias,
                        ),
                        nn.Linear(self.latent_features, 1, bias=bias),
                    ]
                )
                self.decoders[stage][name + "_inv"] = nn.ModuleList(
                    [
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            (
                                2 * self.latent_features
                                if self.config["update_edges_hidden"]
                                else self.latent_features
                            ),
                            self.latent_features,
                            bias=bias,
                        ),
                        nn.Linear(self.latent_features, 1, bias=bias),
                    ]
                )
        if loc == Location.GRAPH:
            if data_type in [Type.MASK, Type.SCALAR, Type.CATEGORICAL, Type.MASK_ONE]:
                in_shape = data_sample.shape[-1] if data_type == Type.CATEGORICAL else 1
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(2 * self.latent_features, in_shape, bias=bias),
                        nn.Linear(self.latent_features, in_shape, bias=bias),
                    ]
                )
            if data_type in [Type.POINTER]:
                in_shape = data_sample.shape[-1] if data_type == Type.CATEGORICAL else 1
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(2 * self.latent_features, in_shape, bias=bias),
                        nn.Linear(self.latent_features, in_shape, bias=bias),
                        nn.Linear(2 * self.latent_features, in_shape, bias=bias),
                    ]
                )

        if loc == Location.EDGE:
            if data_type in (Type.SCALAR, Type.MASK, Type.MASK_ONE):
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(2 * self.latent_features, 1, bias=bias),
                        nn.Linear(2 * self.latent_features, 1, bias=bias),
                        nn.Linear(
                            (
                                2 * self.latent_features
                                if self.config["update_edges_hidden"]
                                else self.latent_features
                            ),
                            1,
                            bias=bias,
                        ),
                    ]
                )
            if data_type == Type.CATEGORICAL:
                in_shape = data_sample.shape[-1]
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(2 * self.latent_features, in_shape, bias=bias),
                        nn.Linear(2 * self.latent_features, in_shape, bias=bias),
                        nn.Linear(self.latent_features, in_shape, bias=bias),
                    ]
                )
            if data_type == Type.POINTER:
                self.decoders[stage][name] = nn.ModuleList(
                    [
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(
                            (
                                2 * self.latent_features
                                if self.config["update_edges_hidden"]
                                else self.latent_features
                            ),
                            self.latent_features,
                            bias=bias,
                        ),
                        nn.Linear(
                            2 * self.latent_features, self.latent_features, bias=bias
                        ),
                        nn.Linear(self.latent_features, 1, bias=bias),
                    ]
                )

    def __init__(
        self,
        spec,
        data,
        latent_features,
        algo_processor,
        bias=True,
        use_TF=False,
        use_sinkhorn=True,
        L1_loss=False,
        xavier_on_scalars=True,
        get_attention=False,
        use_batch_norm=False,
        transferring=False,
        timeit=True,
        drop_pos=False,
        config=None,
        **kwargs,
    ):

        super().__init__()
        self.step_idx = 0
        self.latent_features = latent_features
        self.assert_checks = False
        self.timeit = timeit
        self.debug = False
        self.debug_epoch_threshold = 1e9
        self.L1_loss = L1_loss
        self.global_pool = config["global_pool"]
        self.next_step_pool = True
        self.processor = algo_processor
        self.use_TF = use_TF
        self.use_sinkhorn = use_sinkhorn
        self.get_attention = get_attention
        self.lambda_mul = 1  # 0.0001
        self.transferring = transferring
        self.dataset_spec = spec
        self.drop_pos = drop_pos
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                "input": nn.ModuleDict({}),
                "hint": nn.ModuleDict({}),
            }
        )
        self.decoders = nn.ModuleDict(
            {"hint": nn.ModuleDict({}), "output": nn.ModuleDict({})}
        )
        for name, (stage, loc, datatype) in spec.items():
            if name == "adj":  # we use edge indices
                continue
            if stage == "input":
                self.add_encoder(
                    stage,
                    name,
                    loc,
                    datatype,
                    getattr(data, name),
                    bias,
                    drop_pos=drop_pos,
                )
            if stage == "output":
                self.add_decoder(stage, name, loc, datatype, getattr(data, name), bias)
            if stage == "hint":
                self.add_encoder(stage, name, loc, datatype, getattr(data, name), bias)
                self.add_decoder(stage, name, loc, datatype, getattr(data, name), bias)

        if xavier_on_scalars:
            assert False, "NEEDS REFACTORING"
            torch.nn.init.trunc_normal_(
                self.encoders["input"]["edge_attr"].weight,
                std=1 / torch.sqrt(torch.tensor(latent_features)),
            )

        if config["use_expander"]:
            self.cgp_processor = (
                copy.deepcopy(algo_processor)
                if config["use_expander_sepproc"]
                else algo_processor
            )
            self.cayley_edge_fts = nn.Parameter(torch.randn(latent_features))

        self.add_pooling(
            latent_features, self.global_pool, bias=bias, use_batch_norm=use_batch_norm
        )

    def add_pooling(
        self, latent_features, global_pool, bias=True, use_batch_norm=False
    ):
        if global_pool == "attention":
            inp_dim = latent_features
            self.pooler = GlobalAttentionPlusCoef(
                nn.Sequential(
                    nn.Linear(inp_dim, latent_features, bias=bias),
                    nn.LeakyReLU(),
                    nn.Linear(latent_features, 1, bias=bias),
                ),
                nn=None,
            )

        if global_pool == "predinet":
            lf = latent_features
            self.pooler = PrediNet(
                lf, 1, lf, lf, flatten_pooling=torch_geometric.nn.glob.global_max_pool
            )

        if global_pool == "max":
            self.pooler = torch_geometric.nn.global_max_pool

        self.termination_network = nn.Sequential(
            nn.BatchNorm1d(latent_features) if use_batch_norm else nn.Identity(),
            nn.Linear(latent_features, 1, bias=bias),
        )

    def get_module_state(self, latents, batch_ids):
        return self.pooler(latents, batch_ids)

    def get_continue_logits(self, batch, latent_nodes, sth_else=None):
        batch_ids = batch.batch
        graph_latent = self.get_module_state(latent_nodes, batch_ids)
        if self.global_pool == "attention":
            graph_latent, coef = self.pooler(latent_nodes, batch_ids)
            if self.get_attention:
                self.attentions[self.step_idx] = coef.clone().detach()
                self.per_step_latent[self.step_idx] = sth_else

        if self.get_attention:
            self.attentions[self.step_idx] = latent_nodes
        continue_logits = self.termination_network(graph_latent).view(-1)
        return continue_logits

    def zero_termination(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def zero_steps(self):
        self.sum_of_processed_nodes = 0
        self.sum_of_processed_edges = 0
        self.step_idx = 0
        self.sum_of_steps = 0
        self.cnt = 0

    @classmethod
    def convert_logits_to_outputs(
        cls,
        spec,
        logits,
        fr,
        to,
        num_nodes,
        batch_ids,
        include_probabilities=True,
        dbg=False,
    ):
        outs = defaultdict(dict)

        for stage in logits.keys():
            for name in logits[stage].keys():
                stage, loc, data_type = spec[name]
                assert stage != Stage.INPUT
                if data_type == Type.SOFT_POINTER:
                    assert False, f"Not yet added, please add {name}"
                if data_type in [Type.CATEGORICAL]:
                    indices = logits[stage][name].argmax(-1)
                    outshape = logits[stage][name].shape[-1]
                    outs[stage][name] = F.one_hot(indices, num_classes=outshape).float()
                if data_type == Type.MASK_ONE:
                    _, amax = torch_scatter.scatter_max(
                        logits[stage][name], batch_ids, dim=0
                    )
                    amax = amax.squeeze(-1)
                    outs[stage][name] = torch.zeros_like(logits[stage][name])
                    outs[stage][name][amax] = 1
                if data_type == Type.MASK:
                    outs[stage][name] = (logits[stage][name] > 0).float()
                if data_type == Type.SCALAR:
                    outs[stage][name] = logits[stage][name]
                if loc == Location.NODE and data_type in [
                    Type.POINTER,
                    Type.PERMUTATION_POINTER,
                    Type.SHOULD_BE_PERMUTATION,
                ]:
                    outs[stage][name] = []
                    pointer_logits = logits[stage][name][0]
                    _, pointers = torch_scatter.scatter_max(
                        pointer_logits, fr, dim_size=num_nodes
                    )
                    pointers = to[pointers]
                    pointer_probabilities = torch_geometric.utils.softmax(
                        pointer_logits, fr, num_nodes=num_nodes
                    )
                    outs[stage][name] = pointers.long()
                    if include_probabilities:
                        outs[stage][f"{name}_probabilities"] = pointer_probabilities
                if loc == Location.EDGE and data_type in [
                    Type.POINTER,
                    Type.PERMUTATION_POINTER,
                    Type.SHOULD_BE_PERMUTATION,
                ]:
                    pointer_logits = logits[stage][name]
                    pointers = pointer_logits.argmax(-1)
                    pointer_probabilities = F.softmax(pointer_logits, dim=-1)
                    outs[stage][name] = pointers.int()
                    if include_probabilities:
                        outs[stage][f"{name}_probabilities"] = pointer_probabilities
                if loc == Location.GRAPH and data_type in [Type.POINTER]:
                    pointer_logits = logits[stage][name]
                    _, pointers = torch_scatter.scatter_max(
                        pointer_logits.squeeze(-1), batch_ids
                    )
                    outs[stage][name] = pointers.int()
        return outs

    def set_initial_states(self, batch, init_last_latent=None):
        self.processor.zero_lstm(
            batch.num_nodes
        )  # NO-OP if processor(s) don't use LSTM
        self.last_latent = torch.zeros(
            batch.num_nodes, self.latent_features, device=batch.edge_index.device
        )
        if self.config["use_expander"]:
            self.last_latent_cayley = torch.zeros(
                batch.cayley_num_nodes,
                self.latent_features,
                device=batch.edge_index.device,
            )

        if init_last_latent is not None:
            self.last_latent = init_last_latent
        self.last_latent_edges = torch.zeros(
            batch.num_edges, self.latent_features, device=batch.edge_index.device
        )
        self.last_continue_logits = torch.ones(
            batch.num_graphs, device=batch.edge_index.device
        )
        self.last_logits = defaultdict(dict)

        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage == Stage.INPUT:
                continue
            if stage == Stage.OUTPUT:
                if loc in [Location.NODE, Location.GRAPH]:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)
                    if data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name).unsqueeze(
                            -1
                        )
                    if data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(
                            getattr(batch, name).bool(), 1e9, -1e9
                        ).unsqueeze(-1)
                    if loc == Location.NODE and data_type in [
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        self.last_logits[stage][name] = (
                            torch.where(
                                batch.edge_index[0, :] == batch.edge_index[1, :],
                                1e9,
                                -1e9,
                            ).to(
                                batch.edge_index.device
                            ),  # self-loops
                            torch.where(
                                batch.edge_index[0, :] == batch.edge_index[1, :],
                                1e9,
                                -1e9,
                            ).to(batch.edge_index.device),
                        )  # self-loops
                    else:
                        ll = torch.zeros_like(batch.pos)
                        ll[getattr(batch, name).long()] = 1
                        self.last_logits[stage][name] = torch.where(
                            ll.bool(), 1e9, -1e9
                        ).unsqueeze(-1)

                if loc == Location.EDGE:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)
                    elif data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(
                            getattr(batch, name).bool(), 1e9, -1e9
                        ).unsqueeze(-1)
                    elif data_type in [
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        self.max_nodes_in_graph = (
                            torch.bincount(batch.batch).max().item()
                        )
                        ptrs = getattr(batch, name).int()
                        starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                        ptrs = ptrs - starts_edge
                        self.last_logits[stage][name] = torch.full(
                            (batch.edge_index.shape[1], self.max_nodes_in_graph), -1e9
                        ).to(batch.edge_index.device)
                        self.last_logits[stage][name][
                            torch.arange(ptrs.shape[0]), ptrs
                        ] = 1e9
                    else:
                        assert False, breakpoint()

            if stage == Stage.HINT:

                if loc in [Location.NODE, Location.GRAPH]:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)[0]
                    if data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name)[
                            0
                        ].unsqueeze(-1)
                    if data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(
                            getattr(batch, name)[0, :].bool(), 1e9, -1e9
                        ).unsqueeze(-1)
                    if data_type in [
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        ptr = getattr(batch, name)[0]
                        msk = edge_one_hot_encode_pointers(ptr, batch)
                        msk_inv = edge_one_hot_encode_pointers(ptr, batch, inv=True)
                        self.last_logits[stage][name] = (
                            torch.where(msk.bool(), 1e9, -1e9).to(
                                batch.edge_index.device
                            ),  # self-loops
                            torch.where(msk_inv.bool(), 1e9, -1e9).to(
                                batch.edge_index.device
                            ),
                        )  # self-loops

                if loc == Location.EDGE:
                    if data_type == Type.CATEGORICAL:
                        self.last_logits[stage][name] = getattr(batch, name)[0]
                    elif data_type in [Type.MASK, Type.MASK_ONE]:
                        self.last_logits[stage][name] = torch.where(
                            getattr(batch, name)[0, :].bool(), 1e9, -1e9
                        ).unsqueeze(-1)
                    elif data_type == Type.SCALAR:
                        self.last_logits[stage][name] = getattr(batch, name)[
                            0, :
                        ].unsqueeze(-1)
                    elif data_type in [
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        ptrs = getattr(batch, name)[0, :].int()
                        starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                        ptrs = ptrs - starts_edge
                        self.max_nodes_in_graph = (
                            torch.bincount(batch.batch).max().item()
                        )
                        self.last_logits[stage][name] = torch.where(
                            edge_one_hot_encode_pointers_edge(
                                ptrs, batch, self.max_nodes_in_graph
                            ).bool(),
                            1e9,
                            -1e9,
                        ).to(batch.edge_index.device)
                    else:
                        assert False, breakpoint()

        self.all_hint_logits = []
        self.all_masks_graph = []
        self.all_continue_logits = []

    def update_per_mask(self, before, after, mask=None):
        # NOTE: this does expansion of the mask, if you do
        # NOT use expansion, use torch.where
        if mask is None:
            mask = self.mask

        mask = mask.unsqueeze(-1).expand_as(before)
        return torch.where(mask, after, before)

    def update_state_dict(self, before, after):
        new_before = defaultdict(dict)
        for stage in after.keys():
            for name in after[stage].keys():
                _, loc, data_type = self.dataset_spec[name]
                if loc == Location.GRAPH:
                    if data_type in [Type.POINTER]:
                        new_before[stage][name] = self.update_per_mask(
                            before[stage][name], after[stage][name], mask=self.mask
                        )
                    else:
                        new_before[stage][name] = self.update_per_mask(
                            before[stage][name], after[stage][name], mask=self.mask_cp
                        )
                if loc == Location.EDGE:
                    if data_type in [
                        Type.MASK,
                        Type.MASK_ONE,
                        Type.SCALAR,
                        Type.CATEGORICAL,
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        new_before[stage][name] = self.update_per_mask(
                            before[stage][name],
                            after[stage][name],
                            mask=self.mask_edges,
                        )
                    else:
                        assert False, "Please implement"
                if loc == Location.NODE:
                    if data_type in [
                        Type.MASK,
                        Type.MASK_ONE,
                        Type.SCALAR,
                        Type.CATEGORICAL,
                    ]:
                        new_before[stage][name] = self.update_per_mask(
                            before[stage][name], after[stage][name]
                        )
                    elif data_type in [
                        Type.POINTER,
                        Type.PERMUTATION_POINTER,
                        Type.SHOULD_BE_PERMUTATION,
                    ]:
                        new_before[stage][name] = []
                        for i in range(len(after[stage][name])):
                            new_before[stage][name].append(
                                torch.where(
                                    self.mask_edges,
                                    after[stage][name][i],
                                    before[stage][name][i],
                                )
                            )
                    else:
                        assert False, breakpoint()
        return new_before

    def update_states(
        self,
        batch,
        current_latent,
        edges_current_latent,
        logits,
        continue_logits,
        cayley_latent=None,
    ):
        self.last_continue_logits = torch.where(
            self.mask_cp, continue_logits, self.last_continue_logits
        )
        self.last_latent = self.update_per_mask(self.last_latent, current_latent)
        if cayley_latent is not None and self.config["use_expander"]:
            self.last_latent_cayley = cayley_latent
        self.last_latent_edges = self.update_per_mask(
            self.last_latent_edges, edges_current_latent, mask=self.mask_edges
        )
        self.last_logits = self.update_state_dict(self.last_logits, logits)
        self.all_hint_logits.append(self.last_logits["hint"])
        self.all_masks_graph.append(self.mask_cp)
        self.all_continue_logits.append(continue_logits)
        preds = type(self).convert_logits_to_outputs(
            self.dataset_spec,
            self.last_logits,
            batch.edge_index[0],
            batch.edge_index[1],
            batch.num_nodes,
            batch.batch,
            self.epoch > self.debug_epoch_threshold,
        )
        self.last_hint = preds["hint"]
        self.last_output = preds["output"]

    def prepare_initial_masks(self, batch):
        self.mask = torch.ones_like(
            batch.batch, dtype=torch.bool, device=batch.edge_index.device
        )
        self.mask_cp = torch.ones(
            batch.num_graphs, dtype=torch.bool, device=batch.edge_index.device
        )
        self.mask_edges = torch.ones_like(
            batch.edge_index[0], dtype=torch.bool, device=batch.edge_index.device
        )

    def loop_condition(self, termination, STEPS_SIZE):
        return (
            (not self.training and termination.any())
            or (self.training and termination.any())
        ) and self.step_idx + 1 < STEPS_SIZE

    def loop_body(
        self,
        batch,
        node_fts,
        edge_fts,
        graph_fts,
        true_termination,
        first_n_processors=1000,
        hint_mode="encoded_decoded",
        forward_fn=None,
    ):
        if forward_fn == None:
            forward_fn = self.forward

        current_latent, edges_current_latent, preds, continue_logits, cayley_latent = (
            forward_fn(
                batch,
                node_fts,
                edge_fts,
                graph_fts,
                first_n_processors=first_n_processors,
                hint_mode=hint_mode,
            )
        )
        termination = continue_logits

        self.debug_batch = batch
        if self.timeit:
            st = time.time()
        self.update_states(
            batch,
            current_latent,
            edges_current_latent,
            preds,
            termination,
            cayley_latent=cayley_latent,
        )
        if self.timeit:
            print(f"updating states: {time.time()-st}")
        return preds

    def get_step_input(self, x_curr, batch):
        if self.training and self.use_TF or self.hardcode_outputs:
            return x_curr
        return type(self).convert_logits_to_outputs(
            self.dataset_spec,
            self.last_logits,
            batch.edge_index[0],
            batch.edge_index[1],
            batch.num_nodes,
            batch.batch,
            self.epoch > self.debug_epoch_threshold,
        )["hint"]

    def encode_single(self, encoder, data, loc, data_type, batch):
        node_fts = torch.zeros(
            batch.num_nodes, self.latent_features, device=batch.edge_index.device
        )
        edge_fts = torch.zeros(
            batch.num_edges, self.latent_features, device=batch.edge_index.device
        )
        graph_fts = torch.zeros(
            batch.num_graphs, self.latent_features, device=batch.edge_index.device
        )
        if data_type in [
            Type.POINTER,
            Type.PERMUTATION_POINTER,
            Type.SHOULD_BE_PERMUTATION,
        ]:
            if loc == Location.NODE:

                msk = edge_one_hot_encode_pointers(data.long(), batch)
                msk_inv = edge_one_hot_encode_pointers(data.long(), batch, inv=True)
                normals = encoder[0](msk.unsqueeze(-1))
                invs = torch.zeros_like(normals)
                if self.config["inv_ptr"]:
                    invs = encoder[1](msk_inv.unsqueeze(-1))
                return node_fts, normals + invs, graph_fts
            if loc == Location.EDGE and data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:
                encoder, _ = encoder
                pred_gt_one_hot = edge_one_hot_encode_pointers_edge(
                    data, batch, self.max_nodes_in_graph
                )
                starts_edge = batch.ptr[:-1][batch.batch[batch.edge_index[0]]]
                encoding = encoder[0](pred_gt_one_hot.unsqueeze(-1))
                encoding_2 = encoder[1](pred_gt_one_hot.unsqueeze(-1))
                encoding_sparse = SparseTensor(
                    row=batch.edge_index[0], col=batch.edge_index[1], value=encoding
                )
                res_1 = encoding_sparse.mean(1)[
                    batch.edge_index[0], batch.edge_index[1] - starts_edge
                ]
                res_2 = encoding_2.mean(1)
                edge_fts += res_1 + res_2  # INPLACE
                return node_fts, edge_fts, graph_fts
            else:
                assert False, breakpoint()
        if data_type != Type.CATEGORICAL:
            data = data.unsqueeze(-1)
        if loc == Location.EDGE:
            edge_fts += encoder(data)
        if loc == Location.NODE:
            node_fts += encoder(data)
        if loc == Location.GRAPH:
            graph_fts += encoder(data)
        assert (graph_fts != 0).any() or (edge_fts != 0).any() or (node_fts != 0).any()
        return node_fts, edge_fts, graph_fts

    def encode_inputs(self, batch):
        node_fts = torch.zeros(
            batch.num_nodes, self.latent_features, device=batch.edge_index.device
        )
        edge_fts = torch.zeros(
            batch.num_edges, self.latent_features, device=batch.edge_index.device
        )
        graph_fts = torch.zeros(
            batch.num_graphs, self.latent_features, device=batch.edge_index.device
        )
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.INPUT:
                continue
            if name not in self.encoders[stage]:
                continue
            data = getattr(batch, name)
            nf, ef, gf = self.encode_single(
                (
                    self.encoders[stage][name]
                    if "pointer" not in data_type
                    else (
                        self.encoders[stage][name],
                        self.encoders[stage][name + "_inv"],
                    )
                ),
                data,
                loc,
                data_type,
                batch,
            )
            node_fts += nf
            edge_fts += ef
            graph_fts += gf
        return node_fts, edge_fts, graph_fts

    def encode_hints(self, hints, batch):
        node_fts = torch.zeros(
            batch.num_nodes, self.latent_features, device=batch.edge_index.device
        )
        edge_fts = torch.zeros(
            batch.num_edges, self.latent_features, device=batch.edge_index.device
        )
        graph_fts = torch.zeros(
            batch.num_graphs, self.latent_features, device=batch.edge_index.device
        )

        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.HINT:
                continue
            if name not in self.encoders[stage]:
                continue
            hint = hints[name]
            nf, ef, gf = self.encode_single(
                (
                    self.encoders[stage][name]
                    if "pointer" not in data_type
                    else (
                        self.encoders[stage][name],
                        self.encoders[stage][name + "_inv"],
                    )
                ),
                hint.squeeze(-1),
                loc,
                data_type,
                batch,
            )
            node_fts += nf
            edge_fts += ef
            graph_fts += gf
        return node_fts, edge_fts, graph_fts

    def get_input_output_hints(self, batch):
        hint_inp_curr = {}
        hint_out_curr = {}
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage != Stage.HINT:
                continue
            hint_inp_curr[name] = batch[name][
                min(self.step_idx, batch[name].shape[0] - 1)
            ]
            hint_out_curr[name] = batch[name][
                min(self.step_idx + 1, batch[name].shape[0] - 1)
            ]
            if "mask" in data_type or data_type == Type.SCALAR:
                hint_inp_curr[name] = hint_inp_curr[name].unsqueeze(-1)
                hint_out_curr[name] = hint_out_curr[name].unsqueeze(-1)
        return hint_inp_curr, hint_out_curr

    def process(
        self,
        batch,
        EPSILON=0,
        enforced_mask=None,
        hardcode_outputs=False,
        debug=False,
        first_n_processors=1000,
        init_last_latent=None,
        hint_mode="encoded_decoded",
        forward_fn=None,
        **kwargs,
    ):

        if forward_fn == None:
            forward_fn = self.forward
        SIZE, STEPS_SIZE = prepare_constants(batch)
        STEPS_SIZE = int(STEPS_SIZE * self.config["termination_overshoot_factor"])
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
        all_latent_states = []
        # Prepare masking tensors (each graph does at least 1 iteration of the algo)
        self.prepare_initial_masks(batch)
        # A flag if we had a wrong graph in the batch. Used for visualisation
        # of what went wrong
        self.wrong_flag = False
        assert self.mask_cp.all(), self.mask_cp
        if self.timeit:
            st = time.time()
        node_fts_inp, edge_fts_inp, graph_fts_inp = self.encode_inputs(batch)
        if self.timeit:
            print(f"encoding inputs: {time.time()-st}")

        while True:

            node_fts = node_fts_inp
            edge_fts = edge_fts_inp
            graph_fts = graph_fts_inp
            hint_inp_curr = None
            hint_out_curr = None
            if "encoded" in hint_mode:
                hint_inp_curr, hint_out_curr = self.get_input_output_hints(batch)
                if not self.training:
                    assert (self.last_continue_logits > 0).any() or True

                # Some algorithms output fewer values than they take
                # so if we reuse our last step outputs, they need to be fed back in.
                if self.timeit:
                    st = time.time()
                hint_inp_curr = self.get_step_input(hint_inp_curr, batch)
                if self.timeit:
                    print(f"getting step input : {time.time()-st}")
                    st = time.time()
                node_fts_hint, edge_fts_hint, graph_fts_hint = self.encode_hints(
                    hint_inp_curr, batch
                )
                node_fts = node_fts + node_fts_hint
                edge_fts = edge_fts + edge_fts_hint
                graph_fts = graph_fts + graph_fts_hint
            if self.timeit:
                print(f"encoding hints: {time.time()-st}")

            true_termination = torch.where(
                self.step_idx + 1 >= batch.lengths - 1, -1e9, 1e9
            )

            self.loop_body(
                batch,
                node_fts,
                edge_fts,
                graph_fts,
                true_termination,
                first_n_processors=first_n_processors,
                hint_mode=hint_mode,
                forward_fn=forward_fn,
            )
            all_latent_states.append(self.last_latent.clone())
            # And calculate what graphs would execute on the next step.
            self.mask_cp, self.mask, self.mask_edges = type(self).get_masks(
                self.training,
                batch,
                true_termination if self.training else self.last_continue_logits,
                enforced_mask,
            )
            if not self.loop_condition(self.mask_cp, STEPS_SIZE):
                break
            assert self.mask_cp.any()
            self.step_idx += 1

        self.jac_reg_in = [self.last_latent, torch.zeros_like(self.last_latent_edges)]

        current_latent, edges_current_latent, _, _, _ = forward_fn(
            batch,
            node_fts,
            edge_fts,
            graph_fts,
            first_n_processors=first_n_processors,
            hint_mode=hint_mode,
        )
        self.jac_reg_out = [current_latent, torch.zeros_like(edges_current_latent)]
        all_latent_states_dense = torch.stack(
            [to_dense_batch(als, batch=batch.batch)[0] for als in all_latent_states],
            dim=0,
        )
        return (
            self.all_hint_logits,
            self.last_logits,
            self.all_masks_graph,
            self.all_continue_logits,
            all_latent_states_dense,
        )

    def decode_single(
        self, batch, catted, edge_fts, graph_fts, name, stage, loc, data_type
    ):
        ret = None
        if loc == Location.NODE:

            if data_type in [Type.MASK, Type.SCALAR, Type.CATEGORICAL, Type.MASK_ONE]:
                ret = self.decoders[stage][name](catted)

            if data_type in [
                Type.POINTER,
                Type.PERMUTATION_POINTER,
                Type.SHOULD_BE_PERMUTATION,
            ]:
                ret = []
                for i, nm in enumerate([name, name + "_inv"]):
                    fr = self.decoders[stage][nm][0](catted)[batch.edge_index[0]]  # x_i
                    to = self.decoders[stage][nm][1](catted)[batch.edge_index[1]]  # x_j
                    edge = self.decoders[stage][nm][2](edge_fts)  # e_ij
                    if i == 0:
                        prod = self.decoders[stage][nm][3](to.max(fr + edge)).squeeze(
                            -1
                        )
                    if (
                        i == 0
                        and data_type
                        in [Type.PERMUTATION_POINTER, Type.SHOULD_BE_PERMUTATION]
                        and self.use_sinkhorn
                    ):
                        prod = sinkhorn_normalize(
                            batch,
                            prod,
                            temperature=0.1,
                            steps=10 if self.training else 60,
                            add_noise=self.training,
                        )
                        prod = F.leaky_relu(prod + 6.0) - 6.0
                    if i == 1:
                        prod = self.decoders[stage][nm][3](fr.max(to + edge)).squeeze(
                            -1
                        )
                    ret.append(prod)

        if loc == Location.GRAPH:
            aggr_node_fts = torch_scatter.scatter_max(catted, batch.batch, dim=0)[0]
            ret = self.decoders[stage][name][0](aggr_node_fts) + self.decoders[stage][
                name
            ][1](graph_fts)
            if data_type in [Type.POINTER]:
                expd = ret[batch.batch]
                ret = expd + self.decoders[stage][name][2](catted)

        if loc == Location.EDGE:
            fr = self.decoders[stage][name][0](catted[batch.edge_index[0]])
            to = self.decoders[stage][name][1](catted[batch.edge_index[1]])
            edge = self.decoders[stage][name][2](edge_fts)
            if data_type in (Type.CATEGORICAL, Type.MASK):
                ret = fr + to + edge
            elif data_type == Type.POINTER:
                pred = fr + to + edge
                pred_2 = self.decoders[stage][name][3](catted)
                ebatch = batch.batch[batch.edge_index[0]]
                dense_pred_2, mask_pred_2 = to_dense_batch(pred_2, batch=batch.batch)
                edge_pred_2 = dense_pred_2[ebatch]
                mask_edge_pred_2 = mask_pred_2[ebatch]
                probs_logits = self.decoders[stage][name][4](
                    torch.maximum(pred[:, None, :], edge_pred_2)
                ).squeeze(-1)
                probs_logits[~mask_edge_pred_2] = -1e9
                ret = probs_logits
            else:
                assert False
        assert ret is not None
        return ret

    def decode(
        self,
        batch,
        encoded_nodes,
        hidden,
        edge_fts,
        graph_fts,
        hint_mode="encoded_decoded",
    ):
        catted = torch.cat((encoded_nodes, hidden), dim=1)
        outs = defaultdict(dict)
        for name, (stage, loc, data_type) in self.dataset_spec.items():
            if stage == Stage.INPUT:
                continue
            if "decoded" not in hint_mode and stage == Stage.HINT:
                continue

            outs[stage][name] = self.decode_single(
                batch, catted, edge_fts, graph_fts, name, stage, loc, data_type
            )

        return outs

    def encode_nodes(self, current_input, last_latent):
        return torch.cat((current_input, last_latent), dim=1)

    def call_processor(
        self,
        node_fts,
        edge_fts,
        graph_fts,
        hidden,
        edges_hidden,
        batch,
        edge_index,
        first_n_processors,
    ):

        cayley_hidden = None
        hidden, edges_hidden = self.processor(
            node_fts,
            edge_fts,
            graph_fts,
            edge_index,
            hidden,
            edges_hidden,
            first_n_processors=first_n_processors,
            batch=batch,
        )

        if self.config["use_expander"]:
            cayley_node_fts = torch.zeros(
                batch.cayley_num_nodes,
                self.latent_features,
                device=batch.edge_index.device,
            )
            cayley_edge_fts = self.cayley_edge_fts.repeat([batch.cayley_num_edges, 1])
            cayley_node_fts[: batch.num_nodes] = node_fts

            mask = (
                torch.arange(batch.cayley_num_nodes, device=batch.edge_index.device)
                < batch.num_nodes
            )
            newpart = self.last_latent_cayley.clone()
            newpart[: batch.num_nodes] = hidden
            self.last_latent_cayley = torch.where(
                mask[:, None], newpart, self.last_latent_cayley
            )
            cayley_hidden, _ = self.cgp_processor(
                cayley_node_fts,
                cayley_edge_fts,
                graph_fts,
                batch.cayley_edge_index,
                self.last_latent_cayley,
                edges_hidden,
                first_n_processors=first_n_processors,
                batch=batch,
            )
            hidden = cayley_hidden[: batch.num_nodes]
        return hidden, edges_hidden, cayley_hidden

    def forward(
        self,
        batch,
        node_fts,
        edge_fts,
        graph_fts,
        first_n_processors=1000,
        hint_mode="encoded_decoded",
    ):
        if torch.isnan(node_fts).any():
            breakpoint()
        assert not torch.isnan(self.last_latent).any()
        assert not torch.isnan(node_fts).any()
        if self.timeit:
            st = time.time()
        if self.timeit:
            print(f"projecting nodes: {time.time()-st}")

        if self.timeit:
            st = time.time()
        edge_index = batch.edge_index

        hidden, edges_hidden, cayley_hidden = self.call_processor(
            node_fts,
            edge_fts,
            graph_fts,
            self.last_latent,
            self.last_latent_edges,
            batch,
            edge_index,
            first_n_processors=first_n_processors,
        )
        if self.timeit:
            print(f"message passing: {time.time()-st}")
        assert not torch.isnan(hidden).any()
        if self.timeit:
            st = time.time()
        if self.config["update_edges_hidden"]:
            edge_fts = torch.cat((edge_fts, edges_hidden), -1)
        outs = self.decode(
            batch, node_fts, hidden, edge_fts, graph_fts, hint_mode=hint_mode
        )
        if self.timeit:
            print(f"decoding hints: {time.time()-st}")
        if self.config["learn_termination"]:
            continue_logits = self.get_continue_logits(batch, hidden, sth_else=None)
        else:
            continue_logits = torch.where(
                self.step_idx + 1 >= batch.lengths - 1, -1e9, 1e9
            )
        return hidden, edges_hidden, outs, continue_logits, cayley_hidden
