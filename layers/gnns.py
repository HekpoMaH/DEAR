from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng
from torch_geometric.utils import to_dense_batch, to_dense_adj


class GATv2(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="max",
        bias=False,
        flow="source_to_target",
        **unused_kwargs
    ):
        super().__init__()
        self.gat = nng.GATv2Conv(
            in_channels,
            out_channels,
            edge_dim=edge_dim,
            aggr=aggr,
            bias=bias,
            flow=flow,
            add_self_loops=False,
        )

    def forward(
        self, x, edge_attr, graph_fts, edge_index, hidden, edges_hidden, batch, **kwargs
    ):
        x = x + graph_fts[batch.batch]
        edge_attr = edge_attr + graph_fts[batch.batch][edge_index[0]]
        z = torch.cat((x, hidden), dim=-1)
        gat_hidden = self.gat(z, edge_index, edge_attr=edge_attr)
        if not self.training:
            gat_hidden = torch.clamp(gat_hidden, -1e9, 1e9)
        return gat_hidden + hidden, edges_hidden


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.offset = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv = self.scale * torch.rsqrt(var + self.epsilon)
        x = x - mean
        x = inv * x + self.offset
        return x


class GIN(nng.MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        bias=False,
        flow="source_to_target",
        num_layers=3,
        GRANOLA_NRNF=0,
        use_ln_MLP=False,
        num_mp_steps=1,
        **kwargs
    ):
        super(GIN, self).__init__(flow=flow, aggr="sum")
        self.num_mp_steps = num_mp_steps
        modules = [nn.Linear(in_channels + GRANOLA_NRNF, out_channels, bias=bias)]
        for _ in range(num_layers - 1):
            modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]
            )
        upd = [nn.Sequential(*modules)]
        for _ in range(self.num_mp_steps - 1):
            modules = [nn.Linear(out_channels, out_channels, bias=bias)]
            for _ in range(num_layers - 1):
                modules.extend(
                    [
                        LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels, bias=bias),
                    ]
                )
            upd.append(nn.Sequential(*modules))

        self.GRANOLA_NRNF = GRANOLA_NRNF
        self.upd = nn.ModuleList(upd)
        self.linedge = [nn.Linear(edge_dim, in_channels + GRANOLA_NRNF)]
        self.linedge.extend(
            nn.Linear(edge_dim, out_channels) for _ in range(num_layers - 1)
        )
        self.linedge = nn.ModuleList(self.linedge)

    def forward(
        self,
        node_fts,
        edge_attr,
        graph_fts,
        edge_index,
        hidden,
        edges_hidden,
        batch,
        **kwargs
    ):
        for mps in range(self.num_mp_steps):
            hidden = self.propagate(edge_index, x=hidden, edge_attr=edge_attr, mps=mps)
        if not self.training:
            hidden = torch.clamp(hidden, -1e9, 1e9)
        return hidden, edges_hidden

    def message(self, x_i, x_j, edge_attr, mps):
        return F.relu(x_i + self.linedge[mps](edge_attr))

    def update(self, aggr_out, x, mps):
        x = self.upd[mps](x + aggr_out)
        return x


class MPNN(nng.MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="max",
        bias=False,
        flow="source_to_target",
        use_gate=False,
        use_ln_MLP=True,
        use_ln_GNN=True,
        biased_gate=True,
        update_edges_hidden=False,
        num_layers=3,
        GRANOLA_NRNF=0,
    ):
        super(MPNN, self).__init__(aggr=aggr, flow=flow)
        modules = []
        for _ in range(num_layers - 1):
            modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]
            )
        self.GRANOLA_NRNF = GRANOLA_NRNF
        self.i_map = nn.Linear(in_channels + GRANOLA_NRNF, out_channels, bias=bias)
        self.j_map = nn.Linear(in_channels + GRANOLA_NRNF, out_channels, bias=bias)
        self.edge_map = nn.Linear(edge_dim, out_channels, bias=bias)
        self.graph_map = nn.Linear(edge_dim, out_channels, bias=bias)
        if update_edges_hidden:
            self.edge_hidden_map = nn.Linear(edge_dim, out_channels, bias=bias)
        self.M = nn.Sequential(*modules)
        self.update_edges_hidden = update_edges_hidden
        if self.update_edges_hidden:
            modules = [
                nn.Linear(2 * in_channels + 2 * edge_dim, out_channels, bias=bias)
            ]
            for _ in range(num_layers - 1):
                modules.extend(
                    [
                        LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels, bias=bias),
                    ]
                )
            self.M_e = nn.Sequential(*modules)
        self.use_gate = use_gate
        self.use_ln = use_ln_MLP
        self.biased_gate = biased_gate
        self.U1 = nn.Linear(in_channels + GRANOLA_NRNF, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)
        if use_gate:
            self.gate1 = nn.Linear(in_channels + GRANOLA_NRNF, out_channels, bias=bias)
            self.gate2 = nn.Linear(out_channels, out_channels, bias=bias)
            self.gate3 = nn.Linear(out_channels, out_channels, bias=bias)
            if self.biased_gate:
                assert bias, "Bias has to be enabled"
                torch.nn.init.constant_(self.gate3.bias, -3)
            if self.update_edges_hidden:
                self.gate1_e = nn.Linear(out_channels, out_channels, bias=bias)
                self.gate2_e = nn.Linear(out_channels, out_channels, bias=bias)
                self.gate3_e = nn.Linear(out_channels, out_channels, bias=bias)
                if self.biased_gate:
                    assert bias, "Bias has to be enabled"
                    torch.nn.init.constant_(self.gate3_e.bias, -3)

        self.out_channels = out_channels
        self.ln = LayerNorm(out_channels) if use_ln_GNN else nn.Identity()

    def forward(
        self,
        node_fts,
        edge_attr,
        graph_fts,
        edge_index,
        hidden,
        edges_hidden,
        batch,
        **kwargs
    ):
        z = torch.cat((node_fts, hidden), dim=-1)

        graph_fts_padded = torch.zeros(
            node_fts.shape[0], graph_fts.shape[1], device=batch.edge_index.device
        )
        graph_fts_padded[: batch.batch.shape[0]] = graph_fts[batch.batch]

        hidden = self.propagate(
            edge_index,
            x=z,
            hidden=hidden,
            edges_hidden=edges_hidden,
            edge_attr=edge_attr,
            graph_fts=graph_fts_padded,
        )
        if self.update_edges_hidden:
            edges_hidden = self.edge_updater(
                edge_index,
                x=z,
                hidden=hidden,
                edges_hidden=edges_hidden,
                edge_attr=edge_attr,
            )
        if not self.training:
            hidden = torch.clamp(hidden, -1e9, 1e9)
        return hidden, edges_hidden

    def message(self, x_i, x_j, edge_attr, graph_fts_i, edges_hidden):
        mapped = self.i_map(x_i)
        mapped += self.j_map(x_j)
        mapped += self.edge_map(edge_attr)
        mapped += self.graph_map(graph_fts_i)
        if self.update_edges_hidden:
            mapped += self.edge_hidden_map(edges_hidden)
        return self.M(mapped)

    def edge_update(self, x_i, x_j, edge_attr, edges_hidden):
        m_e = self.M_e(torch.cat((x_i, x_j, edge_attr, edges_hidden), dim=1))
        gate = F.sigmoid(
            self.gate3_e(F.relu(self.gate1_e(edges_hidden) + self.gate2_e(m_e)))
        )
        return m_e * gate + edges_hidden * (1 - gate)

    def update(self, aggr_out, x, hidden):
        hidden = hidden[..., : hidden.shape[-1] - self.GRANOLA_NRNF]
        h_1 = self.U1(x)
        h_2 = self.U2(aggr_out)
        # ret = F.relu()
        ret = self.ln(h_1 + h_2)
        if self.use_gate:
            gate = F.sigmoid(self.gate3(F.relu(self.gate1(x) + self.gate2(aggr_out))))
            # hidden = self.GRU_gate3(self.U(torch.cat((x, aggr_out), dim=1)), hidden)
            ret = ret * gate + hidden * (1 - gate)
        return ret


class TripletMPNN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="max",
        bias=False,
        flow="source_to_target",
        use_gate=False,
        biased_gate=True,
        update_edges_hidden=False,
        num_layers=2,
        use_ln_MLP=True,
        use_ln_GNN=True,
    ):
        super(TripletMPNN, self).__init__()
        assert aggr == "max", "Max only mode, soz!"
        self.update_edges_hidden = update_edges_hidden
        # self.use_ln = use_ln_MLP
        graph_dim = edge_dim
        lst = []
        edim = edge_dim
        for in_dim in [
            in_channels,
            in_channels,
            in_channels,
            edim,
            edim,
            edim,
            in_channels // 2,
        ]:
            modules = [nn.Linear(in_dim, 8, bias=bias)]
            lst.append(nn.Sequential(*modules))
        self.M_tri = nn.ModuleList(lst)
        lst = []
        for in_dim in [in_channels, in_channels, edim, graph_dim]:
            modules = [nn.Linear(in_dim, out_channels, bias=bias)]
            lst.append(nn.Sequential(*modules))

        modules = []
        for _ in range(num_layers):
            modules.extend(
                [
                    LayerNorm(out_channels) if use_ln_MLP else nn.Identity(),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels, bias=bias),
                ]
            )
        lst.append(nn.Sequential(*modules))
        self.M = nn.ModuleList(lst)
        self.use_gate = use_gate
        self.biased_gate = biased_gate
        self.U1 = nn.Linear(2 * out_channels, out_channels, bias=bias)
        self.U2 = nn.Linear(out_channels, out_channels, bias=bias)
        self.U3 = nn.Linear(8, out_channels, bias=bias)
        if use_gate:
            self.gate1 = nn.Linear(2 * out_channels, out_channels, bias=bias)
            self.gate2 = nn.Linear(out_channels, out_channels, bias=bias)
            self.gate3 = nn.Linear(out_channels, out_channels, bias=bias)
            if self.biased_gate:
                assert bias, "Bias has to be enabled"
                torch.nn.init.constant_(self.gate3.bias, -3)

        self.out_channels = out_channels
        self.trifd = self.triplet_forward_dense
        self.ln = LayerNorm(out_channels) if use_ln_GNN else nn.Identity()

    def triplet_forward_dense(
        self, z_dense, e_dense, graph_fts, mask, tri_msgs_mask, msgs_mask
    ):
        assert not torch.isnan(z_dense).any()
        tri_1 = self.M_tri[0](z_dense)
        tri_2 = self.M_tri[1](z_dense)
        tri_3 = self.M_tri[2](z_dense)
        tri_e_1 = self.M_tri[3](e_dense)
        tri_e_2 = self.M_tri[4](e_dense)
        tri_e_3 = self.M_tri[5](e_dense)
        tri_g = self.M_tri[6](graph_fts)
        tri_1[~mask] = 0
        tri_2[~mask] = 0
        tri_3[~mask] = 0

        tri_msgs = (
            tri_1[:, :, None, None, :]  #   (B, N, 1, 1, H)
            + tri_2[:, None, :, None, :]  # + (B, 1, N, 1, H)
            + tri_3[:, None, None, :, :]  # + (B, 1, 1, N, H)
            + tri_e_1[:, :, :, None, :]  # + (B, N, N, 1, H)
            + tri_e_2[:, :, None, :, :]  # + (B, N, 1, N, H)
            + tri_e_3[:, None, :, :, :]  # + (B, 1, N, N, H)
            + tri_g[:, None, None, None, :]  # + (B, 1, 1, 1, H)
        )  # = (B, N, N, N, H)
        assert not torch.isnan(tri_msgs).any()
        msk_tri = (
            mask[:, None, None, :] | mask[:, None, :, None] | mask[:, :, None, None]
        )
        tri_msgs[~msk_tri] = -1e9
        tri_msgs_pooled = tri_msgs.max(1).values
        tri_msgs = F.relu(self.U3(tri_msgs_pooled))  # B x N x N x H

        msg_1 = self.M[0](z_dense)  # B x N x H
        msg_2 = self.M[1](z_dense)  # B x N x H
        msg_e = self.M[2](e_dense)  # B x N x N x H
        msg_g = self.M[3](graph_fts)  # B x H
        msg_1[~mask] = 0
        msg_2[~mask] = 0
        msg_e[~msgs_mask] = 0
        msgs = (
            msg_1[:, None, :, :]
            + msg_2[:, :, None, :]
            + msg_e
            + msg_g[:, None, None, :]
        )  # B x N x N x H
        assert not torch.isnan(msgs).any()
        msgs = self.M[-1](msgs)
        assert not torch.isnan(msgs).any(), breakpoint()
        msgs[~msgs_mask] = -1e9
        msgs = msgs.max(1).values
        assert not torch.isnan(msgs).any()
        h_1 = self.U1(z_dense)
        assert not torch.isnan(h_1).any()
        h_2 = self.U2(msgs)
        assert not torch.isnan(h_2).any()
        ret = h_1 + h_2
        assert not torch.isnan(ret).any()
        return ret, msgs, tri_msgs

    def forward(
        self, node_fts, edge_attr, graph_fts, edge_index, hidden, *args, batch=None
    ):
        z = torch.cat((node_fts, hidden), dim=-1)
        hidden_dense, _ = to_dense_batch(hidden, batch=batch.batch)  # BxNxH
        z_dense, mask = to_dense_batch(z, batch=batch.batch)  # BxNxH
        e_dense = to_dense_adj(
            edge_index, batch=batch.batch, edge_attr=edge_attr
        )  # BxNxNxH
        adj_mat = to_dense_adj(edge_index, batch=batch.batch).bool()
        fn = self.trifd if self.training else self.triplet_forward_dense
        ret, msgs, tri_msgs = self.triplet_forward_dense(
            z_dense,
            e_dense,
            graph_fts,
            mask,
            mask[:, :, None] | mask[:, None, :],
            adj_mat,
        )
        ret = self.ln(ret)
        if self.use_gate:
            gate = F.sigmoid(self.gate3(F.relu(self.gate1(z_dense) + self.gate2(msgs))))
            ret = ret * gate + hidden_dense * (1 - gate)
        ebatch = batch.edge_index_batch
        e1 = batch.edge_index[0] - batch.ptr[ebatch]
        e2 = batch.edge_index[1] - batch.ptr[ebatch]
        ret = ret[mask]
        assert (ret != -1e9).all(), breakpoint()
        return ret, tri_msgs[ebatch, e1, e2]
