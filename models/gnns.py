import torch
import torch.nn as nn
import pytorch_lightning as pl
from layers.gnns import MPNN, GATv2, TripletMPNN, GIN, LayerNorm

_PROCESSSOR_DICT = {
    "MPNN": MPNN,
    "TriMPNN": TripletMPNN,
    "GATv2": GATv2,
}


class LitProcessorSet(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        *args,
        processors=["MPNN"],
        reduce_with_MLP=False,
        update_edges_hidden=False,
        **kwargs
    ):
        super().__init__()
        self.processors = nn.ModuleList([])
        for proc in processors:
            self.processors.append(
                LitGNN(
                    in_channels,
                    out_channels,
                    edge_dim,
                    *args,
                    processor_type=proc,
                    update_edges_hidden=update_edges_hidden,
                    **kwargs
                )
            )
        self.reduce_with_MLP = reduce_with_MLP
        self.update_edges_hidden = update_edges_hidden
        if reduce_with_MLP:

            self.reductor = nn.Sequential(
                nn.Linear(out_channels * len(self.processors), out_channels),
                nn.LayerNorm(out_channels),
                nn.LeakyReLU(),
                nn.Linear(out_channels, out_channels),
            )

            if update_edges_hidden:
                self.reductor_e = nn.Sequential(
                    nn.Linear(edge_dim * len(self.processors), out_channels),
                    nn.LayerNorm(out_channels),
                    nn.LeakyReLU(),
                    nn.Linear(out_channels, out_channels),
                )

    def zero_lstm(self, num_nodes):
        for proc in self.processors:
            proc.zero_lstm(num_nodes)

    def forward(self, *args, first_n_processors=1000, **kwargs):
        proco = [proc(*args, **kwargs) for proc in self.processors[:first_n_processors]]
        if self.reduce_with_MLP:
            re = self.reductor(
                torch.cat(
                    [
                        proco[i][0]
                        for i in range(len(self.processors))[:first_n_processors]
                    ],
                    dim=-1,
                )
            )
            if self.update_edges_hidden:
                re_e = self.reductor_e(
                    torch.cat(
                        [
                            proco[i][1]
                            for i in range(len(self.processors))[:first_n_processors]
                        ],
                        dim=-1,
                    )
                )
            else:
                re_e = sum(
                    [
                        proco[i][1]
                        for i in range(len(self.processors))[:first_n_processors]
                    ]
                ) / len(self.processors[:first_n_processors])
        else:
            re = sum(
                [proco[i][0] for i in range(len(self.processors))[:first_n_processors]]
            ) / len(self.processors[:first_n_processors])
            re_e = sum(
                [proco[i][1] for i in range(len(self.processors))[:first_n_processors]]
            ) / len(self.processors[:first_n_processors])

        return re, re_e


class LitGNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        *args,
        processor_type="MPNN",
        use_LSTM=False,
        config=None,
        **kwargs
    ):
        super().__init__()
        use_GRANOLA = config["use_GRANOLA"]
        GRANOLA_NRNF = config["GRANOLA_NRNF"]
        if use_GRANOLA:
            kwargs.update(use_ln_GNN=False)
        self.processor = _PROCESSSOR_DICT[processor_type](
            in_channels, out_channels, *args, **kwargs
        )
        self.use_LSTM = use_LSTM
        self.use_GRANOLA = use_GRANOLA
        self.out_channels = out_channels
        if use_LSTM:
            self.LSTMCell = nn.LSTMCell(out_channels, out_channels)
        if use_GRANOLA:
            self.GRANOLA = MPNN(
                in_channels,
                out_channels,
                *args,
                **(
                    kwargs
                    | {
                        "use_gate": False,
                        "biased_gate": False,
                        "use_ln_GNN": False,
                        "use_ln_MLP": True,
                        "aggr": "max",
                        "num_layers": 3,
                        "GRANOLA_NRNF": GRANOLA_NRNF,
                    }
                )
            )
            self.GRANOLA_f1 = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                LayerNorm(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.GRANOLA_f2 = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                LayerNorm(out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.GRANOLA_NRNF = GRANOLA_NRNF

    def zero_lstm(self, num_nodes):
        if self.use_LSTM:
            self.lstm_state = (
                torch.zeros((num_nodes, self.out_channels), device=self.device),
                torch.zeros((num_nodes, self.out_channels), device=self.device),
            )

    def forward(
        self, node_fts, edge_attr, graph_fts, edge_index, hidden, *args, **kwargs
    ):
        hidden = self.processor.forward(
            node_fts, edge_attr, graph_fts, edge_index, hidden, *args, **kwargs
        )

        if self.use_GRANOLA:
            hidden, edges_hidden = hidden
            RNF = torch.randn(
                (hidden.shape[0], self.GRANOLA_NRNF), device=hidden.device
            )
            hidden_w_RNF = torch.cat([hidden, RNF], dim=-1)
            Z, _ = self.GRANOLA(
                torch.zeros_like(node_fts),
                edge_attr,
                torch.zeros_like(graph_fts),
                edge_index,
                hidden_w_RNF,
                *args,
                **kwargs
            )
            gamma = self.GRANOLA_f1(Z)
            beta = self.GRANOLA_f2(Z)
            mean = hidden.mean(dim=-1, keepdim=True)
            var = hidden.var(dim=-1, keepdim=True, unbiased=False)
            inv = gamma * torch.rsqrt(var + 1e-5)
            hidden = inv * (hidden - mean) + beta
            hidden = (hidden, edges_hidden)

        if self.use_LSTM:
            self.lstm_state = self.LSTMCell(hidden, self.lstm_state)
            hidden = self.lstm_state[0]
        return hidden
