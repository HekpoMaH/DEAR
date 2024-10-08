import torch
import numpy as np
from collections import deque
from torch_geometric.data import Data, Batch

_CAYLEY_BOUNDS = [
    (6, 2),
    (24, 3),
    (120, 5),
    # (336, 7),
    # (1320, 11),
    # (2184, 13),
    # (4896, 17),
    # (6840, 19),
    # (12144, 23),
    # (24360, 29),
    # (29760, 31),
    # (50616, 37),
]


def build_cayley_bank():
    ret_edges = []

    for _, p in _CAYLEY_BOUNDS:
        generators = np.array(
            [
                [[1, 1], [0, 1]],
                [[1, p - 1], [0, 1]],
                [[1, 0], [1, 1]],
                [[1, 0], [p - 1, 1]],
            ]
        )
        ind = 1

        queue = deque([np.array([[1, 0], [0, 1]])])
        nodes = {(1, 0, 0, 1): 0}

        senders = []
        receivers = []

        while queue:
            x = queue.pop()
            x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
            assert x_flat in nodes
            ind_x = nodes[x_flat]
            for i in range(4):
                tx = np.matmul(x, generators[i])
                tx = np.mod(tx, p)
                tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
                if tx_flat not in nodes:
                    nodes[tx_flat] = ind
                    ind += 1
                    queue.append(tx)
                ind_tx = nodes[tx_flat]

                senders.append(ind_x)
                receivers.append(ind_tx)

        ret_edges.append((p, [senders, receivers]))

    return ret_edges


def get_cayley_data(cayley_bank, ogs):
    cayley_data = []
    for nodes in ogs:
        chosen_i = -1
        for i in range(len(_CAYLEY_BOUNDS)):
            sz, p = _CAYLEY_BOUNDS[i]
            if sz >= nodes.item():
                chosen_i = i
                break

        _, edge_pack = cayley_bank[chosen_i]

        senders, receivers = edge_pack
        edge_index = torch.tensor([senders, receivers])
        cayley_data.append(Data(edge_index=edge_index, num_nodes=sz))

    return Batch.from_data_list(cayley_data)
