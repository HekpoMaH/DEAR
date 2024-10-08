from typing import Tuple

import chex
from clrs._src import probing
from datasets.specs import SPECS
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_Out = Tuple[int, probing.ProbesDict]

_ALPHABET_SIZE = 4


def search(x, A) -> _Out:
    """Just search."""

    chex.assert_rank(A, 1)
    probes = probing.initialize(SPECS["search"])

    T_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            "pos": np.copy(T_pos) * 1.0 / A.shape[0],
            "key": np.copy(A),
            "target": np.copy(x),
        },
    )

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            "filler": probing.mask_one((A.shape[0] - 1) // 2, A.shape[0]),
        },
    )
    low = 0
    high = A.shape[0] - 1  # make sure return is always in array
    while low < high:
        mid = (low + high) // 2
        if x <= A[mid]:
            high = mid
        else:
            low = mid + 1
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                "filler": probing.mask_one((A.shape[0] - 1) // 2, A.shape[0]),
            },
        )

    if len(np.maximum(A - x, 0).nonzero()[0]) == 0:  # CLEARLY WRONG WAY
        high = A.shape[0] - 1
        assert False, breakpoint()
    else:
        high = np.maximum(A - x, 0).nonzero()[0][0].item()
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={
            "return": np.copy(high),
            "absdiff": np.copy(A - x),
            "ispos": np.copy((A - x) > 0),
        },
    )

    probing.finalize(probes)

    return high, probes


def heapsort_local(A: _Array) -> _Out:
    """Heapsort (Williams, 1964)."""

    chex.assert_rank(A, 1)
    probes = probing.initialize(SPECS["heapsort_local"])

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / A.shape[0],
            "key": np.copy(A),
            "pred_h_i": probing.array(np.copy(A_pos)),
        },
    )

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            "parent": probing.heap(np.copy(A_pos), A.shape[0]),
            "i": probing.mask_one(A.shape[0] - 1, A.shape[0]),
            "j": probing.mask_one(A.shape[0] - 1, A.shape[0]),
            "largest": probing.mask_one(A.shape[0] - 1, A.shape[0]),
            "heap_size": probing.mask_one(A.shape[0] - 1, A.shape[0]),
            "phase": probing.mask_one(0, 3),
        },
    )

    def max_heapify(A, i, heap_size, ind, phase):
        l = 2 * i + 1
        r = 2 * i + 2
        if l < heap_size and A[l] > A[i]:
            largest = l
        else:
            largest = i
        if r < heap_size and A[r] > A[largest]:
            largest = r
        if largest != i:
            tmp = A[i]
            A[i] = A[largest]
            A[largest] = tmp

            tmp = A_pos[i]
            A_pos[i] = A_pos[largest]
            A_pos[largest] = tmp

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                "pred_h": probing.array(np.copy(A_pos)),
                "parent": probing.heap(np.copy(A_pos), heap_size),
                "i": probing.mask_one(A_pos[ind], A.shape[0]),
                "j": probing.mask_one(A_pos[i], A.shape[0]),
                "largest": probing.mask_one(A_pos[largest], A.shape[0]),
                "heap_size": probing.mask_one(A_pos[heap_size - 1], A.shape[0]),
                "phase": probing.mask_one(phase, 3),
            },
        )

        if largest != i:
            max_heapify(A, largest, heap_size, ind, phase)

    def build_max_heap(A):
        for i in reversed(range(A.shape[0])):
            max_heapify(A, i, A.shape[0], i, 0)

    build_max_heap(A)
    heap_size = A.shape[0]
    for i in reversed(range(1, A.shape[0])):
        tmp = A[0]
        A[0] = A[i]
        A[i] = tmp

        tmp = A_pos[0]
        A_pos[0] = A_pos[i]
        A_pos[i] = tmp

        heap_size -= 1

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                "pred_h": probing.array(np.copy(A_pos)),
                "parent": probing.heap(np.copy(A_pos), heap_size),
                "i": probing.mask_one(A_pos[0], A.shape[0]),
                "j": probing.mask_one(A_pos[i], A.shape[0]),
                "largest": probing.mask_one(0, A.shape[0]),  # Consider masking
                "heap_size": probing.mask_one(A_pos[heap_size - 1], A.shape[0]),
                "phase": probing.mask_one(1, 3),
            },
        )

        max_heapify(A, 0, heap_size, i, 2)  # reduce heap_size!

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={"pred": probing.array(np.copy(A_pos))},
    )

    probing.finalize(probes)

    return A, probes


def insertion_sort_local(A: _Array) -> _Out:
    """Insertion sort."""

    chex.assert_rank(A, 1)
    probes = probing.initialize(SPECS["insertion_sort_local"])

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / A.shape[0],
            "key": np.copy(A),
            "pred_h_i": probing.array(np.copy(A_pos)),
        },
    )

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            "i": probing.mask_one(0, A.shape[0]),
            "j": probing.mask_one(0, A.shape[0]),
        },
    )

    for j in range(1, A.shape[0]):
        key = A[j]
        # Insert A[j] into the sorted sequence A[1 .. j - 1]
        i = j - 1
        while i >= 0 and A[i] > key:
            A[i + 1] = A[i]
            A_pos[i + 1] = A_pos[i]
            i -= 1
        A[i + 1] = key
        stor_pos = A_pos[i + 1]
        A_pos[i + 1] = j

        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                "i": probing.mask_one(stor_pos, np.copy(A.shape[0])),
                "j": probing.mask_one(j, np.copy(A.shape[0])),
            },
        )

    probing.push(
        probes, specs.Stage.OUTPUT, next_probe={"pred": probing.array(np.copy(A_pos))}
    )

    probing.finalize(probes)

    return A, probes


def strongly_connected_components_local(A: _Array) -> _Out:
    """Kosaraju's strongly-connected components (Aho et al., 1974)."""

    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS["strongly_connected_components"])

    A_pos = np.arange(A.shape[0])

    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            "pos": np.copy(A_pos) * 1.0 / A.shape[0],
            "A": np.copy(A),
            "adj": probing.graph(np.copy(A)),
        },
    )

    scc_id = np.arange(A.shape[0])
    reset = np.zeros(A.shape[0])
    color = np.zeros(A.shape[0], dtype=np.int32)
    d = np.zeros(A.shape[0])
    f = np.zeros(A.shape[0])
    s_prev = np.arange(A.shape[0])
    time = 0
    A_t = np.transpose(A)

    for s in range(A.shape[0]):
        if color[s] == 0:
            s_last = s
            u = s
            v = s
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    "scc_id_h": np.copy(scc_id),
                    "A_t": probing.graph(np.copy(A_t)),
                    "color": probing.array_cat(color, 3),
                    "d": np.copy(d),
                    "f": np.copy(f),
                    "s_prev": np.copy(s_prev),
                    "s": probing.mask_one(s, A.shape[0]),
                    "u": probing.mask_one(u, A.shape[0]),
                    "v": probing.mask_one(v, A.shape[0]),
                    "s_last": probing.mask_one(s_last, A.shape[0]),
                    "time": time,
                    "phase": 0,
                },
            )
            while True:
                if color[u] == 0 or d[u] == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1
                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "scc_id_h": np.copy(scc_id),
                            "A_t": probing.graph(np.copy(A_t)),
                            "color": probing.array_cat(color, 3),
                            "d": np.copy(d),
                            "f": np.copy(f),
                            "s_prev": np.copy(s_prev),
                            "s": probing.mask_one(s, A.shape[0]),
                            "u": probing.mask_one(u, A.shape[0]),
                            "v": probing.mask_one(v, A.shape[0]),
                            "s_last": probing.mask_one(s_last, A.shape[0]),
                            "time": time,
                            "phase": 0,
                        },
                    )
                for v in range(A.shape[0]):
                    if A[u, v] != 0:
                        if color[v] == 0:
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v
                            probing.push(
                                probes,
                                specs.Stage.HINT,
                                next_probe={
                                    "scc_id_h": np.copy(scc_id),
                                    "A_t": probing.graph(np.copy(A_t)),
                                    "color": probing.array_cat(color, 3),
                                    "d": np.copy(d),
                                    "f": np.copy(f),
                                    "s_prev": np.copy(s_prev),
                                    "s": probing.mask_one(s, A.shape[0]),
                                    "u": probing.mask_one(u, A.shape[0]),
                                    "v": probing.mask_one(v, A.shape[0]),
                                    "s_last": probing.mask_one(s_last, A.shape[0]),
                                    "time": time,
                                    "phase": 0,
                                },
                            )
                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "scc_id_h": np.copy(scc_id),
                            "A_t": probing.graph(np.copy(A_t)),
                            "color": probing.array_cat(color, 3),
                            "d": np.copy(d),
                            "f": np.copy(f),
                            "s_prev": np.copy(s_prev),
                            "s": probing.mask_one(s, A.shape[0]),
                            "u": probing.mask_one(u, A.shape[0]),
                            "v": probing.mask_one(v, A.shape[0]),
                            "s_last": probing.mask_one(s_last, A.shape[0]),
                            "time": time,
                            "phase": 0,
                        },
                    )

                    if s_prev[u] == u:
                        assert s_prev[s_last] == s_last
                        break
                    pr = s_prev[s_last]
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    color = np.zeros(A.shape[0], dtype=np.int32)
    s_prev = np.arange(A.shape[0])

    for s in np.argsort(-f):
        if color[s] == 0:
            s_last = s
            u = s
            v = s
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    "scc_id_h": np.copy(scc_id),
                    "A_t": probing.graph(np.copy(A_t)),
                    "color": probing.array_cat(color, 3),
                    "d": np.copy(d),
                    "f": np.copy(f),
                    "s_prev": np.copy(s_prev),
                    "s": probing.mask_one(s, A.shape[0]),
                    "u": probing.mask_one(u, A.shape[0]),
                    "v": probing.mask_one(v, A.shape[0]),
                    "s_last": probing.mask_one(s_last, A.shape[0]),
                    "time": time,
                    "phase": 1,
                },
            )
            scc_id[u] = s
            reset[u] = 1
            while True:
                if color[u] == 0 or d[u] == 0.0:
                    time += 0.01
                    d[u] = time
                    color[u] = 1
                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "scc_id_h": np.copy(scc_id),
                            "A_t": probing.graph(np.copy(A_t)),
                            "color": probing.array_cat(color, 3),
                            "d": np.copy(d),
                            "f": np.copy(f),
                            "s_prev": np.copy(s_prev),
                            "s": probing.mask_one(s, A.shape[0]),
                            "u": probing.mask_one(u, A.shape[0]),
                            "v": probing.mask_one(v, A.shape[0]),
                            "s_last": probing.mask_one(s_last, A.shape[0]),
                            "time": time,
                            "phase": 1,
                        },
                    )
                for v in range(A.shape[0]):
                    if A_t[u, v] != 0:
                        if color[v] == 0:
                            color[v] = 1
                            s_prev[v] = s_last
                            s_last = v
                            assert reset[v] == 0
                            reset[v] = 1
                            scc_id[v] = u
                            probing.push(
                                probes,
                                specs.Stage.HINT,
                                next_probe={
                                    "scc_id_h": np.copy(scc_id),
                                    "A_t": probing.graph(np.copy(A_t)),
                                    "color": probing.array_cat(color, 3),
                                    "d": np.copy(d),
                                    "f": np.copy(f),
                                    "s_prev": np.copy(s_prev),
                                    "s": probing.mask_one(s, A.shape[0]),
                                    "u": probing.mask_one(u, A.shape[0]),
                                    "v": probing.mask_one(v, A.shape[0]),
                                    "s_last": probing.mask_one(s_last, A.shape[0]),
                                    "time": time,
                                    "phase": 1,
                                },
                            )
                            break

                if s_last == u:
                    color[u] = 2
                    time += 0.01
                    f[u] = time

                    probing.push(
                        probes,
                        specs.Stage.HINT,
                        next_probe={
                            "scc_id_h": np.copy(scc_id),
                            "A_t": probing.graph(np.copy(A_t)),
                            "color": probing.array_cat(color, 3),
                            "d": np.copy(d),
                            "f": np.copy(f),
                            "s_prev": np.copy(s_prev),
                            "s": probing.mask_one(s, A.shape[0]),
                            "u": probing.mask_one(u, A.shape[0]),
                            "v": probing.mask_one(v, A.shape[0]),
                            "s_last": probing.mask_one(s_last, A.shape[0]),
                            "time": time,
                            "phase": 1,
                        },
                    )

                    if s_prev[u] == u:
                        assert s_prev[s_last] == s_last
                        break
                    pr = s_prev[s_last]
                    s_prev[s_last] = s_last
                    s_last = pr

                u = s_last

    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={"scc_id": np.copy(scc_id)},
    )
    probing.finalize(probes)

    return scc_id, probes


DIRECTED = np.array(
    [
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)

DIRECTED_2 = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
)

DIRECTED_3 = np.array(
    [
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
)

DIRECTED_4 = np.array(
    [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)

UNDIRECTED = np.array(
    [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ]
)


def test_strongly_connected_components_local():
    expected_directed = np.array([0, 1, 2, 1, 3, 5])
    out, _ = strongly_connected_components_local(DIRECTED)
    assert (out == expected_directed).all()
    expected_undirected = np.array([0, 0, 1, 2, 3])
    out, _ = strongly_connected_components_local(UNDIRECTED)
    assert (out == expected_undirected).all()
    out, _ = strongly_connected_components_local(DIRECTED_2)
    out, _ = strongly_connected_components_local(DIRECTED_3)
    out, _ = strongly_connected_components_local(DIRECTED_4)
    breakpoint()


if __name__ == "__main__":
    test_strongly_connected_components_local()
