import os.path as osp
from datasets.clrs_datasets import CLRS
from datasets.specs import SPECS
from clrs import Type, Location, Stage

_DATASET_CLASSES = {
    "mst_prim": CLRS,
    "bellman_ford": CLRS,
    "floyd_warshall": CLRS,
    "heapsort": CLRS,
    "insertion_sort_local": CLRS,
    "insertion_sort": CLRS,
    "heapsort_local": CLRS,
    "graham_scan": CLRS,
    "find_maximum_subarray": CLRS,
    "find_maximum_subarray_kadane": CLRS,
    "lcs_length": CLRS,
    "optimal_bst": CLRS,
    "matrix_chain_order": CLRS,
    "segments_intersect": CLRS,
    "jarvis_march": CLRS,
    "dfs": CLRS,
    "bfs": CLRS,
    "kmp_matcher": CLRS,
    "articulation_points": CLRS,
    "topological_sort": CLRS,
    "bridges": CLRS,
    "mst_kruskal": CLRS,
    "dijkstra": CLRS,
    "dag_shortest_paths": CLRS,
    "bipartite_matching": CLRS,
    "activity_selector": CLRS,
    "task_scheduling": CLRS,
    "minimum": CLRS,
    "binary_search": CLRS,
    "search": CLRS,
    "quickselect": CLRS,
    "bubble_sort": CLRS,
    "quicksort": CLRS,
    "naive_string_matcher": CLRS,
    "strongly_connected_components": CLRS,
    "strongly_connected_components_local": CLRS,
}

_DATASET_ROOTS = {
    "mst_prim": osp.abspath("./data/clrs/"),
    "kmp_matcher": osp.abspath("./data/clrs/"),
    "bellman_ford": osp.abspath("./data/clrs/"),
    "floyd_warshall": osp.abspath("./data/clrs/"),
    "graham_scan": osp.abspath("./data/clrs/"),
    "find_maximum_subarray": osp.abspath("./data/clrs/"),
    "find_maximum_subarray_kadane": osp.abspath("./data/clrs/"),
    "bfs": osp.abspath("./data/clrs/"),
    "dfs": osp.abspath("./data/clrs/"),
    "matrix_chain_order": osp.abspath("./data/clrs/"),
    "lcs_length": osp.abspath("./data/clrs/"),
    "optimal_bst": osp.abspath("./data/clrs/"),
    "segments_intersect": osp.abspath("./data/clrs/"),
    "topological_sort": osp.abspath("./data/clrs/"),
    "jarvis_march": osp.abspath("./data/clrs/"),
    "articulation_points": osp.abspath("./data/clrs/"),
    "bridges": osp.abspath("./data/clrs/"),
    "mst_kruskal": osp.abspath("./data/clrs/"),
    "dijkstra": osp.abspath("./data/clrs/"),
    "dag_shortest_paths": osp.abspath("./data/clrs/"),
    "bipartite_matching": osp.abspath("./data/clrs/"),
    "activity_selector": osp.abspath("./data/clrs/"),
    "task_scheduling": osp.abspath("./data/clrs/"),
    "minimum": osp.abspath("./data/clrs/"),
    "binary_search": osp.abspath("./data/clrs/"),
    "search": osp.abspath("./data/clrs/"),
    "quickselect": osp.abspath("./data/clrs/"),
    "bubble_sort": osp.abspath("./data/clrs/"),
    "quicksort": osp.abspath("./data/clrs/"),
    "naive_string_matcher": osp.abspath("./data/clrs/"),
    "heapsort": osp.abspath("./data/clrs/"),
    "heapsort_local": osp.abspath("./data/clrs/"),
    "insertion_sort_local": osp.abspath("./data/clrs/"),
    "insertion_sort": osp.abspath("./data/clrs/"),
    # 'max_heapify': osp.abspath('./data/clrs/'),
    "strongly_connected_components": osp.abspath("./data/clrs/"),
    "strongly_connected_components_local": osp.abspath("./data/clrs/"),
}


_DATASET_SPECS = dict()


for algo in _DATASET_ROOTS.keys():
    _DATASET_SPECS[algo] = {
        "dataclass": CLRS,
        "rootdir": _DATASET_ROOTS[algo],
        "data_spec": SPECS[algo],
    }
