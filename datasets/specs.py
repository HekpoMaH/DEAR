from clrs._src.specs import SPECS
from clrs import Type, Location, Stage


LSPECS = {
    "heapsort_local": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "pred_h_i": (Stage.INPUT, Location.NODE, Type.POINTER),
        "pred": (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
        "parent": (Stage.HINT, Location.NODE, Type.POINTER),
        "i": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "j": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "largest": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "heap_size": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "phase": (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
    "insertion_sort_local": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "pred_h_i": (Stage.INPUT, Location.NODE, Type.POINTER),
        "pred": (Stage.OUTPUT, Location.NODE, Type.SHOULD_BE_PERMUTATION),
        "i": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "j": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    "strongly_connected_components_local": SPECS["strongly_connected_components"],
    "search": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "key": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "target": (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        "return": (Stage.OUTPUT, Location.GRAPH, Type.POINTER),
        "absdiff": (Stage.OUTPUT, Location.NODE, Type.SCALAR),
        "ispos": (Stage.OUTPUT, Location.NODE, Type.MASK),
        "filler": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
}

SPECS = SPECS | LSPECS
