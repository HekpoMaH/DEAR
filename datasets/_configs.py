from collections import defaultdict

_DEFAULT_CONFIG = {
    "train": {"num_samples": 10000, "num_nodes": 16},
    "val": {"num_samples": 100, "num_nodes": 16},
    "test": {"num_samples": 100, "num_nodes": 64},
    "test_clrs": {"num_samples": 100, "num_nodes": 64},
    "test_128": {"num_samples": 100, "num_nodes": 128},
    "test_256": {"num_samples": 100, "num_nodes": 256},
    "test_512": {"num_samples": 100, "num_nodes": 512},
}
CONFIGS = defaultdict(lambda: _DEFAULT_CONFIG)
