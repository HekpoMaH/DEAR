"""
Script for data analysis

Usage:
    analyse_datasets.py [options]

Options:
    -h --help               Show this screen.

    --algorithms ALGOS      List of algorithms to train on. Repeatable. [default: dinitz]
"""

import os
from datetime import datetime

from docopt import docopt
import schema
import torch
import pytorch_lightning as pl


from models.gnns import _PROCESSSOR_DICT
from models.algorithm_reasoner import LitAlgorithmReasoner
from models.algorithm_processor import LitAlgorithmProcessor
from hyperparameters import get_hyperparameters
from utils_execution import ReasonerZeroerCallback, get_callbacks, maybe_remove

if __name__ == "__main__":
    args = docopt(__doc__)
    args["--algorithms"] = args["--algorithms"].split(",")
    lit_processor = LitAlgorithmProcessor(
        64,
        args["--algorithms"],
        dict((algo, {}) for algo in args["--algorithms"]),
        dict((algo, LitAlgorithmReasoner) for algo in args["--algorithms"]),
        False,  # args['--ensure-permutation'] is False for non-TSP
        reduce_proc_hid_w_MLP=False,
        update_edges_hidden=False,
        use_TF=False,
        use_gate=True,
        use_LSTM=False,
        freeze_proc=False,  # We don't have a transfer task
        processors=["MPNN"],
        xavier_on_scalars=False,
        biased_gate=False,
    )
    for algo in lit_processor.algorithms:
        module = lit_processor.algorithms[algo]
        print(algo)
        for split in ["test", "test_128"]:
            print(" ", split)
            module.load_dataset(split)
            aggrs = torch.zeros(module.dataset[0].A_o.shape[-1])
            for dt in module.dataset:
                aggrs += dt.A_o.argmax(-1).unique(return_counts=True)[1]
            print(" ", aggrs / aggrs.sum(-1))
