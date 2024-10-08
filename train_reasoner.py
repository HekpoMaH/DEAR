"""
Script to train the reasoner model.

Usage:
    train_reasoner.py [options]

Options:
    -h --help               Show this screen.

    --patience P            Patience value. If present, the training will utilise
                            early stopping based on validation loss.

    --model-name MN         Name of the model when saving. Defaults to current time
                            and date if not provided.

    --processors PS         Which processors to use. String of comma separated values.
                            [default: MPNN]

    --RPHWM                 Whether to Reduce Processor set Hiddens With MLP?

    --gradient-clip-val G   Constant for gradient clipping. 0 means no clipping.
                            [default: 1]

    --xavier-on-scalars     Use Xavier initialisation for linears that encode scalars.

    --biased-gate           Bias the gating mechanism towards less updating

    --update-edges-hidden   Whether to also keep a track of hidden edge state.

    --use-LSTM              Add an LSTMCell just after the processor step
                            (in case of several processors, each has its own LSTM)

    --algorithms ALGOS      List of algorithms to train on. Repeatable. [default: heapsort]

    --seed S                Random seed to set. [default: 47]
    
    --config-path CP        Path to yml config file.

    --agentify SID          Whether to start an agent.

"""

import os
import yaml
from datetime import datetime
import pickle

from docopt import docopt
import schema
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
import wandb
from datasets.specs import SPECS


from models.gnns import _PROCESSSOR_DICT
from models.algorithm_reasoner import LitAlgorithmReasoner
from models.algorithm_processor import LitAlgorithmProcessor
from hyperparameters import get_hyperparameters
from utils_execution import ReasonerZeroerCallback, get_callbacks, maybe_remove
from datasets.clrs_datasets import _load_inputs, _load_hints_and_outputs


def get_fake_batch(algorithm, dataset):
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    with open("fb.pkl", "rb") as f:
        feedback = pickle.load(f)
    _, spec = dataset.get_sampler(4)
    BS = feedback.features.inputs[0].data.shape[0]
    lst = []
    for i in range(BS):
        data = Data()
        data, new_spec = _load_inputs(data, feedback, spec, i=i)
        data, new_spec = _load_hints_and_outputs(data, feedback, new_spec, i=i)
        lst.append(data)
    return Batch.from_data_list(lst, follow_batch=["edge_index"])


def constant_fill(model, value):
    for param in model.parameters():
        param.data.fill_(value)


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def train(config, args=None):
    if args["--config-path"] is not None:
        with open(args["--config-path"], "r") as cfg:
            config = yaml.safe_load(cfg)
    config["update_edges_hidden"] = (
        config.get("update_edges_hidden", False)
        | args["--update-edges-hidden"]
        | ("TriMPNN" in args["--processors"])
    )
    wandb.init(
        project="nardeq",
        entity="clrs-cambridge", # NOTE CHANGE HERE
        group=None,
        settings=wandb.Settings(code_dir="."),
        save_code=True,
    )
    if args["--agentify"] is not None:
        config = deep_update(config, wandb.config.as_dict())
        args["--seed"] = config["seed"]
    hidden_dim = get_hyperparameters()["dim_latent"]
    serialised_models_dir = os.path.abspath("./serialised_models/")

    name = (
        args["--model-name"]
        if args["--model-name"] is not None
        else datetime.now().strftime("%b-%d-%Y-%H-%M")
    )
    pl.utilities.seed.seed_everything(args["--seed"])

    lit_processor = LitAlgorithmProcessor(
        hidden_dim,
        args["--algorithms"],
        dict((algo, {}) for algo in args["--algorithms"]),
        dict((algo, LitAlgorithmReasoner) for algo in args["--algorithms"]),
        False,  # args['--ensure-permutation'] is False for non-TSP
        reduce_proc_hid_w_MLP=args["--RPHWM"],
        update_edges_hidden=config["update_edges_hidden"],
        use_TF=False,
        use_gate=True,
        use_LSTM=args["--use-LSTM"],
        freeze_proc=False,  # We don't have a transfer task
        processors=args["--processors"],
        xavier_on_scalars=args["--xavier-on-scalars"],
        biased_gate=args["--biased-gate"],
        test_with_val_every_n_epoch=1,
        config=config,
    )

    all_cbs = get_callbacks(name, serialised_models_dir, args["--patience"])
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=config["max_epochs"],
        callbacks=[ModelSummary(max_depth=10)] + all_cbs,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        gradient_clip_val=args["--gradient-clip-val"],
        logger=pl.loggers.WandbLogger(
            project="nardeq",
            entity="clrs-cambridge", # NOTE CHANGE HERE
            log_model=True,
            group=None,
            settings=wandb.Settings(code_dir="."),
            save_code=True,
        ),
        # profiler="simple",
    )
    maybe_remove(f"./serialised_models/best_{name}.ckpt")
    maybe_remove(f"./serialised_models/{name}-epoch_*.ckpt")
    trainer.test(
        model=lit_processor,
    )
    trainer.fit(
        model=lit_processor,
    )
    trainer.test(
        ckpt_path="best",
    )


if __name__ == "__main__":
    schema = schema.Schema(
        {
            "--help": bool,
            "--xavier-on-scalars": bool,
            "--biased-gate": bool,
            "--update-edges-hidden": bool,
            "--use-LSTM": bool,
            "--patience": schema.Or(None, schema.Use(int)),
            "--model-name": schema.Or(None, schema.Use(str)),
            "--processors": schema.And(
                schema.Use(lambda x: x.split(",")),
                lambda lst: all(x in _PROCESSSOR_DICT for x in lst),
            ),
            "--RPHWM": bool,
            "--gradient-clip-val": schema.Use(int),
            "--algorithms": schema.Use(lambda x: x.split(",")),
            "--seed": schema.Use(int),
            "--config-path": schema.Or(None, os.path.exists),
            "--agentify": schema.Or(None, schema.Use(str)),
        }
    )
    args = docopt(__doc__)
    args = schema.validate(args)

    torch.multiprocessing.set_sharing_strategy("file_system")
    if args["--agentify"] is None:
        train(None, args=args)
    else:
        wandb.agent(
            args["--agentify"],
            function=lambda: train(None, args=args),
            entity="clrs-cambridge", # NOTE CHANGE HERE
            project="nardeq",
        )
