# Deep Equilibrium Algorithmic Reasoning
Official code repository for the paper [Deep Equilibrium Algorithmic Reasoning](www.google.com).

## Key files/locations

- `datasets/`: Code responsible for handling the datasets
- `layers/`+`models/`: Our model's implementations are in those two locations. We have aimed to keep to the following "rule": If a class is NOT a [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) class, it is responsible for processing a datapoint, but NOT for loss computation, dataloaders, etc. If it is a [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) class, then the code related to loss computation/dataloaders/logging/etc. is likely to be there, but not the code for processing (neurally executing) an input.
- `prepare_datasets.py`: Script to generate and preprocess the data used for training and testing.
- `train_reasoner.py`: Main script for training neural algorithmic reasoners.
- `test_*.py`: Scripts for testing models. (`*` denotes a wildmark, there is no file `test_*.py`)
- `configs/*.yaml`: Configuration files for our experiments. (We have intentionally used `.yml` for the `conda` environment).

Any models that you train will be saved in the directory in the format `best_`+given name. If you do not explicitly provide a name (using `--model-name`), the date+time at the time of the **starting** of the training script is used.

We also provide plotting scripts we have used, but you may need to adjust them yourself akin to test scripts (cf. below).

## Environment installation

To install an environment (`dear`) with packages needed, do the following:

1. Download [CLRS-30](https://github.com/google-deepmind/clrs) repository to your home directory (using `git clone`). E.g. for me (Dobrik) this is `/home/dobrik/clrs`. You will need to change the path in `gpuenv.yml` to point to your path to CLRS-30
1. Install the conda environment and activate it
   ```
   conda env create -f gpuenv.yml
   conda activate dear
   ```
1. Downgrade `protobuf`:
   ```
   pip install protobuf==3.20
   ```

## Preparation

Run `prepare_datasets.py` to generate the datasets required for training and
evaluating the models. Alternatively, you can
[download](https://mega.nz/file/nN0XADQI#xohgBdOKa54u6dQw4MzJhJOS6fwkrGONyY3BUlC__Kw),
move it to the repository and extract it (`tar -xvzf dataclrs.tar.gz`). The
tarball also contains larger datasets, so you don't have to wait.

If you choose to use `wandb`, our choice of logging tool, you'll also need to
change the entity in the training and testing scripts. We should have added
NOTE-s to where the flags are, but if we have omitted some, let us know. If you prefer to NOT use wandb run:
```
wandb offline
wandb disabled
```

## Training and testing models

For training, our models utilise config files. E.g. if you want to train a model to execute BFS, that also uses deep equilibrium reasoning, run:
```bash
python train_reasoner.py --algorithms bfs --config-path configs/deq.yaml
```
The `algo_cls` flag controls the class of NAR model -- `'deq'` for DEAR and `'normal'` for standard.

Testing is a little bit more complicated, as we attempted to automate our work as much as possible. We provide two scripts: 
- `test_reasoner.py` will test models that are only on your machine (cf. `MODEL_PATHS` variable), i.e. you either need to download them manually from `wandb` or to have them trained on the *same* machine before testing.
- `test_reasoner2.py` is an upgraded version, which integrates with `wandb` (remember to change the entity). Simply pass a list of the `wandb` run names, e.g. for 3 seeds and one algorithm, that is a list of 3 names -- cf. `EXAMPLE_LIST` in the script.
  
  As it is unlikely that you will be able to access our `wandb` project, please reach out if you want any pretrained models.

---
For more detailed instructions and documentation, refer to the individual script files and comments within the code. `[OPTIONS]` for each script can be viewed in the beginning of each file or by calling the script with the `--help` command.
