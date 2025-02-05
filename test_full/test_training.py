## Test the training function. Train model on linear dataset for one epoch.
from experiment_utils.train_models import main, arg_parse
import sys
from copy import deepcopy
import os


def test_training_linear(monkeypatch):
    """Train a model on the linear dataset for one epoch."""
    old_argv = deepcopy(sys.argv)
    sys.argv = sys.argv[:1]  ## Reset argv to only feature the script name argument.
    args = arg_parse()  ## get empty arg struct
    ## Set some values:
    args.epochs = 1
    args.model = "bert"
    args.device = "cpu"
    args.n_heads = 12
    args.batch_size = 20
    args.layers = 2
    args.dataset = "sparsebow"
    args.model_path = os.getcwd()
    args.logfile = "log.json"

    main(args)

    ## Delete the model again
    os.remove("sparsebow_bert_2_r0.pt")
    os.remove("log.json")
    sys.argv = old_argv
