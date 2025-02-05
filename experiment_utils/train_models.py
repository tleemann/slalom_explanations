from experiment_utils.transformer_models import DistilBert, GPT2, Bert, RoBERTa, Trainer
from experiment_utils.attribution_methods import BoW 
from experiment_utils.utils import Correlation, Logger, LoggerAttMat
from experiment_utils.json_logging import JSONLogger
from experiment_utils.bow_dataset import BoWDataset, SparseBoWDataset, SLALOMDataset
from datasets import load_dataset, Dataset, DatasetDict
import torch
import os
import numpy as np
from transformers import AutoTokenizer
import seaborn as sns
import itertools
import argparse
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
import os

#from line_profiler import LineProfiler

def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batchsize to use', default=5)
    parser.add_argument('--epochs', type=float, help='number of epochs to train', default=1.0)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
    parser.add_argument('--layers', type=int, help='number of layers to use', default=6)
    parser.add_argument('--ground_truth', type=str, nargs="+", help='which ground truth importance to use, options: svm, nb, lr', default=[])
    parser.add_argument('--dataset', type=str, default="imdb", help='dataset. supported datsets are imdb, yelp, imdbbow, sparsebow, slalom, slalom200')
    parser.add_argument('--device', type=str, help='device to use for training', default="cuda:0")
    parser.add_argument('--model', type=str, help='model architectures', default="distilbert")
    parser.add_argument('--n_heads', type=int, help='model number of attention heads', default=12)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--logfile', type=str, help='filename for resultsfile', default='/mnt/ssd3/tobias/AttentionMatricesRaw/log.json')
    parser.add_argument('--model_path', type=str, help='where to store the trained model', default='/mnt/ssd3/tobias/AttentionMatricesRaw/models')
    parser.add_argument('--evalevery', type=bool, help='evaluate correlation in every epoch', default=False)
    parser.add_argument('--log_attn_steps', type=int, help='log attention matrices every number of steps, 0 means no logging', default=0)
    parser.add_argument('--max_att_logs', type=int, help='maximum number of batches that the att matrices are logged in ech test intervall, -1 means no limit', default=-1)
    parser.add_argument('--vocab_size', type=int, help='number of tokens in the vocabulary to use for the synthetic SLALOM-200 dataset', default=10)
    parser.add_argument('--sample_len', type=int, help="maximum length of the samples for synthetically generated datasets.", default=30)
    parser.add_argument('--fixed_len', type=bool, help="use fixed length of the samples for synthetically generated datasets.", default=False)
    parser.add_argument('--runid', type=str, help="name run to identify it in logfiles", default="0")
    args = parser.parse_args()
    return args

def compute_ref_importances(config, model, use_dataset, tokenizer, max_seq_len) -> dict:
    """ Compute reference importance scores """
    # define BoW model and train it
    if len(config.ground_truth) == 0:
        return {}

    print(f"Tokenizing with max_seq_len = {max_seq_len}")
    bow = BoW(ds=use_dataset, tokenizer=tokenizer, max_seq_len=max_seq_len)
    bow_grad = BoWGrad(model=model, ds=use_dataset, tokenizer=tokenizer, max_seq_len=max_seq_len)
    importances = {}
    for mygt in config.ground_truth:
        print("getting ground_truth for model", mygt)
        bow_svm = get_groundtruth_importance(mygt, bow, bow_grad)
        importances[mygt] = deepcopy(bow_svm.get_importance())

    print("Got reference importances.")
    return importances


def main(config = None):
    if config is None:
        config = arg_parse()

    device = config.device
    print("device=", device, "cwd", os.getcwd())
    # Init model
    if config.model == "distilbert":
        model = DistilBert(n_layers=config.layers, n_heads=config.n_heads, pretrained=config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
        max_seq_len = 512
    elif config.model == "bert":
        model = Bert(n_layers=config.layers, n_heads=config.n_heads, pretrained=config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
        max_seq_len = 512
    elif config.model == "gpt2":
        model = GPT2(n_layer=config.layers, n_head=config.n_heads, pretrained=config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        max_seq_len = 1024
        if tokenizer.pad_token is None:
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # default to left padding
            # tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
    elif config.model == "roberta":
        model = RoBERTa(n_layers=config.layers, n_heads=config.n_heads, pretrained=config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', use_fast=True)
        max_seq_len = 510
    else:
        raise NotImplementedError(f"model {model} not supported.")

    # Load dataset
    req_tokenization = True
    if config.dataset == "imdb":
        imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
        imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
        imdb["test"] = imdb["test"].shuffle(seed=42).select(range(50))
        # delete the unsupervised dataset (for speedup)
        #imdb.pop("unsupervised")
        dataset = imdb
    elif config.dataset == "yelp":
        yelp = load_dataset('yelp_polarity').with_format('torch', device='cpu')
        yelp["train"] = yelp["train"].shuffle(seed=42).select(range(5000))
        yelp["test"] = yelp["test"].shuffle(seed=42).select(range(50))
        #yelp = yelp.filter(lambda x: x["label"] != 2) # remove neutral reviews (scores range from 0 - 4)
        #yelp = yelp.map(lambda x: {"label": 0 if x["label"] < 2 else 1}) # translate scores into bad (0) or good (1)
        dataset = yelp
    elif config.dataset == "imdbbow":
        # IMDB dataset with analytical bow distribution
        imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
        imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
        imdb.pop("unsupervised")
        dataset = {}
        dataset["train"] = BoWDataset(imdb, tokenizer, ds_length=10000)
        ds_copy = deepcopy(dataset["train"])
        ds_copy.length = 100
        dataset["test"] = ds_copy
        req_tokenization = False
    elif config.dataset == "sparsebow": ## For experiment A (Motivation)
        words_dict = {"the": (0.2, 0.0), "we": (0.2, 0.0), "movie": (0.2, 0.0), "watch": (0.2, 0.0),
              "best": (0.06, 1.0), "perfect": (0.06, 1.5), "good": (0.08, 0.7),
              "worst": (0.06, -1.5), "poor": (0.06, -1.0),"ok": (0.08, -0.7)}
        if config.fixed_len:
            fixed = True
            binomial = False
            print("using fixed sample length.")
        else:
            fixed = False
            binomial = True
        if config.model == "distilbert" or config.model=="bert":
            myds = SparseBoWDataset(tokenizer, all_words_dict =  words_dict, fixed_len=fixed, binomial_sampling=binomial)
            myds_test = SparseBoWDataset(tokenizer, all_words_dict =  words_dict, ds_length=50, fixed_len=fixed, binomial_sampling=binomial)
        else:
            myds = SparseBoWDataset(tokenizer, all_words_dict =  words_dict, start_tok=None, end_tok=None, fixed_len=fixed, binomial_sampling=binomial)
            myds_test = SparseBoWDataset(tokenizer, all_words_dict =  words_dict, ds_length=50, fixed_len=fixed, binomial_sampling=binomial)
        dataset = {"train": myds, "test": myds_test}
        req_tokenization = False
    elif config.dataset == "slalom" or config.dataset == "slalom200":
        if config.dataset == "slalom200":
            words_dict = torch.load("datasets/slalom_200words.pt")
            ## Vocab size
            words_dict = {k: words_dict[k] for k in list(words_dict.keys())[:config.vocab_size]}
        else:
            words_dict = {"best": (0.3, 0.6, 1.1), "the": (0.4, -1.0, 0.7), "worst": (0.3, 0.4, -1.6)}
        myds = SLALOMDataset(tokenizer, all_words_dict =  words_dict, ds_length=20000, start_tok="[CLS]", sample_max_length=config.sample_len)
        myds_test = SLALOMDataset(tokenizer, all_words_dict =  words_dict, ds_length=50, start_tok="[CLS]", sample_max_length=config.sample_len)
        dataset = {"train": myds, "test": myds_test}
        req_tokenization = False
    else:
        raise ValueError(f"Unsupported dataset {config.dataset}")

    logger = JSONLogger(config.logfile, ["dataset", "model", "layers", "pretrained", "runid"])

    ref_importances = None

    # logger for attention matrices
    run_name=f'{config.dataset}_{config.model}_{config.layers}{"" if not config.pretrained else "_pretrained"}_r{config.runid}'

    if isinstance(dataset, DatasetDict):
        dataset["train"] = dataset["train"].shuffle(seed=(int(hash(config.runid)) % 65536))

    trainer = Trainer(ds=dataset, model=model, tokenizer=tokenizer, logger=logger, device=device, max_seq_len=max_seq_len, tokenization_required=req_tokenization)
    
    corrs = []
    step_cnt = 0
    split_ep = config.epochs-int(config.epochs)
    for ep in range(int(config.epochs)):
        print(step_cnt)
        step_cnt = trainer.train(epochs=1, test_interval=1, batch_size=config.batch_size, lr=config.lr, 
                logsteps = config.log_attn_steps, start_steps = step_cnt, start_epoch=ep)
        model_trained = trainer.model
    if split_ep > 0.0:
        step_cnt = trainer.train(epochs=split_ep, test_interval=1, batch_size=config.batch_size, lr=config.lr, 
                logsteps = config.log_attn_steps, start_steps = step_cnt, start_epoch=int(config.epochs))
        model_trained = trainer.model
    logger.update_result(trainer.last_acc, dataset=config.dataset, layers=config.layers, model=config.model, pretrained=config.pretrained, runid=config.runid)
    # save model
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    torch.save(model_trained.state_dict(), f'{config.model_path}/{run_name}.pt')
    #logger.update_result(corrs, model=config.model, layers=config.layers, heads=config.n_heads)

    print("Run-ID: ", run_name)

if __name__ == '__main__':
    main()