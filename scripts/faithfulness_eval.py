import argparse
import json
import torch
from slalom_explanations.transformer_models import DistilBert, DistilGPT2, Trainer, Bert
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from slalom_explanations.attribution_methods import InputGradEstim, LIMEExplanation, SLALOMLocalExplanantions, ShapleyValues, LRPExplanation, ZeroExplanation
from slalom_explanations.metrics import compute_insertion_deletion, spearman_removal_corr_metric, compute_auc
from slalom_explanations.metrics import multi_removal_curve

from slalom_explanations.reference_models import TFIDFModel, LSTMEmbeddings
from lxt.models.bert import BertForSequenceClassification as LRPBert, attnlrp

def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('n_samples', type=int, help='how many test samples to use')
    parser.add_argument('config_file', type=str, help='metrics config file to use')
    parser.add_argument('--device', type=str, help='which device to run on', default='cuda')
    parser.add_argument('--skip_explanations', type=bool, help='if passed, only compute the metrics', default=False)
    parser.add_argument('--skip_metrics', type=bool, help='if passed, only compute the explanations', default=False)
    parser.add_argument('--filter_length', type=int, default=-1, help="Filter length of the samples that are used")
    parser.add_argument('--dataset', type=str, default="imdb", help="which dataset to choose.")
    args = parser.parse_args()
    return args


def instantiate_explanation(key, args, model, tokenizer, dataset, device="cuda", use_cls=True, modellrp=None):
    args["device"] = device
    args["model"] = model
    if key == "LIME":
        args["tokenizer"] = tokenizer
        return LIMEExplanation(**args)
    elif key == "InputGrad":
        args["ds"] = dataset
        args["tokenizer"] = tokenizer
        return InputGradEstim(**args)
    elif key== "SHAP":
        args["tokenizer"] = tokenizer
        return ShapleyValues(**args)
    elif key== "LRP":
        del args["model"]
        args["modellrp"] = modellrp
        args["use_cls"] = use_cls
        return LRPExplanation(**args)
    elif key=="SLALOM":
        args["use_cls"] = use_cls
        return SLALOMLocalExplanantions(**args)
    elif key =="Zero":
        return ZeroExplanation()
    else:
        raise ValueError(f"Unknown explanation key {key}.")


def compute_explanations_rankings(input_dataset, explainer_list, maxlen = 510, use_cls=True):
    token_list = []
    ranking_list = []
    expl_score_list = []
    for record in input_dataset:
        if use_cls:
            input_tokens = tokenizer.encode(record["text"])[1:-1]
        else:
            input_tokens = tokenizer.encode(record["text"])
        input_tokens= input_tokens[:maxlen]
        token_list.append(input_tokens)
        expl_ranks = []
        expl_scores = []
        for my_expl in explainer_list:
            print(len(input_tokens))
            expls = my_expl.get_signed_importance_for_tokens(input_tokens)
            if len(expls.shape) == 1:
                expls = expls.reshape(1, -1)
            #if expls.shape[1] == 1:
            #if not isinstance(my_expl, LRPExplanation) or my_expl.mode != "classlogit": ## This explanation is already class specific.
            sign = 2*record["label"]-1
            ranks = np.argsort(np.argsort(-sign*expls, axis=-1), axis=-1)
            #else:
            #    print("Class logit.")
            #    ranks = np.argsort(np.argsort(-expls, axis=-1), axis=-1)
            print(ranks.shape)
            expl_ranks.append(ranks)
            #else:
            #    expl_ranks.append([])
            expl_scores.append(expls)
        expl_ranks = np.concatenate(expl_ranks, axis=0)
        ranking_list.append(expl_ranks)
        expl_scores = np.concatenate(expl_scores, axis=0)
        expl_score_list.append(expl_scores)
    return token_list, ranking_list, expl_score_list


if __name__ == "__main__":
    config = arg_parse()
    device = config.device
    metrics_config = json.load(open(config.config_file))

    if config.dataset == "imdb":
        imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
        imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
        imdb["test"] = imdb["test"].shuffle(seed=42).select(range(20000))
        dataset = imdb
    elif config.dataset == "yelp":
        yelp = load_dataset('yelp_polarity').with_format('torch', device='cpu')
        yelp["train"] = yelp["train"].shuffle(seed=42).select(range(5000))
        yelp["test"] = yelp["test"].shuffle(seed=42).select(range(20000))
        dataset = yelp
    else:
        raise ValueError(f"Unknown dataset {config.dataset}.")
    target_labels = dataset["test"]["label"]


    model_to_explain, tokenizer = None, None
    use_cls=False
    ## Need to define model_to_explain
    ## Need to define tokenzizer
    for run in metrics_config["model"][1]:
        if metrics_config["model"][0] == "Transformer":
            if "epoch" in metrics_config:
                epoch = int(metrics_config["epoch"])
            else:
                epoch = 9
            if "prefix" in metrics_config:
                prefix = metrics_config["prefix"]
            else:
                prefix = "/mnt/ssd3/tobias/"
            #run = metrics_config["model"][1]
            s_dict = torch.load(f"{prefix}AttentionMatricesRaw/{run}/epoch_{str(epoch).zfill(3)}/model_trained.pt")
            # extract num_layers, num_heads
            parts = run.split("_")
            n_heads = int(parts[-1])
            n_layers = int(parts[-2])
            if "distilgpt2" in run:
                model_obj = DistilGPT2(n_layer=n_layers, n_head=n_heads)
                tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True, padding=512)
            elif "distilbert" in run:
                model_obj = DistilBert(n_layers=n_layers, n_heads=n_heads)
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
                use_cls = True
            else:
                print("Loading BERT model.")
                model_obj = Bert(n_layers=n_layers, n_heads=n_heads)
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
                use_cls = True
            model_obj.model.load_state_dict(s_dict)
            model_obj.model.to(device)
            model_obj.model.eval()
            model_to_explain = model_obj.model
            if "_bert_" in run: ##LRP...
                mylrpmodel = LRPBert.from_pretrained("bert-base-uncased").to(device)
                mylrpmodel.bert.encoder.layer = mylrpmodel.bert.encoder.layer[:len(model_to_explain.bert.encoder.layer)]
                ### Patch version inconsistency in state dict
                s_dict = model_to_explain.state_dict()
                s_dict['bert.embeddings.position_ids'] = mylrpmodel.bert.embeddings.position_ids
                mylrpmodel.load_state_dict(s_dict)
                #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                #input_ids = torch.tensor([tokenizer.encode("This is a test.")])
                attnlrp.register(mylrpmodel)
                #record = dataset["test"][35]
                #input_tokens = tokenizer.encode(record["text"])[1:-1]
                #relevance = mlrpexpl.get_signed_importance_for_tokens(input_tokens)
                #relevance2 = mlrpexpl.get_signed_importance_for_tokens(input_tokens)
                #print(relevance, relevance2)
                #exit(0)
            else:
                mylrpmodel = None
        elif metrics_config["model"][0] == "TFIDFModel":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
            model_to_explain = TFIDFModel(dataset, tokenizer, metrics_config["model"][1])
        elif metrics_config["model"][0] == "LSTMEmbeddings":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
            model_to_explain = LSTMEmbeddings(tokenizer, metrics_config["model"][1], hidden_dim=60)
        elif metrics_config["model"][0] == "BLOOM":
            checkpoint = "bigscience/bloom-7b1"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=device, num_labels=2, torch_dtype="auto")
            print("Loading model...")
            trained_model_ckp = metrics_config["model"][1]
            model.load_state_dict(torch.load(trained_model_ckp,  map_location=device))
            model_to_explain = model
            use_cls = False
        else:
            raise ValueError("Unknown model.")
        print("use cls: ", use_cls)
        if config.filter_length > 0:
            lengths = []
            for i, record in enumerate(dataset["test"]):
                input_tokens = tokenizer.encode(record["text"])[1:-1]
                lengths.append(len(input_tokens))
            lengths = np.array(lengths)
            imdb_test_use = dataset["test"].select(np.nonzero(lengths < 100)[0][:config.n_samples])
        else:
            imdb_test_use = dataset["test"].select(range(config.n_samples))
        print(f"Using {len(imdb_test_use)} samples.")

        if not config.skip_explanations:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
    
            ## get explanation objects:
            xai_methods = []
            for explainer_key, explainer_args in metrics_config["explanations"]:
                if "active" in explainer_args:
                    if not explainer_args["active"]:
                        continue
                    else:
                        del explainer_args["active"]
                print(explainer_key)
                xai_methods.append(instantiate_explanation(explainer_key, explainer_args, model_to_explain, tokenizer, dataset=dataset,\
                device=device, use_cls=use_cls, modellrp=mylrpmodel))

            toks, rankings, expl_values = compute_explanations_rankings(imdb_test_use, xai_methods)
            

            torch.save((toks, rankings, expl_values), metrics_config['explanation_file'])
        else:
            toks, rankings, expl_values = torch.load(metrics_config['explanation_file'])

        if not config.skip_metrics:
            metrics_res = {}
            ## Compute metrics
            if "insertion" in metrics_config["metrics"]:
                resdict_insertion, _ = compute_insertion_deletion(model_to_explain, toks, rankings, target_labels, device=device, morf=True, step=5, batch_size=8, use_cls=use_cls)
                insertion_score = compute_auc(resdict_insertion)
                metrics_res["insertion"]  = insertion_score
            if "deletion" in metrics_config["metrics"]:
                resdict_deletion, _ = compute_insertion_deletion(model_to_explain, toks, rankings, target_labels, device=device, morf=False, step=5, batch_size=8, use_cls=use_cls)
                deletion_score = compute_auc(resdict_deletion)
                metrics_res["deletion"] = deletion_score
            if "removal" in metrics_config["metrics"]:
                res_corr, res_mse = spearman_removal_corr_metric(model_to_explain, toks, expl_values, logits=True, device=device, batch_size=8, use_cls=use_cls)
                metrics_res["removal_corr"] = res_corr
                metrics_res["removal_mse"] = res_mse
            if "multiremoval" in metrics_config["metrics"]:
                print("Compute Multiremoval.")
                res = multi_removal_curve(model_to_explain, toks, expl_values, input_classes=None, use_cls=True, device=device, batch_size = 32, logits=True, max_deletion=5, deletion_samples=100, slalom_idx=5)
                metrics_res["multiremoval_mse"] = res
            torch.save(metrics_res, metrics_config['metrics_file'] + (run if metrics_config["model"][0] == "Transformer" else metrics_config["model"][0]) + ".pt")