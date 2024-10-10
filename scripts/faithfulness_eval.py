import argparse
import json
import torch
from slalom_explanations.transformer_models import DistilBert, GPT2, Trainer, Bert, RoBERTa
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from slalom_explanations.attribution_methods import InputGradEstim, LIMEExplanation, SLALOMLocalExplanantions, ShapleyValues, LRPExplanation, ZeroExplanation, LinearRegressionDeletion
from slalom_explanations.metrics import compute_insertion_deletion, spearman_removal_corr_metric, compute_auc
from slalom_explanations.metrics import multi_removal_curve, correlation_with_gt
from slalom_explanations.reference_models import TFIDFModel, LSTMEmbeddings
from lxt.models.bert import BertForSequenceClassification as LRPBert, attnlrp as bert_attnlrp
from lxp_models.distilbert import DistilBertForSequenceClassification as LRPDistilBertForSequenceClassification, attnlrp as distilbert_attnlrp
from lxp_models.gpt2 import GPT2ForSequenceClassification as LRPGPT2ForSequenceClassification, attnlrp as gpt2_attnlrp

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
    parser.add_argument('--run', type=str, nargs="+", help="list the model runs")
    parser.add_argument('--model_type', type=str, nargs="+", help="list the model types. Number of types should be the same as model.")
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
        args["pad_token_id"] = tokenizer.pad_token_id
        return SLALOMLocalExplanantions(**args)
    elif key=="LinearRegressionDeletion":
        args["use_cls"] = use_cls
        args["pad_token_id"] = tokenizer.pad_token_id
        return LinearRegressionDeletion(**args)
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

def load_transformer_model(run, device):
    prefix = "/mnt/ssd3/tobias/"
    #run = metrics_config["model"][1]
    s_dict = torch.load(f"{prefix}AttentionMatricesRaw/models/{run}.pt", map_location="cpu")
    # extract num_layers, num_heads
    parts = run.split("_")
    n_heads = 12
    pretrained = ("pretrained" in run)
    n_layers = int(parts[2])
    model_name = parts[1]
    use_cls = False
    if "gpt2" in run:
        model_obj = GPT2(n_layer=n_layers, n_head=n_heads, pretrained=pretrained)
        #print("output attn config:", model_obj.model.config.output_attentions)
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True, padding=512)
        #if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elif "distilbert" in run:
        model_obj = DistilBert(n_layers=n_layers, n_heads=n_heads, pretrained=pretrained)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
        use_cls = True
    elif "roberta" in run:
        model_obj = RoBERTa(n_layers=n_layers, n_heads=n_heads, pretrained=pretrained)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', use_fast=True, padding=512)
        use_cls = True
    else:
        print("Loading BERT model.")
        model_obj = Bert(n_layers=n_layers, n_heads=n_heads, pretrained=pretrained)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
        use_cls = True
    model_obj.model.load_state_dict(s_dict)
    model_obj.model.to(device)
    model_obj.model.eval()
    model_to_explain = model_obj.model
    if "_bert_" in run: ##LRP...
        mylrpmodel = LRPBert.from_pretrained("bert-base-uncased").to(device)
        mylrpmodel.bert.encoder.layer = mylrpmodel.bert.encoder.layer[:len(model_to_explain.bert.encoder.layer)]
        ### Patch version inconsistency between HF version in this code and lrp code in state dict
        s_dict = model_to_explain.state_dict()
        s_dict['bert.embeddings.position_ids'] = mylrpmodel.bert.embeddings.position_ids

        mylrpmodel.load_state_dict(s_dict)
        bert_attnlrp.register(mylrpmodel)
    elif "distilbert" in run:
        mylrpmodel = LRPDistilBertForSequenceClassification(model_obj.model.config)
        mylrpmodel.distilbert.transformer.layer = mylrpmodel.distilbert.transformer.layer[:len(model_obj.model.distilbert.transformer.layer)]
        mylrpmodel.load_state_dict(s_dict)
        distilbert_attnlrp.register(mylrpmodel)
    elif "gpt2" in run:
        mylrpmodel = LRPGPT2ForSequenceClassification(model_obj.model.config)
        mylrpmodel.transformer.h = mylrpmodel.transformer.h[:len(model_obj.model.transformer.h)]
        mylrpmodel.load_state_dict(s_dict)
        gpt2_attnlrp.register(mylrpmodel)
    else:
        mylrpmodel = None
    return model_obj, mylrpmodel, tokenizer, use_cls

    

if __name__ == "__main__":
    config = arg_parse()
    device = config.device
    print("Using device:", device)
    metrics_config = json.load(open(config.config_file))
    ##
    if "model" not in metrics_config:
        metrics_config["model"] = []
    for modelidx in range(len(config.run)):
        metrics_config["model"].append((config.model_type[modelidx], config.run[modelidx]))
    print(metrics_config["model"])
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
    for modeltype, run in metrics_config["model"]:
        if modeltype == "Transformer":
            model_obj, mylrpmodel, tokenizer, use_cls = load_transformer_model(run, device)
            model_to_explain = model_obj.model
        elif modeltype == "TFIDFModel":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
            model_to_explain = TFIDFModel(dataset, tokenizer, metrics_config["model"][1])
        elif modeltype == "LSTMEmbeddings":
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
            model_to_explain = LSTMEmbeddings(tokenizer, metrics_config["model"][1], hidden_dim=60)
        elif modeltype == "BLOOM":
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
                input_tokens = tokenizer.encode(record["text"])
                lengths.append(len(input_tokens))
            lengths = np.array(lengths)
            imdb_test_use = dataset["test"].select(np.nonzero(lengths < 100)[0][:config.n_samples])
        else:
            imdb_test_use = dataset["test"].select(range(config.n_samples))
        print(f"Using {len(imdb_test_use)} samples.")

        if not config.skip_explanations:
            #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)
    
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
            slalom_idx = []
            if "slalom_idx" in metrics_config:
                slalom_idx = metrics_config["slalom_idx"]
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
                res = multi_removal_curve(model_to_explain, toks, expl_values, input_classes=None, use_cls=use_cls, device=device, batch_size = 32, logits=True, max_deletion=10, deletion_samples=100, slalom_idx=slalom_idx)
                metrics_res["multiremoval_mse"] = res
            if "groundtruth-nb" in metrics_config["metrics"]:
                res = correlation_with_gt(toks, expl_values, tokenizer, dataset=config.dataset, model_name=model_name, slalom_idx=slalom_idx)
                metrics_res["groundtruth-nb"] = res
            torch.save(metrics_res, metrics_config['metrics_file'] + f"{modeltype}_{run}.pt")