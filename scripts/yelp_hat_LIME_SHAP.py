from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slalom_explanations.transformer_models import DistilBert, GPT2, Bert
import torch
from datasets import load_dataset
import math
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import traceback
from scripts.faithfulness_eval import instantiate_explanation, load_transformer_model
import argparse
import json

from lxt.models.bert import BertForSequenceClassification as LRPBert, attnlrp as bert_attnlrp
from lxp_models.distilbert import DistilBertForSequenceClassification as LRPDistilBertForSequenceClassification, attnlrp as distilbert_attnlrp
from lxp_models.gpt2 import GPT2ForSequenceClassification as LRPGPT2ForSequenceClassification, attnlrp as gpt2_attnlrp
## arguments: <model> <BackgroundDSsize> [layers]
## argument 1: Bloom
    

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_samples', type=int, help='how many test samples to use')
    parser.add_argument('config_file', type=str, help='metrics config file to use')
    parser.add_argument('--device', type=str, help='which device to run on', default='cuda:1')
    parser.add_argument('--filter_length', type=int, default=-1, help="Filter length of the samples that are used")
    parser.add_argument('--run', type=str, nargs="+", help="list the model runs")
    parser.add_argument('--model_type', type=str, nargs="+", help="list the model types. Number of types should be the same as model.")
    args = parser.parse_args()
    return args


##### helper functions #####
def custom_tokenizer(text: str):
    tok_list = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    special_list = tokenizer.all_special_tokens
    toklist = list(filter(lambda a: a not in special_list, tok_list))
    #toklist = [t.strip(" ").strip("Ġ") for t in toklist]
    return toklist

def tokens_to_ids(sample):
    return torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample)))


def load_tokens(file_name, stop_sep=False):
    mylist = []
    with open(file_name, 'r') as file:
        for line in file:
            if not stop_sep or ((line.strip("\n") != "[PAD]") and not ('<|endoftext|>' in line)):
                mylist.append(line.strip("\n"))
    return mylist


def convert_token_list_to_inputs(inp_list, tok):
    """ TODO: make more solid implementation. """
    return tok.convert_tokens_to_ids(inp_list)


"""
Original Attention: List of 0's and 1's per word (not token!).
Steps: 
    - tokenize each word. split the word's og_attention over the tokens it got tokenized into.
    - if token appears multiple times in sample, its attention is being summed up from all occurrences.
    - finally, we build a dictionary (like bow gt) with each (unique) token having an attention score.
"""
def adjust_tok_to_hat_scores(og_text, og_attention):    
    tok_to_att = defaultdict(int) # final token-attention mapping
    pos = 0
    for tok in og_text.split():
        if not use_cls:
            if tok != og_text.split()[0]: ## Append space for every token that is not the first one.
                tok = " "+tok
        tokenized_word = tokenizer.convert_ids_to_tokens(tokenizer.encode(tok))
        ## Remove attention from punctuation tokens
        tokens_no_punc = []
        for t in tokenized_word:
            if t.strip(" ").strip("Ġ") in ["!", ".", ",", ";", ":", "?"]:
                continue
            else:
                tokens_no_punc.append(t)
        current_att = og_attention[pos]
        shared_att = current_att
        for t in tokens_no_punc:
            tok_to_att[t] = shared_att
        pos += 1
    return tok_to_att
            
if __name__ == "__main__":
    config = parseargs()
    metrics_config = json.load(open(config.config_file))
    device=config.device
    print(device)
    dataset="yelp"
    ##
    if "model" not in metrics_config:
        metrics_config["model"] = []
    for modelidx in range(len(config.run)):
        metrics_config["model"].append((config.model_type[modelidx], config.run[modelidx]))

    for model_vs, run in metrics_config["model"]:
        if model_vs == "BLOOM":
            device = "cuda:3"
            checkpoint = "bigscience/bloom-7b1"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=device, num_labels=2, torch_dtype="auto")

            print("Loading model...")
            trained_model_ckp = "/mnt/ssd2/tobias/models/bloom-7b1_trained_100000_1.pt"
            model.load_state_dict(torch.load(trained_model_ckp,  map_location=device))
            use_cls = False

            lime_expl = LIMEExplanation(model, tokenizer, device=device, n_samples=bg_ds_size)
            shap_expl = ShapleyValues(model, tokenizer, device=device, method="kernel", num_samples=bg_ds_size, impute_token="<pad>")
            use_cls_off = False
            layers = 1
        elif model_vs == "Transformer":
            model_obj, mylrpmodel, tokenizer, use_cls = load_transformer_model(run, device)
            model_to_explain = model_obj.model
        xai_methods = []
        for explainer_key, explainer_args in metrics_config["explanations"]:
            if "active" in explainer_args:
                if not explainer_args["active"]:
                    continue
                else:
                    del explainer_args["active"]
            print(explainer_key)
            xai_methods.append(instantiate_explanation(explainer_key, explainer_args, model_to_explain, tokenizer, dataset=dataset,\
            device=device, use_cls=use_cls, modellrp=mylrpmodel, pad_token = '<|endoftext|>'if "gpt2" in run else None))

        #toks, rankings, expl_values = compute_explanations_rankings(imdb_test_use, xai_methods)

        #lime_expl = LIMEExplanation(model_obj.model, tokenizer, device=device, n_samples=bg_ds_size)
        #if "distilgpt2" in run:
        #    shap_expl = ShapleyValues(model_obj.model, tokenizer, device=device, method="kernel", impute_token='<|endoftext|>', num_samples=bg_ds_size)
        #else:
        #    shap_expl = ShapleyValues(model_obj.model, tokenizer, device=device, method="kernel", num_samples=bg_ds_size)
        ##### Computing SLALOM-scores vs. human attention (HAT) correlations #####

        corr_dict = {}
        hat_versions = ["and"]
        sample_size = 2462

        ## we're taking 50 samples here, not all 2462
        sample_size = config.n_samples #500 if model_vs != "BLOOM" else 50
        print(f"sample size: {sample_size}")
        for version in hat_versions:
            lime_corrs = []
            shap_corrs = []
            values_vs_imp = []
            importance_vs_imp = []
            fuse_vs_imp = []

            yelp_hat = load_dataset("json", data_files=f"datasets/yelp_hat_processed_{version}.json", field="test").with_format('torch', device='cpu')
            random.seed = 42
            #for sample in random.sample(range(0, 2461), sample_size):
            samples_list = []
            for sample in range(0, sample_size):
                print("Current sample: ", sample)
                try:
                    hat_attention = adjust_tok_to_hat_scores(yelp_hat["train"]["text"][sample], yelp_hat["train"]["attention"][sample])
                    token_list_hat = custom_tokenizer(yelp_hat["train"]["text"][sample])
                    input_ids_hat = convert_token_list_to_inputs(token_list_hat, tokenizer)
                
                    use_idx =[]
                    use_imps = []
                    tokens_use = []
                    for i, token_str in enumerate(token_list_hat):
                        if token_str in hat_attention or token_str.strip(" ").strip("Ġ") in hat_attention:
                            if token_str in tokens_use:
                                continue
                            use_idx.append(i)
                            tokens_use.append(token_str)
                            use_imps.append(hat_attention[token_str])
                        else:
                            print("No ground truth found for token: ", token_str)
                            
                    use_imps = np.array(use_imps)
                    use_idx = torch.tensor(use_idx)
                    print(use_imps)
                    expl_scores = []
                    for my_expl in xai_methods:
                        print(len(input_ids_hat))
                        expls = my_expl.get_signed_importance_for_tokens(input_ids_hat)
                        if len(expls.shape) == 1:
                            expls = expls.reshape(1, -1)
                        expl_scores.append(expls)
                    expl_scores = np.concatenate(expl_scores, axis=0)
                    print("Shape of expl_scores: ", expl_scores.shape)
                    res_list = []
                    for i in range(len(expl_scores)):
                        if i-1 in metrics_config["slalom_idx"]: ## Unsigned-> Unsigned
                            res_list.append(roc_auc_score(use_imps, expl_scores[i, use_idx]))
                        else: ## Signed-> unsigned
                            res_list.append(roc_auc_score(use_imps, np.abs(expl_scores[i, use_idx])))
                    samples_list.append(res_list)
                    print(len(res_list))
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                print("Len:", len(samples_list))
                
        corr_dict[version] = samples_list
        
        print("Results:")
        print(corr_dict)
        import json
        import os
        res_filename = f"metrics/corr_100_yelp_results_{run}.json"
        json.dump(corr_dict, open(res_filename, "w"))