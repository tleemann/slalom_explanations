## Evaluate the runtime of different approaches for computing local explanations.
## Arguments <numsamples_eval> <num_queries> [<BLOOM>] 


import torch
from transformer_models import DistilBert, DistilGPT2, Trainer
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from ground_truth_models import InputGradEstim, LIMEExplanation, SLALOMLocalExplanantions, ShapleyValues
import numpy as np
import time
from collections import defaultdict
import json

num_evals = int(sys.argv[1])
num_samples = int(sys.argv[2])
use_device = "cuda:2"

imdb = load_dataset('imdb').with_format('torch', device="cpu") # format to pytorch tensors, but leave data on cpu
imdb["train"] = imdb["train"].shuffle(seed=42).select(range(5000))
imdb["test"] = imdb["test"].shuffle(seed=42).select(range(num_evals))

if len(sys.argv) > 3 and sys.argv[3] == "BLOOM":
    
    checkpoint = "bigscience/bloom-7b1"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=use_device, num_labels=2, torch_dtype="auto")

    print("Loading model...")
    trained_model_ckp = "/mnt/ssd3/tobias/models/bloom-7b1_trained_100000_1.pt"
    model.load_state_dict(torch.load(trained_model_ckp,  map_location=use_device))
    use_cls = True

    my_lime = LIMEExplanation(model, tokenizer, device=use_device, n_samples=num_samples)
    my_shap = ShapleyValues(model, tokenizer, device=use_device, method="kernel", num_samples=num_samples, impute_token="<pad>")
    sle2 = SLALOMLocalExplanantions(model, device=use_device, n_samples = num_samples, sgd_lr=8e-3, sgd_epochs=20, modes=["lin"])
    myigradestim = InputGradEstim(model, imdb, tokenizer, agg_norm=False, use_ig=True, ig_steps=8, times_input=False, device=use_device)
    mygrad = InputGradEstim(model, imdb, tokenizer, agg_norm=False, use_ig=False, ig_steps=8, times_input=False, device=use_device)
else:
    epoch = 99
    prefix = "/mnt/ssd3/tobias/"
    run = "2024-03-22-18:26:16_distilbert_6_12"
    #run = "2024-01-03-16:54:36_distilbert_2_12" # BOW run
    #run = "2024-01-03-18:45:49_distilbert_1_12" # BOW run
    s_dict = torch.load(f"{prefix}AttentionMatricesRaw/{run}/epoch_{str(epoch).zfill(3)}/model_trained.pt")

    # extract num_layers, num_heads
    parts = run.split("_")
    n_heads = int(parts[-1])
    n_layers = int(parts[-2])
    if "distilgpt2" in run:
        model_obj = DistilGPT2(n_layer=n_layers, n_head=n_heads)
    else:
        model_obj = DistilBert(n_layers=n_layers, n_heads=n_heads)
    model_obj.model.load_state_dict(s_dict)
    model_obj.model=model_obj.model.to(use_device)
    model_obj.model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, padding=512)

 

    ## initiate explainers
    my_shap = ShapleyValues(model_obj.model, tokenizer, method="kernel", device=use_device, num_samples=num_samples)
    my_lime = LIMEExplanation(model_obj.model, tokenizer, device=use_device, n_samples=num_samples)
    myigradestim = InputGradEstim(model_obj.model, imdb, tokenizer, agg_norm=False, use_ig=True, ig_steps=20, times_input=False, device=use_device)
    mygrad = InputGradEstim(model_obj.model, imdb, tokenizer, agg_norm=False, use_ig=False, ig_steps=20, times_input=False, device=use_device)
    sle2 = SLALOMLocalExplanantions(model_obj.model, device=use_device, n_samples = num_samples, sgd_lr=8e-3, sgd_epochs=20, modes=["lin"])

#target_labels = imdb["test"]["label"]
explainers = {"SHAP": my_shap, "LIME": my_lime, "IG": myigradestim, "Grad": mygrad, "SLALOM-Lin": sle2}

maxlen = 510
recordings = defaultdict(list)
for rid, record in enumerate(imdb["test"]):
    print("Running sample", rid)
    input_tokens = tokenizer.encode(record["text"])[1:-1]
    input_tokens= input_tokens[:maxlen]
    for expl_name, expl_object in explainers.items():
        start_time = time.time()
        expls = expl_object.get_signed_importance_for_tokens(input_tokens)
        end_time = time.time()
        passed_time = start_time-end_time
        recordings[expl_name].append(passed_time)

json.dump(recordings, open(f"results/runtime_{num_evals}_{num_samples}.json", "w"))



