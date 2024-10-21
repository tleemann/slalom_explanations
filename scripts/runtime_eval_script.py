## Evaluate the runtime of different approaches for computing local explanations.
## Arguments <numsamples_eval> <num_queries> [<BLOOM>] 


import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
from slalom_explanations.attribution_methods import InputGradEstim, LIMEExplanation, SLALOMLocalExplanantions, ShapleyValues, LRPExplanation
import numpy as np
import time
from collections import defaultdict
import json
from scripts.faithfulness_eval import load_transformer_model


num_evals = int(sys.argv[1])
num_samples = int(sys.argv[2])
use_device = "cuda:3"

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
    run = "imdb_distilbert_2_r7"
    model_obj, mylrpmodel, tokenizer, use_cls = load_transformer_model(run, use_device)
    ## initiate explainers
    my_shap = ShapleyValues(model_obj.model, tokenizer, method="kernel", device=use_device, num_samples=num_samples)
    my_lime = LIMEExplanation(model_obj.model, tokenizer, device=use_device, n_samples=num_samples)
    myigradestim = InputGradEstim(model_obj.model, imdb, tokenizer, agg_norm=False, use_ig=True, ig_steps=20, times_input=False, device=use_device)
    mygrad = InputGradEstim(model_obj.model, imdb, tokenizer, agg_norm=False, use_ig=False, ig_steps=20, times_input=False, device=use_device)
    sle2 = SLALOMLocalExplanantions(model_obj.model, device=use_device, n_samples = num_samples, sgd_lr=8e-3, sgd_epochs=20, modes=["lin"], use_cls=use_cls, seq_len=2)
    sle_faith = SLALOMLocalExplanantions(model_obj.model, device=use_device, n_samples = num_samples, modes=["lin"], sampling_strategy="deletion", use_cls=use_cls, fit_sgd=False)
    lrp = LRPExplanation(mylrpmodel, normalize = False, mode = "classlogit", use_cls=use_cls, device=use_device)

#target_labels = imdb["test"]["label"]
explainers = {"SHAP": my_shap, "LIME": my_lime, "IG": myigradestim, "Grad": mygrad, "SLALOM-eff": sle2, "SLALOM-faith": sle_faith, "LRP": lrp}

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

json.dump(recordings, open(f"metrics/runtime_{num_evals}_{num_samples}.json", "w"))