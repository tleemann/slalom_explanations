import torch.nn.functional as F
from collections import defaultdict
import torch
import numpy as np
from scipy.stats import spearmanr
from copy import deepcopy
import numpy as np

def predict_removal_from_expl(removal_mask: torch.tensor, input_expl: np.ndarray):
    """ predict removal score from explanation. """
    avail_tokens = 1-removal_mask
    if input_expl.shape[0] == 1: # linear scoring
        print("lin")
        new_score = np.sum(avail_tokens.numpy()*input_expl, axis=1)
        old_score = np.sum(input_expl)
        diff = new_score - old_score
    else: #SLALOM model score
        print("slalom")
        s_expand = torch.tensor(np.expand_dims(input_expl[1, :], 0), dtype=torch.float)*torch.ones(removal_mask.shape[0], 1)
        s_expand[removal_mask==1] = torch.finfo(torch.float).min
        new_score = torch.sum(torch.softmax(s_expand, dim=1)*torch.tensor(input_expl[0,:], dtype=torch.float).reshape(1,-1), axis=1)
        new_score = new_score.numpy()
        old_score = np.sum(torch.softmax(torch.tensor(input_expl[1,:], dtype=torch.float), dim=-1).numpy()*input_expl[0,:])
        diff = new_score - old_score
    return diff


def multi_removal_curve(model, input_tokens, input_expl, input_classes=None, use_cls=True, 
        device="cuda:3", batch_size = 32, logits=True, max_deletion=5, deletion_samples=20, slalom_idx=[0]):
    """ Compute the multiple deletion curve metric. If input_expl is a single score, it is treated as a linear model.
        If the index is in the list slalom_idx, [index, index+1] scores are treated together as a SLALOM model.
        input_expl: list of (N, numtok) or (N, numtok) tensor with explanations. SLALOM is used to compute deletion the first index is in slalom_idx
        input_tokens: list of tokens
    """
    model.eval()
    # First obtain original score as a reference.
    all_res_lists = []
    #print(input_tokens)
    for sid in range(len(input_tokens)):
        #print("SID", sid)
        if use_cls:
            org_inp =  torch.tensor([101] + input_tokens[sid] + [102])
        else:
            org_inp =  torch.tensor(input_tokens[sid])
        org_inp = org_inp.reshape(1,-1)
        org_score = model(org_inp.to(device), attention_mask = torch.ones_like(org_inp).to(device))["logits"].detach().cpu()
        if logits:
            org_target = org_score[:,1]-org_score[:,0]
        else:
            org_target = torch.softmax(org_score, dim=-1)
            org_target = org_target[:,input_classes[sid]]
        org_target = org_target.cpu().numpy()
        mse_list = []
        print("org_target", org_target)
        for num_deleted in range(1, max_deletion+1):
            sample_mask = torch.zeros(deletion_samples, len(input_tokens[sid]), dtype=torch.long)
            for k in range(deletion_samples):
                gen = torch.manual_seed(k*42)
                sample_mask[k, torch.randperm(len(input_tokens[sid]), generator=gen)[:num_deleted]] = 1 # mark deleted
            
            ## Set up inputs with tokens deleted
            all_inps = torch.tensor(input_tokens[sid]).repeat(deletion_samples, 1)[sample_mask == 0].reshape(deletion_samples, -1)
            #print(all_inps.shape)
            if use_cls:
                start_tok = 101
                end_tok = 102
                all_inps = torch.cat((start_tok *torch.ones(len(all_inps), 1, dtype=torch.long), all_inps, end_tok*torch.ones(len(all_inps), 1, dtype=torch.long)), dim=1)
            #print(all_inps)
            ## Forward pass
            with torch.no_grad():
                p_list=[]
                for i in range(0, len(all_inps), batch_size):
                    probas = model(all_inps[i:i+batch_size].to(device), attention_mask = torch.ones_like(all_inps[i:i+batch_size]).to(device))["logits"].detach().cpu()
                    p_list.append(probas)
                
            probas = torch.cat(p_list, dim=0)
            if logits:
                probas_target = probas[:,1]-probas[:,0]
                sign=1
            else:
                probas = torch.softmax(probas, dim=-1)
                if input_classes is None:
                    probas_target = probas[:,1] ## Reference class 1.
                    sign=1
                else:
                    probas_target = probas[:,input_classes[sid]]
                    sign = 2*input_classes[sid]-1
            #print(probas_target)
            explanations = input_expl[sid]
            num_explanations = explanations.shape[0]
            #print(num_explanations)
            explmse_list = []
            for i in range(num_explanations):
                if i in slalom_idx:
                    use_expl = explanations[i:i+2, :]
                elif i-1 not in slalom_idx: # not already processed with previous slalom.
                    use_expl = explanations[i:i+1, :]
                else:
                    continue
                pred_target = predict_removal_from_expl(sample_mask, use_expl)
                #print(pred_target, (probas_target.numpy()-org_target))
                mse = np.power((probas_target.numpy()-org_target)-pred_target, 2).mean()
                explmse_list.append(mse)
            mse_list.append(explmse_list)
            print(mse_list)
        all_res_lists.append(mse_list)
    return all_res_lists


def compute_insertion_deletion(model, input_tokens, input_ranks, input_classes, maxlen=512, morf=True, step = 10, use_cls=True, device="cuda", batch_size=32):
    """ 
        Compute Insertion and Deletion Faithfulness Metrics. 
    """
    input_rankings = deepcopy(input_ranks)
    if morf == False: ## Invert ranking
        for k in range(len(input_rankings)):
            input_rankings[k] = input_rankings[k].shape[1] - 1 - input_rankings[k]

    num_expl = len(input_rankings[0])
    iters = []
    resdict = defaultdict(list)
    for i in range(step, maxlen, step): ## Loop through lengths
        iters.append(i)
        all_inps = []
        sid_list = []
        for sid in range(len(input_tokens)):
            if len(input_tokens[sid]) >= i:
                res_inps = torch.tensor(input_tokens[sid]).repeat(num_expl, 1)[torch.from_numpy(input_rankings[sid]) < i].reshape(num_expl, -1)
                #print(res_inps.shape)
                all_inps.append(res_inps)
                sid_list.append(sid)
        if len(all_inps) > 0:
            all_inps = torch.cat(all_inps, dim=0)
            if use_cls:
                start_tok = 101
                end_tok = 102
                all_inps = torch.cat((start_tok *torch.ones(len(all_inps), 1, dtype=torch.long), all_inps, end_tok*torch.ones(len(all_inps), 1, dtype=torch.long)), dim=1)
            with torch.no_grad():
                output_list = []
                for i in range(0, len(all_inps), batch_size):
                    output_list.append(model(all_inps[i:i+batch_size].to(device), attention_mask = torch.ones_like(all_inps[i:i+batch_size]).to(device))["logits"])
                outputs = torch.cat(output_list, dim=0)
            probas = torch.softmax(outputs, dim=-1).detach().cpu()

            input_classes_rep = (input_classes[sid_list].reshape(-1,1)*torch.ones(1,num_expl)).flatten().long()
            probas_target = torch.sum(F.one_hot(input_classes_rep, num_classes=2)*probas, dim=1)
            #print(probas_target.shape)
            for j, s in enumerate(sid_list):
                resdict[s].append(probas_target[j*num_expl:(j+1)*num_expl])
    return resdict, torch.tensor(iters)


def compute_auc(resdict):
    res_list = []
    for k, v in resdict.items():
        v = torch.stack(v, dim=0)
        res_list.append(v.mean(dim=0, keepdim=True))
    return torch.cat(res_list, dim=0)
        

def spearman_removal_corr_metric(model, input_tokens, input_expl, input_classes=None, use_cls=True, device="cuda:2", batch_size = 32, logits=True):
    """ Compute spearman correlation between the explanation scores and the scores obtained when actually removing the tokens. """
    corr_list = []
    mse_list = []
    for sid in range(len(input_tokens)):
        all_inps = torch.tensor(input_tokens[sid]).repeat(len(input_tokens[sid]), 1)[torch.eye(len(input_tokens[sid]), dtype=torch.long) == 0].reshape(len(input_tokens[sid]), -1)
        print(all_inps.shape)
        if use_cls:
            start_tok = 101
            end_tok = 102
            all_inps = torch.cat((start_tok *torch.ones(len(all_inps), 1, dtype=torch.long), all_inps, end_tok*torch.ones(len(all_inps), 1, dtype=torch.long)), dim=1)
        p_list = []
        
        ## Forward pass original instance as reference
        if use_cls:
            org_inp =  torch.tensor([101] + input_tokens[sid] + [102])
        else:
            org_inp =  torch.tensor(input_tokens[sid])
        org_inp = org_inp.reshape(1,-1)
        org_score = model(org_inp.to(device), attention_mask = torch.ones_like(org_inp).to(device))["logits"].detach().cpu()
        if logits:
            org_target = org_score[:,1]-org_score[:,0]
        else:
            org_target = torch.softmax(org_score, dim=-1)
            org_target = org_target[:,input_classes[sid]]
        with torch.no_grad():
            for i in range(0, len(all_inps), batch_size):
                probas = model(all_inps[i:i+batch_size].to(device), attention_mask = torch.ones_like(all_inps[i:i+batch_size]).to(device))["logits"].detach().cpu()
                p_list.append(probas)
                
            probas = torch.cat(p_list, dim=0)
        if logits:
            probas_target = probas[:,1]-probas[:,0]
            sign=1
        else:
            probas = torch.softmax(probas, dim=-1)
            if input_classes is None:
                probas_target = probas[:,1] ## Reference class 1.
                sign=1
            else:
                probas_target = probas[:,input_classes[sid]]
                sign = 2*input_classes[sid]-1
        ## Compute MSE and Spearman corr
        mses = torch.mean(torch.pow((org_target-probas_target)-input_expl[sid], 2), axis=1)
        mse_list.append(mses.numpy())
        ## Compute Spearman corr
        input_rankings = np.argsort(np.argsort(-sign*input_expl[sid], axis=1), axis=1) # High value after removal = negative explanation
        rankings = torch.argsort(torch.argsort(probas_target))
        corr_list.append(spearmanr(rankings, input_rankings.T).statistic[0,1:])
    print(mse_list[0].shape)
    return np.stack(corr_list), np.stack(mse_list)


def correlation_with_gt(input_tokens, input_expl, tokenizer, dataset, model_name, gt_use="nb", slalom_idx=[]):
    """ Compute correlation with ground truth metrics. """
    gt_tokens = torch.load(f"ground_truth/gt_{model_name}_{dataset}.pt")
    importances = gt_tokens[gt_use]
    special_tokens = tokenizer.all_special_tokens
    all_res_lists = []
    for sid in range(len(input_tokens)):
        decoded_tokens = []
        for input in input_tokens[sid]:
            decoded_tokens.append(tokenizer.convert_ids_to_tokens([input])[0])
        
        tokens_use = [] # List of tokens used for computing corrlations
        use_idx = [] # Their idx
        use_imps_mult = []
        for i, token_str in enumerate(decoded_tokens):
            if token_str in importances:
                if token_str in tokens_use:
                    continue
                use_idx.append(i)
                tokens_use.append(token_str)
                #use_imps_bin.append(importances_nb_bin[token_str])
                use_imps_mult.append(importances[token_str])
            else:
                print("No ground truth found for token: ", token_str)
                
        use_imps_mult = np.array(use_imps_mult)
        use_idx = np.array(use_idx)
        num_explanations = input_expl[sid].shape[0]
        explmse_list = []
        print(use_imps_mult.shape, input_expl[sid][0, use_idx].shape)
        for i in range(num_explanations):
            if i-1 in slalom_idx: ## Usigned
                explmse_list.append(spearmanr(np.abs(use_imps_mult), input_expl[sid][i, use_idx])[0])
            else:
                explmse_list.append(spearmanr(use_imps_mult, input_expl[sid][i, use_idx])[0])
        all_res_lists.append(explmse_list)
    return all_res_lists
        

                    

        