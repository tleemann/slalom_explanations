import os
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import spearmanr
from transformers import DistilBertForSequenceClassification
from typing import Union, List
from datetime import datetime
from collections import defaultdict
import torch


class Correlation(object):
    def __init__(self, att_matrix: np.ndarray, decoded_tokens: List, importances: Dict, special_tokens: List) -> None:
        """
            att_matrix: An attention matrix for a single sample, averaged over heads. shape [layers, seq_len, seq_len]
        """
        self.att_matrix = att_matrix
        self.decoded_tokens = decoded_tokens
        self.importances = importances
        self.special_tokens = special_tokens

    #@profile
    def average_attention_on_duplicates(self, att_mat: np.ndarray, start_token: List) -> np.ndarray:
        """Average attention weights on duplicate tokens"""

        '''
        # get all duplicated tokens
        duplicates = []
        for token in self.decoded_tokens:
            if self.decoded_tokens.count(token) > 1:
                if token not in duplicates:
                    duplicates.append(token)

        # get indices of duplicated tokens
        duplicates_indices = {}
        for token in duplicates:
            indices = [i for i, x in enumerate(self.decoded_tokens) if x == token]
            duplicates_indices[token] = indices
        '''

        
        # -------------------------------- speedup --------------------------------
        seen_tokens = set()
        duplicates = set()
        for token in self.decoded_tokens:
            if token in seen_tokens:
                duplicates.add(token)
            else:
                seen_tokens.add(token)

    
        duplicates_indices = defaultdict(list)
        for index, token in enumerate(self.decoded_tokens):
            if token in duplicates:
                duplicates_indices[token].append(index)
        # --------------------------------------------------------------------------
        
        
        
        # average over all duplicated tokens in the attention matrix
        #print(att_mat.shape)
        att_mat = att_mat[:,start_token]
        for token in duplicates_indices.keys():
            indices = duplicates_indices[token]
            for layer in range(att_mat.shape[0]):
                att_on_token = 0
                for idx in indices:
                    att_on_token += att_mat[layer, idx]
                att_on_token_mean = att_on_token / len(indices)
                # set the attention on the token to the mean of the attention on all duplicated tokens
                for idx in indices:
                    att_mat[layer, idx] = att_on_token_mean
        return att_mat
    
    #@profile
    def delete_special_tokens(self, att_mat: np.ndarray) -> Tuple[np.ndarray, List, Dict]:
        """Deletes special tokens from the given 2D attention matrix (not 3D!!)
        Returns:
        att_mat_wo_special_tokens: Attention matrix without special tokens
        decoded_tokens_wo_special_tokens: List of decoded tokens without special tokens
        importances_wo_special_tokens: Dictionary of importances without special tokens
        """
        # get special tokens that are in the BoW
        special_tokens_in_bow = []
        for token in self.special_tokens:
            if token in self.importances.keys():
                special_tokens_in_bow.append(token)
                
        # get special tokens that are in the decoded tokens
        special_tokens_in_decoded_tokens = []
        for token in self.special_tokens:
            if token in self.decoded_tokens:
                special_tokens_in_decoded_tokens.append(token)

        # get the indices of the special tokens in the decoded tokens list (get indices of duplicated special tokens too (GPT2))
        special_tokens_indices_decoded_tokens = []
        for token in special_tokens_in_decoded_tokens:
            indices = [i for i, x in enumerate(self.decoded_tokens) if x == token]
            special_tokens_indices_decoded_tokens.extend(indices)

        # delete special tokens from decoded_tokens
        decoded_tokens_wo_special_tokens = self.decoded_tokens.copy()
        decoded_tokens_wo_special_tokens = [token for idx, token in enumerate(self.decoded_tokens) if idx not in special_tokens_indices_decoded_tokens]

        # delete special tokens from attention matrix
        att_mat_wo_special_tokens = att_mat.copy()
        for idx in special_tokens_indices_decoded_tokens:
            att_mat_wo_special_tokens = np.delete(att_mat_wo_special_tokens, idx, axis=1)

        # delete special tokens from importances
        special_tokens_in_importance = []
        for token in self.special_tokens:
            if token in self.importances.keys():
                special_tokens_in_importance.append(token)

        importances_wo_special_tokens = self.importances.copy()
        for token in special_tokens_in_importance:
            del importances_wo_special_tokens[token]

        return att_mat_wo_special_tokens, decoded_tokens_wo_special_tokens, importances_wo_special_tokens

    #@profile
    def _compute_corr_scores(self, layer: int=-1, start_token: int=0):
        if layer == -1:
            att_mat = np.sum(self.att_matrix, axis=0)
            # add on dimension for layer
            att_mat = att_mat[None,...]
        else:
            att_mat = self.att_matrix[layer]
            att_mat = att_mat[None,...]
        
        # average attention on duplicate tokens
        att_mat = self.average_attention_on_duplicates(att_mat=att_mat, start_token=start_token)

        # delete special tokens from attention matrix
        if self.special_tokens:
            att_mat, decoded_tokens, importances = self.delete_special_tokens(att_mat=att_mat)
        else:
            decoded_tokens = self.decoded_tokens
            importances = self.importances
       
        # delete layer dimension
        att_mat = att_mat[0]

        imp = [importances[token] if token in importances else 0 for token in decoded_tokens]
        imp = np.array(imp)
        return att_mat, imp

    def compute_spearmanr(self, layer: int=-1, start_token: int=0) -> float:
        att_mat, imp = self._compute_corr_scores(layer, start_token)
        return spearmanr(att_mat, imp)[0]


class Logger(object):
    def __init__(self, log_name: str, model: Union[DistilBertForSequenceClassification, None]) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.log_path = 'logs/' + log_name + '_' + timestamp + '.csv'
        if not os.path.exists('logs'):
            os.makedirs('logs')

        if model is not None:
            self.model_name = model.__class__.__name__
        else:
            self.model_name = 'None'

        if model is not None:
            self.model_params = model.config.to_dict()
        else:
            self.model_params = {}

        with open(self.log_path, 'w') as f:
            f.write('Model name: '+self.model_name+'\n')
            f.write('Model parameters: '+str(self.model_params)+'\n')
            f.write('-'*50+'\n')

    def log(self, text: str) -> None:
        with open(self.log_path, 'a') as f:
            f.write(text+'\n')


class LoggerAttMat(object):
    def __init__(self, path: str, run_name: str, tokenizer) -> None:
        self.path = path
        self.run_name = run_name
        self.tokenizer = tokenizer
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
    def save_att_mat(self, epoch: int, batch_nr: int, model_output: Dict, input_ids: List) -> None:
        attentions = model_output['attentions']
        attentions = [att.detach().cpu().numpy() for att in attentions]
        att_mats = np.asarray(attentions)
        num_batches = att_mats.shape[1]  # second dimension is the batch dimension
        eps = 'epoch_' + str(epoch).zfill(3)
        for i in range(num_batches):
            att_mat = att_mats[:,i]
            nr = batch_nr * num_batches + i
            sample_nr = 'sample_' + str(nr).zfill(5)
            if epoch == 0:
                self.save_test_tokens(batch_nr, input_ids)
            p = os.path.join(self.path, self.run_name, eps, sample_nr)
            if not os.path.exists(p):
                os.makedirs(p)
            len_input = torch.nonzero(input_ids[i])[-1][0].item()
            np.save(os.path.join(p, 'att_mat.npy'), att_mat[:,:, :len_input+1, :len_input+1])
    
    def save_test_tokens(self, batch_nr, input_ids):
        """ save decoded tokens. """
         # need to save decoded tokens only once
        num_batches = len(input_ids)
        p = os.path.join(self.path, self.run_name, 'decoded_tokens')
        if not os.path.exists(p):
            os.makedirs(p)

        for i in range(num_batches):
            #att_mat = att_mats[:,i]
            nr = batch_nr * num_batches + i
            sample_nr = 'sample_' + str(nr).zfill(5)
            decoded_tokens = []
            for input in input_ids[i]:
                decoded_tokens.append(self.tokenizer.convert_ids_to_tokens([input])[0])
            with open(os.path.join(p, sample_nr+'.txt'), 'w') as f:
                f.write('\n'.join(decoded_tokens))

    def save_model(self, model_trained, epoch):
        eps = 'epoch_' + str(epoch).zfill(3)
        p = os.path.join(self.path, self.run_name, eps)
        if not os.path.exists(p):
            os.makedirs(p)
        torch.save(model_trained.state_dict(), os.path.join(p, 'model_trained.pt'))

    def save_importances(self, importances):
        p = os.path.join(self.path, self.run_name)
        if not os.path.exists(p):
            os.makedirs(p)
        torch.save(importances, os.path.join(p, 'bow_importances.pt'))


def compute_logits_for_single_token(tokenizer, model, n_max=10, use_cls=True, input_str="perfect", neutral_input="the", use_device="cuda"):
    logit_results = []
    cls_tok = torch.tensor([101])
    input_tok = torch.tensor(tokenizer.convert_tokens_to_ids(input_str)).reshape(1)
    neutal_tok = torch.tensor(tokenizer.convert_tokens_to_ids(neutral_input)).reshape(1)
    sep_tok = torch.tensor([102])
    ## Evaluate neutral input
    if use_cls:
        input = torch.cat((cls_tok, neutal_tok, sep_tok))
    else:
        input = neutal_tok
        
    with torch.no_grad():
        logit_score = model(input.reshape(1,-1).to(use_device))["logits"]
        logit_results.append(logit_score.detach().cpu())
        
        x = input_tok
        for i in range(n_max):
            #print(x)
            if use_cls:
                input = torch.cat((cls_tok, x, sep_tok))
            else:
                input = x
            logit_score = model(input.reshape(1,-1).to(use_device))["logits"]
            logit_results.append(logit_score.detach().cpu())
            x = torch.cat((x, input_tok), dim=0)
    return torch.stack(logit_results)