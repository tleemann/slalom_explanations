import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions import Categorical
from torch.nn import Parameter
from experiment_utils.attribution_methods import BoW

class BoWDataset(Dataset):
    """ A dataset that follows a BoW distribution. """

    def __init__(self, base_ds, tokenizer, sample_length = 100, ds_length=5000, alpha=20, start_tok = "[CLS]", end_tok="[SEP]"):
        """ Init the simulated dataset with a length len. (determines the size of an epoch,
            however each sample is randomly drawn new, i.e., the getitem function is non-deterministic.)
        
        """
        bow = BoW(ds=base_ds, tokenizer=tokenizer)

        counts = bow.convert_ds_to_count_vectors(bow.X_train, norm_len=False)
        counts = counts.toarray()
        poscnt = counts[np.array(bow.y_train) == 1].sum(axis=0) + alpha # total occurance of word in positive class
        negcnt = counts[np.array(bow.y_train) == 0].sum(axis=0) + alpha # total occurance of word in positive class

        print(poscnt.shape, negcnt.shape)
        pos_occ = np.sum(counts[np.array(bow.y_train) == 1]) + len(poscnt)*alpha # total number of words in positive class
        neg_occ = np.sum(counts[np.array(bow.y_train) == 0]) + len(negcnt)*alpha # total number of words in negative class
        print(pos_occ, neg_occ)
        self.probs = torch.from_numpy(np.stack((negcnt/neg_occ, poscnt/pos_occ))).float()
        self.cats = [Categorical(self.probs[0]), Categorical(self.probs[1])]
        feature_names = np.array(bow.count_vec.get_feature_names_out())
        self.token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(feature_names), dtype=torch.long)
        self.length = ds_length
        self.sample_length = sample_length
        self.start_tok = torch.tensor(tokenizer.convert_tokens_to_ids([start_tok])) if start_tok is not None else torch.tensor([])
        self.end_tok = torch.tensor(tokenizer.convert_tokens_to_ids([end_tok])) if start_tok is not None else torch.tensor([])

    def get_bow_weights(self):
        # Return weights and token ids for weights.
        return torch.log(self.probs[1]/self.probs[0]), self.token_ids

    def __getitem__(self, index: int):
        label = (torch.rand([]) > 0.5).long()
        features = torch.cat((self.start_tok, self.token_ids[self.cats[label.item()].sample([self.sample_length])], self.end_tok))
        return {"label": label, "input_ids": features, "attention_mask": torch.ones_like(features)}
    
    def __len__(self):
        return self.length

# A dataset class that represent a markov chain.
class SparseBoWDataset(Dataset):
    """ A dataset that follows a sparse Bag-of-Words distribution. """

    def __init__(self, tokenizer, max_sample_length = 30, ds_length=5000, all_words_dict = {"best": (0.1, 1.0), "the": (0.8, 0.0), "worst": (0.1, -1.0)},
            start_tok = "[CLS]", end_tok="[SEP]", fixed_len=True, binomial_sampling=True):
        """ Init the simulated dataset with a length len. (determines the size of an epoch,
            however each sample is randomly drawn new, i.e., the getitem function is non-deterministic.)
            all_words_dict: dict with key: token_string, value: (p_occurance, word_logit_score)
        """
        self.length = ds_length
        self.sample_length = max_sample_length
        self.start_tok = torch.tensor(tokenizer.convert_tokens_to_ids([start_tok])) if start_tok is not None else torch.tensor([], dtype=torch.long)
        self.end_tok = torch.tensor(tokenizer.convert_tokens_to_ids([end_tok])) if end_tok is not None else torch.tensor([], dtype=torch.long)
        self.word_toks = torch.tensor(tokenizer.convert_tokens_to_ids([k for k in all_words_dict.keys()]), dtype=torch.long)
        self.word_probs = torch.tensor([k[0] for k in all_words_dict.values()])
        self.word_scores = torch.tensor([k[1] for k in all_words_dict.values()])
        self.cats = Categorical(self.word_probs)
        self.fixed_len = fixed_len
        self.binomial = binomial_sampling

    def __getitem__(self, index: int):
        if self.fixed_len:
            sample_len = self.sample_length
        else:
            if self.binomial:
                sample_len = torch.sum(torch.rand(self.sample_length)> 0.5).long()
            else:
                sample_len = torch.randint(self.sample_length, tuple())+1
        feature_ids = self.cats.sample([sample_len])
        logit = torch.sum(self.word_scores[feature_ids])
        odds = torch.exp(logit)
        p1 = odds/(1 + odds)
        label = (torch.rand([]) < p1).long()
        features = torch.cat((self.start_tok, self.word_toks[feature_ids], self.end_tok))
        features_pad = torch.zeros(self.sample_length+(len(features)-sample_len), dtype=torch.long)
        attn_mask = torch.zeros_like(features_pad)
        features_pad[:len(features)] = features
        attn_mask[:len(features)] = 1
        #print(features_pad.shape)
        return {"label": label, "input_ids": features_pad, "attention_mask": attn_mask, "gt_score": logit}
    
    def __len__(self):
        return self.length

class SLALOMDataset(Dataset):
    def __init__(self, tokenizer, sample_max_length = 30, ds_length=5000,
            all_words_dict = {"best": (0.1, 2.0, 1.0), "the": (0.8, -1.0, 0.0), "worst": (0.1, 4.0, -1.0)},
            start_tok = None, end_tok=None, pad_token="[PAD]", seed=False):
        """ Init the simulated dataset with a length len. (determines the size of an epoch,
            however each sample is randomly drawn new, i.e., the getitem function is non-deterministic.)
            all_words_dict: dict with key: token_string, value: (p_occurance, word_importance_score, word_value_score)
        """
        self.length = ds_length
        self.sample_length = sample_max_length
        self.pad_token = torch.tensor(tokenizer.convert_tokens_to_ids([pad_token]))
        self.start_tok = torch.tensor(tokenizer.convert_tokens_to_ids([start_tok])) if start_tok is not None else torch.tensor([], dtype=torch.long)
        self.end_tok = torch.tensor(tokenizer.convert_tokens_to_ids([end_tok])) if end_tok is not None else torch.tensor([], dtype=torch.long)
        self.word_toks = torch.tensor(tokenizer.convert_tokens_to_ids([k for k in all_words_dict.keys()]), dtype=torch.long)
        self.word_probs = torch.tensor([k[0] for k in all_words_dict.values()])
        self.word_scores = torch.tensor([k[2] for k in all_words_dict.values()])
        self.word_importances = torch.tensor([k[1] for k in all_words_dict.values()])
        self.cats = Categorical(self.word_probs)
        if seed:
            torch.manual_seed(5)

    def __getitem__(self, index: int):
        feature_ids = self.cats.sample([self.sample_length])
        sample_len = torch.randint(self.sample_length, tuple())+1 # set length of the sample
        attn = torch.softmax(self.word_importances[feature_ids[:sample_len]], dim=-1)
        logit = torch.sum(self.word_scores[feature_ids[:sample_len]]*attn)
        odds = torch.exp(logit)
        p1 = odds/(1 + odds)
        label = (torch.rand([]) < p1).long()
        features = torch.cat((self.start_tok, self.word_toks[feature_ids[:sample_len]], self.end_tok)).flatten()
        sample_size = self.sample_length + (len(features) - sample_len)
        #print(sample_size, features.shape, sample_len)
        features_padded = self.pad_token.item()*torch.ones(sample_size, dtype=torch.long)
        features_padded[:len(features)] = features
        att_mask = torch.ones(sample_size, dtype=torch.long)
        att_mask[len(features):]  = 0
        return {"label": label, "input_ids": features_padded, "attention_mask": att_mask, "score": logit, "true_len": sample_len}
    
    def __len__(self):
        return self.length

