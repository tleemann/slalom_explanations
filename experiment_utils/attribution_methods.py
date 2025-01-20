from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression

from transformers import AutoTokenizer, BertTokenizerFast, BertForSequenceClassification
import datasets
import numpy as np
import torch
from typing import List, Tuple, Dict, Union
import regex
import shap, lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F

from experiment_utils.transformer_models import Bert, DistilBert, GPT2

#from line_profiler import LineProfiler


class BoW():
    #@profile
    def __init__(self, ds: datasets.dataset_dict.DatasetDict, tokenizer: AutoTokenizer, min_df: int=1, token_pattern: str=r'[\S]+' , max_seq_len: int=512) -> None:  # token_pattern: str=r'[a-zA-Z]+'  '[^\s]+' (does not preserve upper lower case?)
        self.ds = ds
        self.count_vec = CountVectorizer(tokenizer=self.__custom_tokenizer, token_pattern=None)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        print(f'self.tokenizer: {self.tokenizer} type(self.tokenizer: {type(self.tokenizer)}')
        self.tfidf_vec = TfidfVectorizer(min_df=min_df, tokenizer=self.__custom_tokenizer, token_pattern=None)
        self.X_train_bow, self.X_train, self.y_train, self.X_test_bow, self.X_test, self.y_test = self.__convert_ds_to_tfidf(self.ds)


    #@profile
    def __custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer for TfidfVectorizer used for AutoTokenizer from HuggingFace"""
        if self.tokenizer is None:
            tokens = text.split()
            return tokens
        else:
            tok_list = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            special_list =  self.tokenizer.all_special_tokens
            return list(filter(lambda a: a not in special_list, tok_list))

        # return regex.findall(r'\b\w+\b|.', text)
    
    def __ds_to_lists(self, ds: datasets.arrow_dataset.Dataset) -> Tuple[List[str], List[int]]:
        """Convert a HuggingFace dataset to a list of texts and labels"""
        texts = []
        labels = []

        for i in range(len(ds)):
            labels.append(int(ds[i]['label']))
            texts.append(ds[i]['text'])
        labels = [int(i) for i in labels]
        return texts, labels

    #@profile
    def __convert_ds_to_tfidf(self, ds: datasets.dataset_dict.DatasetDict) -> Tuple[np.ndarray, List[str], List[int], np.ndarray, List[str], List[int]]:
        """Convert a HuggingFace dataset to a BoW representation using TfidfVectorizer
        
        Returns:
            X_train_bow: BoW representation of the training set
            X_train: List of texts of the training set
            y_train: List of labels of the training set
            X_test_bow: BoW representation of the test set
            X_test: List of texts of the test set
            y_test: List of labels of the test set
        """
        X_train, y_train = self.__ds_to_lists(self.ds['train'])
        X_test, y_test = self.__ds_to_lists(self.ds['test'])
        X_train_bow = self.tfidf_vec.fit_transform(X_train)
        X_test_bow = self.tfidf_vec.transform(X_test)
        return X_train_bow, X_train, y_train, X_test_bow, X_test, y_test

    #@profile
    def convert_ds_to_count(self, X_train, y_train) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = self.count_vec.fit_transform(X_train)
        #counts = self.count_vec.fit_transform(X_train)
        counts = (counts > 0).astype(int)
        occ_counts_pos = counts[y_train == 1].sum(axis=0)
        occ_counts_neg = counts[y_train == 0].sum(axis=0)
        return y_train.float().mean(), occ_counts_pos, occ_counts_neg
    
    def convert_ds_to_count_vectors(self, X_train, norm_len=True):
        """ Compute count normalized by lenght of documents. Then fit logistic model. """
        #Xtok = self.__apply_autotokenizer(X_train)
        counts = self.count_vec.fit_transform(X_train)
        #print(counts.shape)
        if norm_len:
            total_doc_counts=counts.sum(axis=1) # total number of words in documents
            counts /= total_doc_counts
        return counts


class SVMEstim(object):
    def __init__(self, bow: BoW, kernel: str='linear', C: float=1.0) -> None:
        self.X_train = bow.X_train_bow
        self.y_train = bow.y_train
        self.svm = svm.SVC(C=C, kernel=kernel, probability=True)
        self.tfidf_vec = bow.tfidf_vec

    def get_importance(self) -> Dict[str, float]:
        self.svm.fit(self.X_train, self.y_train)
        importances = np.abs(self.svm.coef_.toarray().flatten())
        top_indices_pos = np.argsort(importances)[::-1]
        feature_names = np.array(self.tfidf_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], importances[top_indices_pos]))


class LogRegEstim(object):
    def __init__(self, bow: BoW) -> None:
        self.X_train = bow.X_train_bow
        self.y_train = bow.y_train
        self.logreg = LogisticRegression()
        self.tfidf_vec = bow.tfidf_vec

    def get_importance(self) -> Dict[str, float]:
        self.logreg.fit(self.X_train, self.y_train)
        importances = np.abs(self.logreg.coef_.flatten())
        top_indices_pos = np.argsort(importances)[::-1]
        feature_names = np.array(self.tfidf_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], importances[top_indices_pos]))


class NormCountsLogRegEstim(object):
    """ Fit logistic regression model on word counts. """
    def __init__(self, bow: BoW, norm_len=True) -> None:
        self.X_train, self.y_train = bow.X_train, bow.y_train
        self.logreg = LogisticRegression(penalty=None, max_iter=10000, tol=1e-6)
        self.bow=bow
        self.cnt_vec = self.bow.convert_ds_to_count_vectors(self.X_train, norm_len=norm_len)
        print(self.cnt_vec.shape)

    def get_signed_importance(self) -> Dict[str, float]:
        self.logreg.fit(self.cnt_vec, self.y_train)
        importances = self.logreg.coef_.flatten()
        top_indices_pos = np.argsort(importances)[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], importances[top_indices_pos]))

    def get_importance(self) -> Dict[str, float]:
        self.logreg.fit(self.cnt_vec, self.y_train)
        importances = np.abs(self.logreg.coef_.flatten())
        top_indices_pos = np.argsort(importances)[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], importances[top_indices_pos]))


class NaiveBayesEstim():
    def __init__(self, bow: BoW, alpha: float=20.0, multiplicities=False, length_weighting=False):
        self.bow = bow
        self.X_train, self.y_train = bow.X_train, torch.tensor(bow.y_train)
        self.classes_ = [0,1]
        self.alpha = alpha
        self.coef_ = None
        self.multiplicities = multiplicities
        self.length_weighting = length_weighting
        self._estimator_type = "classifier" # pretend to be sklearn classifier
        
    def fit(self):
        counts = self.bow.convert_ds_to_count_vectors(self.X_train, norm_len=False)
        print(counts.shape)
        if self.multiplicities:
            counts = counts.toarray()
            poscnt = counts[self.y_train == 1].sum(axis=0) # total occurance of word in positive class
            negcnt = counts[self.y_train == 0].sum(axis=0) # total occurance of word in positive class

            if self.length_weighting:
                doc_lens = counts.sum(axis=1).reshape(-1,1)
                non_occurances = (doc_lens-counts)/doc_lens # Share of non-i tokens in the word
                pos_nocc = np.sum(non_occurances[self.y_train == 1], axis=0) # total number of words in positive class
                neg_nocc = np.sum(non_occurances[self.y_train == 0], axis=0) # total number of words in negative class
                # Compute non-occurance shares
                print(pos_nocc[0], pos_nocc.shape)
                self.coef_ = np.array(np.log(((poscnt + self.alpha)*(neg_nocc))/ \
                        ((negcnt + self.alpha)*(pos_nocc)))).flatten()
            else:
                self.coef_ = np.array(np.log((poscnt + self.alpha)/(negcnt + self.alpha))).flatten()
        else:
            counts = (counts > 0).astype(int)
            poscnt = counts[self.y_train == 1].sum(axis=0) # docs with word in postive class 
            negcnt = counts[self.y_train == 0].sum(axis=0) # docs with word in negative class
            pos_docs = torch.sum(self.y_train == 1).item()
            neg_docs = torch.sum(self.y_train == 0).item()
            self.coef_ = np.array(np.log((poscnt + self.alpha)/(negcnt + self.alpha))).flatten()
        #print(self.coef_)

    def sigmoid(self, X):
        return 1.0/(np.exp(-X)+1)
    
    def predict_proba(self, text_test):
        X = self.count_vec.transform(text_test)
        X = (X > 0).astype(int)
        #print(X.shape, weights.shape)
        res = X @ weights.reshape(-1,1)
        return self.sigmoid(res.reshape(-1,1)*np.array([-1,1]))
    
    def predict(self, text_test):
        return np.argmax(self.predict_proba(text_test), axis=1)
        
    def get_importance(self) -> Dict[str, float]:
        self.fit()
        top_indices_pos = np.argsort(np.abs(self.coef_))[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], (np.abs(self.coef_[top_indices_pos]))))
    
    def get_signed_importance(self) -> Dict[str, float]:
        self.fit()
        top_indices_pos = np.argsort(self.coef_)[::-1]
        feature_names = np.array(self.bow.count_vec.get_feature_names_out())
        return dict(zip(feature_names[top_indices_pos], (self.coef_[top_indices_pos])))
    


class ZeroExplanation():
    """ A dummy class that only returns zeros as explanations. Can be used as a baseline in removal benchmarks. """
    def __init__(self, model=None):
        pass

    def get_signed_importance_for_tokens(self, input_ids):
        return np.zeros(len(input_ids))
    
class ShapleyValues():
    """ A class to compute ground truth Shapley values. """
    def __init__(self, model, tokenizer, num_samples: Union[int, str] = "auto", method="partition", impute_token="[UNK]", device="cuda"):
        """
            We implement three sampling strategies for Shapley values,
            Partition (method="partition") Permutation (method="permutation") and Kernel (method="kernel")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.impute_token = impute_token
        self.method = method
        filler = tokenizer.convert_tokens_to_ids([impute_token])
        bg = filler*min(tokenizer.model_max_length, 1024)
        self.bg = np.array([bg]) # background data used for kernel shap
        self.device = device

    def get_signed_importance(self, input_sample: str):
        """ Compute shapley values for an input sample. """
        if self.method=="kernel":
            my_shap_vals = self._get_kernel_shap_values(input_sample)
        else:
            my_shap_vals = self._get_permutation_or_partition_shap_values(input_sample)
        
        top_indices_pos = np.argsort(my_shap_vals)[::-1]
        feature_names = np.array(self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(input_sample)))
        return dict(zip(feature_names[top_indices_pos], my_shap_vals[top_indices_pos]))

    def get_signed_importance_for_tokens(self, input_ids):
        """ Compute the importances for a list of input tokens """
        if self.method == "kernel":
            return self._get_kernel_shap_values(input_ids)
        else:
            return self._get_permutation_or_partition_shap_values(input_ids)

    def _get_kernel_shap_values(self, input_sample: Union[str, list]):
        if type(input_sample) == str:
            inputs = self.tokenizer(input_sample, padding=True)["input_ids"]
        else:
            inputs = input_sample

        def kernel_predictor(x):
            #print(x)
            outputs = self.model(x.to(self.device))["logits"]
            probas = outputs.detach().cpu().numpy()
            val = probas[:,1]-probas[:,0]
            return val

        def kernel_batch(x):
            val = np.array([])
            for i in x:
                val = np.append(val, kernel_predictor(torch.from_numpy(i).long().reshape(1,-1)))
            return val

        explainer_bert_kernel = shap.KernelExplainer(kernel_batch, data=self.bg[:,:len(inputs)])
        shap_values = explainer_bert_kernel.shap_values(np.array(inputs), nsamples=self.num_samples)
        return shap_values

    def _get_permutation_or_partition_shap_values(self, input_sample: Union[str, list]):
        def predictor(x):
            outputs = self.model(self.tokenizer(x, return_tensors="pt", padding=True)["input_ids"].to(self.model.device))["logits"]
            probas = outputs.detach().cpu().numpy()
            val = probas[:,1]-probas[:,0]
            return val

        def f_batch(x):
            val = np.array([])
            for i in x:
                val = np.append(val, predictor(i))
            return val

        def predictor_tokens(x):
            outputs = self.model(x.to(self.device))["logits"]
            probas = outputs.detach().cpu().numpy()
            val = probas[:,1]-probas[:,0]
            return val

        def f_batch_tokens(x):
            val = np.array([])
            for i in x:
                val = np.append(val, predictor_tokens(torch.from_numpy(i).long().reshape(1,-1)))
            return val

        def constant_masker(mask, x):
            # in this simple example we just zero out the features we are masking
            #print("cmask:", x, mask)
            return ((x * mask) + (1-mask)*self.bg[0,0]).reshape(1, len(x))

        if type(input_sample) == str:
            my_masker = shap.maskers.Text(tokenizer=self.tokenizer, mask_token=self.impute_token)
            test = {'text': [input_sample]}
        else:
            my_masker = constant_masker #shap.maskers.Partition(data=self.bg[:,:len(input_sample)])
            test = np.array([input_sample])

        if self.method=="permutation":
            if type(input_sample) == str:
                explainer_bert = shap.PermutationExplainer(f_batch, masker=my_masker)
            else:
                explainer_bert = shap.PermutationExplainer(f_batch_tokens, masker=my_masker)

        elif self.method=="partition":
            if type(input_sample) == str:
                explainer_bert = shap.PartitionExplainer(f_batch, masker=my_masker)
            else:
                raise ValueError("Other input types than str unsuppeorted for partition.")
        else:
            raise ValueError(f"Unsupported shapley method {self.method}.")

        shap_values = explainer_bert(test, max_evals = self.num_samples)
        return shap_values.values[0]

class LIMEExplanation():
    def __init__(self, model, tokenizer, n_samples=200, device="cuda"):
        self.model = model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
    
        def predictor(x):
            with torch.no_grad():
                inps = tokenizer(x, return_tensors="pt", padding=True)
                p_list = []
                for i in range(0, len(inps["input_ids"]), 16):
                    outputs = self.model(inps["input_ids"][i:i+16].to(self.device), attention_mask = inps["attention_mask"][i:i+16].to(self.device))["logits"]
                    probas = F.softmax(outputs).detach().cpu().numpy()
                    p_list.append(probas)
            ret = np.concatenate(p_list, axis=0)
            return ret

        def custom_tokenizer(text: str):
            """Custom tokenizer for TfidfVectorizer used for AutoTokenizer from HuggingFace"""
            if self.tokenizer is None:
                tokens = text.split()
                return tokens
            else:
                tok_list = self.tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
                special_list =  self.tokenizer.all_special_tokens
                for i in range(len(tok_list)):
                    tok_list[i] = tok_list[i].replace("##", "").replace("Ċ", "\n")
                    if tok_list[i] == "Ġ.":
                        tok_list[i] = tok_list[i].replace("Ġ", "")
                    else:
                        tok_list[i] = tok_list[i].replace("Ġ", " ")
                return list(filter(lambda a: a not in special_list, tok_list))
        self.predictor = predictor

        self.explainer = LimeTextExplainer(class_names=['positive', 'negative'], 
                split_expression=custom_tokenizer)

    def _adapt_text(self, text):
        tok_list =  self.tokenizer.encode(text)
        special_list =  self.tokenizer.all_special_tokens
        tok_list = list(filter(lambda a: a not in special_list, tok_list))
        return self.tokenizer.decode(tok_list), len(tok_list)

    def get_signed_importance_for_tokens(self, input_ids: list):
        inp_text = self.tokenizer.decode(input_ids)
        individual_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        for i in range(len(individual_tokens)):
            individual_tokens[i] = individual_tokens[i].replace("##", "").replace("Ċ", "\n")
            if individual_tokens[i] == "Ġ.":
                individual_tokens[i] = individual_tokens[i].replace("Ġ", "")
            else:
                individual_tokens[i] = individual_tokens[i].replace("Ġ", " ")
        lime_res = self._compute_lime(inp_text, len(input_ids))
        ret = np.zeros(len(input_ids))
        for i in range(len(ret)):
            ret[i] = lime_res[individual_tokens[i]]
        return ret


    def get_lime_for_text(self, text: str) -> None:
        inp_text, len_list = self._adapt_text(text)
        return self._compute_lime(inp_text, len_list)

    def _compute_lime(self, clean_text, nf: int):
        explanation = self.explainer.explain_instance(clean_text, self.predictor, num_samples=self.n_samples, num_features=nf)
        return {k: v for (k, v) in explanation.as_list()}

class LinearRegressionDeletion:
    """ A linear regression model fitted on deletions for comparision. """
    def __init__(self, model, n_samples=200, device="cuda", sampling_strategy="deletion", pad_token_id=0, use_cls=True, batch_size_sample=32):
        self.model = model.to(device)
        self.n_samples = n_samples
        self.device = device
        self.mode = sampling_strategy
        self.pad_token_id = pad_token_id
        self.use_cls = use_cls
        self.batch_size_sample=32

    def get_signed_importance_for_tokens(self, input_ids: list):
        unique_list, return_inverse = torch.tensor(input_ids).unique(return_inverse=True)
        device = self.device
        inps_list, mask_list, bin_list, output_list = [], [], [], []
        for i in range(0, self.n_samples, self.batch_size_sample):
            _, bin_features, _, outputs = sample_dataset(self.batch_size_sample, self.model, unique_list, torch.tensor(input_ids), seq_len=3, use_cls=self.use_cls, fixed_len=True, 
                                                                device=self.device, mode=self.mode, pad_token_id=self.pad_token_id)
            output_list.append(outputs)
            bin_list.append(bin_features)

        binary_features = torch.cat(bin_list, dim=0)
        labels = torch.cat(output_list, dim=0)
        mylr = LinearRegression(fit_intercept=False)
        mylr.fit(binary_features.detach().cpu(), labels.cpu())
        return mylr.coef_[return_inverse.numpy()]

class LRPExplanation():
    """ LRP for transformers. """
    def __init__(self, modellrp, device="cuda", use_cls=True, normalize=True, mode="logit"):
        #transform the model 
        print(device)
        self.mymodel = modellrp.to(device)
        self.mymodel = self.mymodel.eval()
        self.device = device
        self.use_cls = use_cls
        self.normalize = normalize
        self.mode = mode

    def get_signed_importance_for_tokens(self, input_ids: list, vs_scores=False):
        if self.use_cls:
            org_inp =  torch.tensor([101] + input_ids + [102]).reshape(1,-1)
        else:
            org_inp = torch.tensor(input_ids).reshape(1,-1)
        #print(org_inp)
        # Compute reference logit output
        with torch.no_grad():
            res = self.mymodel(org_inp.to(self.device), attention_mask = torch.ones_like(org_inp).to(self.device))["logits"]
        res_cls = res[:,1]-res[:,0]
        
        if hasattr(self.mymodel, "bert"):
            input_embeddings = self.mymodel.bert.embeddings.word_embeddings(org_inp.to(self.device)).detach().clone()
        elif hasattr(self.mymodel, "distilbert"):
            input_embeddings = self.mymodel.distilbert.embeddings.word_embeddings(org_inp.to(self.device)).detach().clone()
        elif hasattr(self.mymodel, "transformer"):
            input_embeddings = self.mymodel.transformer.wte(org_inp.to(self.device)).detach().clone()
        else:
            raise ValueError("Unknown model type.")
        input_embeddings.requires_grad = True
        
        res_logits = self.mymodel(inputs_embeds=input_embeddings)["logits"] # [B, 2] array with class logits
        print(res_logits[:,1]-res_logits[:,0], res_cls)

        if self.mode =="logit":
            cls_score = res_logits[:,1]-res_logits[:,0]
        else:
            cls_score = res_logits[0, :].max()
        
        cls_score.backward(cls_score)
        relevance = input_embeddings.grad.float().sum(-1).cpu()
        #print("Obtained relevance scores: ", relevance)
        #exit(0)
        if self.mode =="classlogit" and res_cls.item() < 0:
            relevance = -relevance
        relevance = relevance[0]
        if self.normalize:
            relevance = relevance.cpu()*(torch.sum(res_cls.cpu())/torch.sum(relevance))
        else:
            relevance = relevance.cpu()
        if self.use_cls:
            relevance = relevance[1:-1]
        return relevance.numpy()


class InputGradEstim():
    def __init__(self, model: Union[Bert, DistilBert, GPT2], ds: datasets.dataset_dict.DatasetDict, tokenizer: AutoTokenizer, max_seq_len: int=512,
            agg_norm=True, use_ig=False, ig_steps=20, times_input=False, device="cuda") -> None:
        """ Compute input gradient attributions.
            use_ig: Use IG attribution instead of normal gradients, implementation inspired by 
                https://medium.com/@madhubabu.adiki/integrated-gradients-for-natural-language-processing-from-scratch-c81c50c5bc4d
            agg_norm: If true, gradients will be aggregated by the norm, if false, gradients will be aggregated by the sum.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.ds = ds
        self.device = device
        self.max_seq_len = max_seq_len
        self.token_count = {}
        self.token_importance = {}
        self.agg_norm = agg_norm
        self.use_ig = use_ig
        self.ig_steps = ig_steps
        self.times_input = times_input

    def get_correct_span_mask(self, correct_index, token_size):
        span_mask = F.one_hot(correct_index, num_classes=token_size)
        return span_mask

    def get_signed_importance_for_tokens(self, input_ids: list):
        return self._compute_gradients_tokens(torch.tensor(input_ids))[0].numpy()

    def compute_input_gradients(self, text: str) -> None:
        encoded_tokens = self.tokenizer(text, return_tensors="pt")
        token_ids = torch.tensor(encoded_tokens["input_ids"])[0]
        return self._compute_gradients_tokens(token_ids)

    def _compute_gradients_tokens(self, token_ids: torch.tensor):
        embedding_matrix = self.model.get_input_embeddings().weight
        token_ids = token_ids[:self.max_seq_len]
        #print(token_ids.shape, embedding_matrix.shape)
        vocab_size = embedding_matrix.size(0)
        attention_mask = torch.ones_like(token_ids)
        # Convert token ids to one hot.
        #token_ids_tensor = torch.tensor([token_ids], dtype=torch.int64)
        #token_ids_tensor_one_hot = F.one_hot(token_ids_tensor, num_classes=vocab_size).float()
        inputs_embeds = embedding_matrix[token_ids].unsqueeze(0) #torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
        
        # IG model with 0 baseline
        if self.use_ig:
            ig_mult = torch.linspace(0.0, 1.0, self.ig_steps, device=self.device, dtype=embedding_matrix.dtype).reshape(-1, 1, 1)
            inputs_embeds = ig_mult*inputs_embeds
            attention_mask = attention_mask.repeat(self.ig_steps, 1)
        else:
            attention_mask = attention_mask.unsqueeze(0)
        print(inputs_embeds.shape, attention_mask.shape)
        inputs_embeds.requires_grad_(True)
        pred_scores = self.model(inputs_embeds=inputs_embeds.to(self.device), attention_mask=attention_mask.to(self.device)).logits
        max_class = torch.argmax(pred_scores, dim=1)
        
        # get mask for predicted score class
        score_mask = self.get_correct_span_mask(max_class, pred_scores.shape[1])
        # zero out all predictions outside of the correct prediction class
        predict_correct_class = (pred_scores * score_mask).sum()

        grads = torch.autograd.grad(predict_correct_class, inputs_embeds)[0].detach()

        if self.use_ig:
            grads = torch.sum(grads*inputs_embeds.detach(), dim=0, keepdim=True)
        else:
            if self.times_input:
                grads = grads*inputs_embeds.detach()

        if self.agg_norm:
            gradient_non_normalized = torch.norm(grads, dim=2)
        else:
            gradient_non_normalized = torch.sum(grads, dim=2)

        decoded_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # delete all special tokens
        gradient_non_normalized, decoded_tokens = self.delete_special_tokens(gradient_non_normalized.cpu(), decoded_tokens)
        
        # delete last token if GPT model because attention is aggregated there
        if isinstance(self.model, GPT2) or isinstance(self.model, GPT2):
            gradient_non_normalized = gradient_non_normalized[:, :-1]
            decoded_tokens = decoded_tokens[:-1]

        # normalize
        gradient_tensor = gradient_non_normalized # / gradient_non_normalized.max()
        gradients = gradient_tensor[0]
        return gradients, decoded_tokens

    def delete_special_tokens(self, gradients: torch.Tensor, decoded_tokens: List) -> Tuple[torch.Tensor, List]:
        special_tokens = self.tokenizer.all_special_tokens
        special_token_indices = [i for i, x in enumerate(decoded_tokens) if x in special_tokens]
        gradients = np.delete(gradients, special_token_indices)
        gradients = gradients.reshape(1, -1) # add dimension back
        decoded_tokens = [i for j, i in enumerate(decoded_tokens) if j not in special_token_indices]
        return gradients, decoded_tokens

    def update_count_and_importance(self, gradients: torch.Tensor, decoded_tokens: List) -> None:
        for token in decoded_tokens:
            if token in self.token_count:
                self.token_count[token] += 1
            else:
                self.token_count[token] = 1

        for i, token in enumerate(decoded_tokens):
            if token in self.token_importance:
                self.token_importance[token] += gradients[i]
            else:
                self.token_importance[token] = gradients[i]

    def get_importance_input_grad(self) -> Dict[str, float]:
        for i in range(len(self.ds)):
            text = self.ds['train'][i]['text']
            gradients, decoded_tokens = self.compute_input_gradients(text)
            self.update_count_and_importance(gradients, decoded_tokens)

        # normalize importance
        for token in self.token_importance:
            self.token_importance[token] /= self.token_count[token]

        # sort by importance
        sorted_importance = sorted(self.token_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_importance = {k: v for k, v in sorted_importance}
        return sorted_importance

def get_groundtruth_importance(gt_str, bow):
    if gt_str == "nb":
        return NaiveBayesEstim(bow) 
    elif gt_str == "lr":
        return LogRegEstim(bow)
    elif gt_str == "svm":
        return SVMEstim(bow)
    else:
        raise NotImplementedError(f"Non-supported ground truth {gt_str}.")
