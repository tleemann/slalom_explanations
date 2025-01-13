""" Compute SLALOM explanations for OPENAI models. """

### GPT-API model
from openai import OpenAI
import os
import json
import torch
from multiprocessing import Pool
import numpy as np

API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_LEGACY_ENDPOINT = "https://api.openai.com/v1/completions"


DEFAULT_PROMPT = "You are assessing movie reviews in an online forum. Your goal is to assess the reviews overall sentiment as 'overall negative' (label '0') or 'overall prositive' (label '1'). Your will see a review now and you will output a label. Make sure to only answer with either '0' or '1'."


class OpenAImodel:
    """ Use the OpenAI API to perform classifications. """

    def __init__(self, target_model="gpt-4-turbo", prompt_template=DEFAULT_PROMPT, openai_keys_file="openai.json", n_threads=6, device="cpu", tokenizer=None):
        """
            target_model: the API to use
            prompt template: the prompt template to use. will be used with (template % sequence) to generate prompt. Should contain the command to just output either "0" or "1".
            openai_keys: tuple of (openai_org, openai_apikey)
        """
        self.n_threads = n_threads
        self.model = target_model
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer
        self.device = device
        if openai_keys_file is not None:
            credentials = json.load(open(openai_keys_file))
            self.openai_org, self.openai_key = credentials["org"], credentials["key"]
            self.client = OpenAI(
                organization=self.openai_org,
                api_key =self.openai_key
            )
        else:
            elf.openai_org, self.openai_key = None, None

    def train(self, ds):
        pass # No training requred

    def predict_proba(self, input_sequences):
        #print(len(input_sequences))
        #with p as Pool(self.n_threads):
        #f_map = lambda seq: self._predict_proba_single(input_sequences, seq)
        tasklist =  list([(self.openai_org, self.openai_key, self.model, self.prompt_template, i) for i in input_sequences])
        with Pool(self.n_threads) as p:
            res_array = p.map(predict_logits_single, tasklist)
        return np.stack(res_array, axis=0)

    def forward(self, input_ids, attention_mask=None):
        """ Imitate the transformers interface for usage with SLALOM and other attribution methods. """
        
        ## First, convert tokens back to text.
        input_list_text = []
        for line in range(len(input_ids)):
            seq = input_ids[line]
            if attention_mask is not None:
                seq = seq[attention_mask[line]==1]
            text = self.tokenizer.decode(seq)
            input_list_text.append(text)
        #print(input_list_text)
        ## 
        res = self.predict_proba(input_list_text)
        return {"logits": torch.tensor(res, dtype=torch.float).to(self.device)}

    def __call__(self, input_ids, attention_mask=None):
        return self.forward(input_ids, attention_mask)

    def to(self, device):
        """ For compatibility purposes. """
        self.device = device
        return self

    def eval(self):
        return self
        
    def predict(self, input_sequences):
        if self.client is None:
            raise ValueError("Client is not initialized. Please pass an API key to the constructor or load a model file.")
        message_dict = [{
            "role": "system",
            "content": self.prompt_template
        },
        {
            "role": "user",
            "content": None
        }]
        res_array = []
        for seq in input_sequences:
            message_dict[1]["content"] = seq #  self.prompt_template.format(seq)
            results = self.client.chat.completions.create(
                model=self.model,
                messages=message_dict,
                logprobs=True,
                temperature=0.000000001
            )
            print(results.choices[0])
            res_array.append((results.choices[0].message.content, results.choices[0].logprobs))
        return res_array

        
    def save(self, filename):
        torch.save({"model": self.model, "prompt": self.prompt_template, "api_key": self.openai_key, "api_org": self.openai_org}, filename)

    def load(self, filename):
        res = torch.load(filename)
        self.model=res["model"]
        self.prompt_template =res["prompt"]
        self.openai_key = res["api_key"]
        self.openai_org = res["api_org"]
        self.client = OpenAI(
                organization=self.openai_org,
                api_key = self.openai_key
            )

    
""" Subthread. """

def compute_logits_from_logprobs(choice_logprobs, logits=True):
    """ Compute the probabilities of chosing the labels 0, 1. """
    p1 = -float("inf")
    p0 = -float("inf")
    for token in choice_logprobs:
        if token.token == "1":
            p1 = token.logprob
        if token.token == "0":
            p0 = token.logprob
    #print("0", p0, "1", p1)
    C=20
    odds_diff = p1-p0
    if np.isnan(odds_diff):
        ret_prob = 0.5
        ret_logit = 0.0
    elif odds_diff == float("inf"):
        ret_prob = 1.0
        ret_logit = C
    elif odds_diff == -float("inf"):
        ret_prob = 0.0
        ret_logit = -C
    else:
        ret_prob = np.exp(odds_diff)/(np.exp(odds_diff)+1)
        ret_logit = odds_diff
    if logits:
        return np.array([-0.5*ret_logit, 0.5*ret_logit])
    else: # Return probabilities.
        return np.array([1-p1, p1])


def predict_logits_single(arglist):
    openai_org, openai_key, target_model, prompt_template, predict_seq = arglist
    client = OpenAI(
                organization=openai_org,
                api_key =openai_key
            )
    if client is None:
        raise ValueError("Client is not initialized. Please pass an API key to the constructor or load a model file.")

    message_dict = [{
        "role": "system",
        "content": prompt_template
    },
    {
        "role": "user",
        "content": None
    }]

    message_dict[1]["content"] = predict_seq #self.prompt_template.format(seq)
    #print(message_dict)
    results = client.chat.completions.create(
        model=target_model,
        messages=message_dict,
        logprobs=True,
        top_logprobs = 5,
        temperature=0.00001,
        seed=1
    )
    return compute_logits_from_logprobs(results.choices[0].logprobs.content[0].top_logprobs)
    #print(results.choices[0])