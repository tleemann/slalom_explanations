## Implement two non-transformer reference models for classification and implement functions to load them.
## We implement these models to show that SLALOM can also be applied to non-transformer models.
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from experiment_utils.attribution_methods import BoW

class LogitReturnType():
    def __init__(self, logits):
        self.logits = logits
    
    def __getitem__(self, string):
        if string == "logits":
            return self.logits
        else:
            raise ValueError(f"Invalid attribute {string} for LogitReturnType.")
        
class TwoLayerTFIDF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2 = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim1))
        if hidden_dim2 is None:
            self.layers.append(nn.Linear(hidden_dim1, 2))
        else:
            self.layers.append(nn.Linear(hidden_dim1, hidden_dim2))
            self.layers.append(nn.Linear(hidden_dim2, 2))

    def forward(self, x):
        for i, lay in enumerate(self.layers):
            x = lay(x)
            if i < len(self.layers) - 1: ## all but last
                x = torch.relu(x)
        return x

class TFIDFModel(nn.Module):
    def __init__(self, dataset, tokenizer, weights_file=None, hidden_dim=768, hidden_dim2=150):
        super().__init__()
        self.tokenizer = tokenizer
        self.imdbbow = BoW(dataset, self.tokenizer)
        voc_size = len(self.imdbbow.tfidf_vec.vocabulary_)
        self.tfidf = TwoLayerTFIDF(voc_size, hidden_dim, hidden_dim2)
        if weights_file is not None:
            self.tfidf.load_state_dict(torch.load(weights_file))
        self.device = "cpu"

    def to(self, device):
        self.device = device
        super().to(device)
        return self 

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor = None):
        inp_texts = []
        if attention_mask is not None:
            for k in range(len(input_ids)):
                valid_ids = input_ids[k][attention_mask[k]==1]
                inp_texts.append(self.tokenizer.decode(valid_ids).replace("##", ""))
        else: 
            for k in range(len(input_ids)):
                inp_texts.append(self.tokenizer.decode(input_ids[k]).replace("##", ""))
        #print(inp_texts[0])
        x_in = self.imdbbow.tfidf_vec.transform(inp_texts)
        #print(x_in.toarray()[0])
        return LogitReturnType(self.tfidf(torch.tensor(x_in.toarray(), dtype=torch.float).to(self.device)))

class LSTMEmbeddings(nn.Module):
    def __init__(self, tokenizer, weights_file, hidden_dim=150, batch_size=32):
        super().__init__()
        self.mysmallmodel = TwoLayerTFIDF(input_dim = 2048, hidden_dim1 = hidden_dim, hidden_dim2 = None)
        self.offset, state_dict = torch.load(weights_file)
        self.mysmallmodel.load_state_dict(state_dict)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.embedder = SentenceTransformer('hli/lstm-qqp-sentence-transformer')
        self.device="cpu"

    def to(self, device):
        self.device = device
        super().to(device)
        return self 

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor = None):
        inp_texts = []
        if attention_mask is not None:
            for k in range(len(input_ids)):
                valid_ids = input_ids[k][attention_mask[k]==1]
                inp_texts.append(self.tokenizer.decode(valid_ids).replace("##", ""))
        else: 
            for k in range(len(input_ids)):
                inp_texts.append(self.tokenizer.decode(input_ids[k]).replace("##", ""))

        ret_list = []
        for j in range(0, len(input_ids), self.batch_size):
            embeddings = self.embedder.encode(inp_texts[j:j+self.batch_size])
            in_tensor = torch.from_numpy(embeddings)
            #print(in_tensor-self.offset)
            ret_tensor = self.mysmallmodel.forward((in_tensor-self.offset).to(self.device))
            ret_list.append(ret_tensor)

        return LogitReturnType(torch.cat(ret_list, dim=0))
