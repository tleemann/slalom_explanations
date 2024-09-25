from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizerFast, \
         GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config, BertConfig, BertForSequenceClassification
import torch
import numpy as np
import datasets
from typing import Union, List
from tqdm import tqdm

from slalom_explanations.utils import Logger, LoggerAttMat


class DistilBert():
    def __init__(self, dim: int=768, hidden_dim: int=3072, n_heads: int=12, n_layers: int=6) -> None:
        self.config = DistilBertConfig(dim=dim,
                                        hidden_dim=hidden_dim,
                                        n_heads=n_heads,
                                        n_layers=n_layers,
                                        output_attentions=True,
                                        output_hidden_states=True)
        self.model = DistilBertForSequenceClassification(self.config)

    def get_model(self) -> DistilBertForSequenceClassification:
        return self.model
    
    def set_model(self, model: DistilBertForSequenceClassification) -> DistilBertForSequenceClassification:
        self.model = model


class Bert():
    def __init__(self, dim: int=768, hidden_dim: int=3072, n_heads: int=12, n_layers: int=12, pretrained=True) -> None:
        self.config = BertConfig(hidden_size = dim,
                                num_hidden_layers = n_layers,
                                num_attention_heads = n_heads,
                                intermediate_size = hidden_dim,
                                output_attentions=True)
        if not pretrained:
            self.model = BertForSequenceClassification(self.config)
        else:
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
            self.model.bert.encoder.layer = self.model.bert.encoder.layer[:n_layers] #.word_embeddings.weight
            #self.model.bert.embeddings.position_embeddings.weight = ref_model.bert.embeddings.position_embeddings.weight

    def get_model(self) -> BertForSequenceClassification:
        return self.model


class GPT2():
    def __init__(self, n_embd: int=768, n_head: int=12, n_inner: int=None, n_layer: int=12) -> None:
        self.config = GPT2Config(n_embd=n_embd,
                                n_head=n_head,
                                n_inner=n_inner,
                                n_layer=n_layer,
                                output_attentions=True,
                                output_hidden_states=True)
        self.model = GPT2ForSequenceClassification(self.config)

    def get_model(self) -> GPT2ForSequenceClassification:
        return self.model

    def set_model(self, model: GPT2ForSequenceClassification) -> None:
        self.model = model


class DistilGPT2():
    def __init__(self, n_embd: int=768, n_head: int=12, n_inner: int=None, n_layer: int=6) -> None:
        self.config = GPT2Config(
            n_embd=n_embd,
            n_head=n_head,
            n_inner=n_inner,
            n_layer=n_layer,
            num_labels=2,
            output_attentions=True,
            output_hidden_states=True)
        #self.model = GPT2LMHeadModel(self.config)
        # fix model padding token id
        self.config.pad_token_id = 50256
        self.model = GPT2ForSequenceClassification(self.config)

    def get_model(self) -> GPT2ForSequenceClassification:
        return self.model

    def set_model(self, model: GPT2ForSequenceClassification) -> None:
        self.model = model


class Trainer():
    def __init__(self, ds: datasets.dataset_dict.DatasetDict,
                 model: Union[DistilBert, Bert, DistilGPT2, GPT2],
                 tokenizer: Union[DistilBertTokenizer, BertTokenizerFast, GPT2Tokenizer],
                 logger: Logger,
                 device: torch.device,
                 logger_att_mat: LoggerAttMat,
                 max_att_logs: int, 
                 max_seq_len: int,
                 tokenization_required: bool = True) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.get_model().to(self.device)
        self.logger = logger
        self.last_acc = 0.0
        self.max_seq_len = max_seq_len
        # tokenize dataset and create dataloader
        if tokenization_required:
            self.ds_tokenized = ds.map(self.preprocess_function_maxlen, batched=True)
        else:
            self.ds_tokenized = ds
        self.ds_tokenized.set_format(type="torch")
        self.logger_att_mat = logger_att_mat
        self.max_att_logs = max_att_logs
        if isinstance(self.model, GPT2ForSequenceClassification):
            print('GPT model detected, resizing token embeddings')
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def preprocess_function_maxlen(self, examples):
        """Preprocess function for the dataset, forcing the sequence length to be maxlen"""
        return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=self.max_seq_len, return_tensors='pt')

    def preprocess_function(self, examples):
        """Preprocess function for the dataset"""
        return self.tokenizer(examples['text'], padding=True, truncation=True, max_length=self.max_seq_len, return_tensors='pt')

    def train(self, epochs: int=2, test_interval: int=1, batch_size=1, lr=5e-5, logsteps=0, start_steps=0, start_epoch=0) -> None:
        """Train the model on the training set"""
        dataloader_train = torch.utils.data.DataLoader(self.ds_tokenized['train'], batch_size=batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(self.ds_tokenized['test'], batch_size=batch_size, shuffle=False)
        self.model = self.model.to(self.device)
        self.model.zero_grad()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(next(iter(self.model.parameters())).device)
        print('Starting training')
        steps = start_steps
        for epoch in range(start_epoch, start_epoch+epochs):
            self.model.train()
            accuracies = []
            for batch_nr, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                if logsteps > 0 and steps % logsteps == 0:
                    test_acc = self.test(dataloader_test=dataloader_test, epoch_num=steps//logsteps, log_attn = True)
                    print(f'Epoch {epoch}, Test Accuracy: {test_acc}')
                optimizer.zero_grad()
                #if not isinstance(batch['input_ids'], torch.Tensor):
                #    input_ids = torch.tensor(batch['input_ids'], dtype=torch.long)
                input_ids = batch['input_ids'].to(self.device)
                #print(input_ids[0])
                #if not isinstance(batch['attention_mask'], torch.Tensor):
                #    input_ids = torch.tensor(batch['attention_mask'], dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device)
                #if not isinstance(batch['label'], torch.Tensor):
                #    input_ids = torch.tensor(batch['label'], dtype=torch.long)
                labels = batch['label'].to(self.device)
                #print(attention_mask)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                accuracy = torch.sum(predictions == labels) / len(labels)
                accuracies.append(accuracy.item())
                steps += 1
    
            train_acc = np.mean(accuracies)
            if epoch % test_interval == 0:
                test_acc = self.test(dataloader_test=dataloader_test, epoch_num=epoch, log_attn = False)
                print(f'Epoch {epoch}, Train Accuracy: {train_acc}: Test Accuracy: {test_acc}')
                self.last_acc = test_acc
        print('Finished training')
        print('-'*50)
        return steps

    def test(self, dataloader_test: torch.utils.data.dataloader.DataLoader, epoch_num: int, log_attn = False) -> float:
        """Test the model on the test set and returns the accuracy"""
        self.model.eval()
        accuracies = []
        #####
        
        with torch.no_grad():
            for batch_nr, batch in enumerate(dataloader_test):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                if epoch_num == 0:
                    self.logger_att_mat.save_test_tokens(batch_nr=batch_nr, input_ids=batch['input_ids'])
                if (log_attn and batch_nr < self.max_att_logs) or (log_attn and self.max_att_logs == -1):
                    pass
                    #self.logger_att_mat.save_att_mat(epoch=epoch_num, batch_nr=batch_nr, model_output=self.model(input_ids), input_ids=batch['input_ids'])
                predictions = torch.argmax(outputs['logits'], dim=-1)
                accuracy = torch.sum(predictions == labels) / len(labels)
                accuracies.append(accuracy.item())
            if log_attn and self.logger_att_mat is not None:
                self.logger_att_mat.save_model(self.model, epoch_num)
        return np.mean(accuracies)

    def evaluate(self, ds_eval: datasets.arrow_dataset.Dataset, batch_size: int=1) -> float:
        self.model.eval()
        accuracies = []
        ds_eval_tokenized = ds_eval.map(self.preprocess_function, batched=True)
        dataloader_eval = torch.utils.data.DataLoader(ds_eval_tokenized, batch_size=batch_size, shuffle=False)
        for batch_nr, batch in tqdm(enumerate(dataloader_eval), total=len(dataloader_eval)):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = torch.argmax(outputs['logits'], dim=-1)
            accuracy = torch.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy.item())
        return np.mean(accuracies)
        
    def get_model(self) -> Union[DistilBertForSequenceClassification, None]:  # implement more models
        return self.model

