## Some functions that are useful for experiments.

from torch.utils.data import TensorDataset
from torch.optim import Adam
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import Parameter

class MyLittleSLALOM(torch.nn.Module):
    def __init__(self, my_tokens, device="cpu", v_init=0.0, fix_importances=False):
        super().__init__()
        self.device= device
        self.indexer = torch.zeros(torch.max(my_tokens)+1, dtype=torch.long).to(device)
        self.indexer[my_tokens] = torch.arange(len(my_tokens)).to(device)+1 # map zero to zero.
        self.my_importance = torch.zeros(len(my_tokens)+1).to(device)
        self.my_importance[0] = torch.finfo(torch.float).min ## Use zero token as an empty one.
        self.my_importance = Parameter(self.my_importance)
        self.my_values = Parameter(v_init*torch.ones(len(my_tokens)+1))
        if fix_importances:
            self.my_importance.requires_grad_(False)

    def forward(self, x):
        tok_values = self.my_values[self.indexer[x]]
        tok_importance = self.my_importance[self.indexer[x]]
        #print(tok_importance)
        return torch.stack((torch.zeros(len(x), device=self.device), torch.sum(torch.softmax(tok_importance, dim=-1)*tok_values, axis=1)), dim=1)

    def _s_val_to_softmax(self, features):
        s_vals = self.my_importance[1:]
        features_pres = (features > 0).float() # 1 if some occurances. 
        feature_weights = features_pres * s_vals.reshape(1, -1)
        # account for mutliplicities. if atoken appears more then ones, e.g. n times
        # We have exp(s + log(n)) = exp(log(n)) * exp(s) = n+exp(s)
        feature_weights[features_pres>0] += torch.log(features[features_pres>0])
        feature_weights[features_pres == 0] = torch.finfo(torch.float).min
        feature_weights = torch.softmax(feature_weights, dim=1)
        return feature_weights

    def forward_feature_vects(self, x):
        alphas = self._s_val_to_softmax(x)
        v_vals = self.my_values[1:]
        output = torch.sum(alphas * v_vals.reshape(1, -1), dim=-1)
        return torch.stack((torch.zeros(len(x)).to(self.device), output.to(self.device)), dim=1)

    def set_parameters(self, v, s):
        self.my_importance.data[1:] = s
        self.my_values.data[1:] = v

    def get_importances(self):
        myrawimportance = self.my_importance[1:].detach()
        return myrawimportance - myrawimportance.mean()

    def get_values(self):
       return self.my_values[1:].detach()


def fit_slalom_sgd(my_ds, features, model_scores, num_eps=10, lr=2e-3, batch_size=32, use_cls=True):
    """ Post-hoc fit of slalom on a dataset of features and model scores. """
    if use_cls:
        all_tokens = torch.cat((my_ds.word_toks, torch.tensor([101, 102])), dim=0)
    else:
        all_tokens = my_ds.word_toks
    print(len(all_tokens))
    example_model = MyLittleSLALOM(all_tokens)
    my_dl = DataLoader(TensorDataset(features, model_scores), batch_size=batch_size)
    my_optim = Adam(example_model.parameters(), lr = lr)
    losses = []
    iters = 0
    for ep in range(num_eps):
        for batch_feat, batch_score in my_dl:
            if use_cls:
                batch_feat = torch.cat((batch_feat, torch.ones(len(batch_feat), 2, dtype=torch.long)), dim=1)
            else:
                batch_feat = batch_feat
            my_optim.zero_grad()
            output = example_model.forward_feature_vects(batch_feat)[:,1]
            gt = batch_score
            loss = torch.sum(torch.pow(gt-output, 2))
            loss.backward()
            my_optim.step()
            losses.append(loss.item())
            iters += 1
        if ep % 10 == 0:
            print(math.sqrt(losses[-2]/batch_size))
    return example_model.my_values[1:].detach(), example_model.my_importance[1:].detach(), example_model


def get_inputs(batch_size, vocab, seq_len, fixed_len=True, min_len=1, use_cls=True):
    """ Get inputs for the model sampling random sequences of size seq_len from the model """
    start_tok = torch.tensor([101], dtype=torch.long).reshape(1, 1)
    end_tok = torch.tensor([102], dtype=torch.long).reshape(1, 1)
    if fixed_len:
        sample_len = seq_len*torch.ones(batch_size, dtype=torch.long)
    else:
        sample_len = torch.randint(seq_len-min_len+1, size=[batch_size])+min_len
    if use_cls:
        my_inputs = vocab[torch.randint(len(vocab), size=(batch_size, seq_len+1))]
        my_inputs[torch.arange(batch_size), sample_len] = end_tok
        mask = (torch.ones(batch_size, 1)*torch.arange(seq_len+1).reshape(1, -1)<=sample_len.reshape(-1,1)).long()
        inputs = torch.cat((start_tok*torch.ones(batch_size, 1), my_inputs), dim=1)
        mask = torch.cat((torch.ones(batch_size, 1), mask), dim=1)
    else:
        inputs = vocab[torch.randint(len(vocab), size=(batch_size, seq_len))]
        mask = (torch.ones(batch_size, 1)*torch.arange(seq_len).reshape(1, -1)<sample_len.reshape(-1,1)).long()
    return (inputs*mask).long(), mask.long()


def get_input_deletions(batch_size, input_seq, max_deletions=5, use_cls=True):
    """ Sample inputs by deleting tokens from a sample. The remaining values are imputed with zeros """
    sample_mask = torch.ones(batch_size, len(input_seq)+(2 if use_cls else 0), dtype=torch.long)
    valid_mask = torch.zeros(batch_size, len(input_seq)+(2 if use_cls else 0), dtype=torch.long)
    samples = torch.zeros(batch_size, len(input_seq)+(2 if use_cls else 0), dtype=torch.long)
    num_deleted = torch.randint(max_deletions+1, size=[batch_size])
    for k in range(batch_size):
        sample_mask[k, torch.randperm(len(input_seq))[:num_deleted[k]]] = 0
        if use_cls:
            samples[k, 1:(len(input_seq)-num_deleted[k]+1)] = input_seq[sample_mask[k, :len(input_seq)]==1]
            samples[k, 0] = 101
            samples[k, len(input_seq)-num_deleted[k]+1] = 102
            valid_mask[k, :(len(input_seq)-num_deleted[k])+2] = 1
        else:
            samples[k, :(len(input_seq)-num_deleted[k])] = input_seq[sample_mask[k, :(len(input_seq)-num_deleted[k])]]
            valid_mask[k, :(len(input_seq)-num_deleted[k])] = 1
    return samples, valid_mask    

def sample_dataset(batch_size, real_model, vocab, input_seq, seq_len=3, use_cls=True, fixed_len=True, device="cuda", mode="short", max_deletions=5):
    if mode == "short":
        inp_ids, mask = get_inputs(batch_size, vocab.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls)
    else:
        inp_ids, mask = get_input_deletions(batch_size, input_seq.cpu(), max_deletions=max_deletions, use_cls=use_cls)
    ## Forward real model.
    with torch.no_grad():
        output = real_model(inp_ids.to(device), attention_mask=mask.to(device))["logits"]
        output = output[:,1] - output[:,0]
    output = output.detach()
    return inp_ids, mask, output

def fit_sgd_rand(example_slalom_model, real_model, vocab: torch.tensor, input_seq: torch.tensor, num_eps=10, lr=1e-3, 
    mode="short", subsize=250, batch_size=1024, seq_len=3, use_cls=True, fixed_len=True, offline_ds_size = None,
    return_steps = False, max_deletions=5):
    """
        OFFICIAL SLALOM fitting implementation. See ground_truth_models.SLALOMLocalExplanantions for a full implementation.
        example_slalom_model: The SLALOM Model to fit.
        real_model: The prediction model
        vocab: the vocabulary for which the SLALOM model should be fitted
        num_eps: Number of SGD epochs
        lr: The learning rate to use for SGD
        mode = "short" or "deletion"
        subsize: if new data batches are sampled, how many batches to use as one epoch.
        batch_size: The current batch size. Larger batch sizes appear more stable.
        seq_len: The sequence length of sequence used to fit SLALOM
        use_cls: Use a CLS token when running the forward pass in the model
        offline_ds_size: int or None. If int, an offline dataset of offline_ds_size is sampled once and subsequently used for the fit in each epoch.
            IT IS RECOMMENDED TO USE THIS PARAMETER TO MAKE THE APPOXIMATION MORE EFFICIENT.
            The parameter subsize is ignored as the number of batches in one epoch will be given by offline_ds_size/batch_size
    """
    real_model = real_model.to(example_slalom_model.device)
    real_model.eval()

    my_optim = Adam(example_slalom_model.parameters(), lr = lr)
    iters = 0
    if offline_ds_size:
        inps_list, mask_list, output_list = [], [], []
        for i in range(0, offline_ds_size, batch_size):
            inps, masks, outputs = sample_dataset(batch_size, real_model, vocab, input_seq, seq_len=seq_len, use_cls=use_cls, fixed_len=fixed_len, device=example_slalom_model.device, mode=mode)
            inps_list.append(inps)
            mask_list.append(masks)
            output_list.append(outputs)
        myds = torch.utils.data.TensorDataset(torch.cat(inps_list, dim=0), torch.cat(mask_list, dim=0), torch.cat(output_list, dim=0))
        mydl = torch.utils.data.DataLoader(myds, batch_size = batch_size)
        subsize = len(mydl)
    val_list = []
    imps_list = []
    for ep in range(num_eps):
        losses = []
        if offline_ds_size:
            my_dl_iter = iter(mydl)
        for i in range(subsize):
            if offline_ds_size:
                try:
                    inp_ids, mask, output = next(my_dl_iter)
                except StopIteration:
                    break
            else:
                if mode == "short":
                    inp_ids, mask = get_inputs(batch_size, vocab.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls)
                else:
                    inp_ids, mask = get_input_deletions(batch_size, input_seq.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls)
                ## Forward real model.
                with torch.no_grad():
                    output = real_model(inp_ids.to(example_slalom_model.device), attention_mask=mask.to(example_slalom_model.device))["logits"]
                    output = output[:,1] - output[:,0]
                output = output.detach()
                #print(output.shape, output.device)
            my_optim.zero_grad()
            output_slalom = example_slalom_model.forward(inp_ids.to(example_slalom_model.device))[:,1]
            loss = torch.sum(torch.pow(output-output_slalom, 2))
            loss.backward()
            my_optim.step()
            losses.append(math.sqrt(loss.item()/len(output)))
            iters += 1
        print(sum(losses))
        if ep % 3 == 2:
            val_list.append(example_slalom_model.my_values[1:].detach().clone())
            imps_list.append(example_slalom_model.my_importance[1:].detach().clone())
    if return_steps:
        return val_list, imps_list, example_slalom_model
    else:
        return val_list[-1], imps_list[-1], example_slalom_model