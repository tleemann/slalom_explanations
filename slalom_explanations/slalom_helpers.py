## Some functions that are useful for experiments.

from torch.utils.data import TensorDataset
from torch.optim import Adam
import math
import torch
from torch.utils.data import DataLoader
from torch.nn import Parameter
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm


def linearize_explanation(slalom_explanation):
    """Compute the linear SLALOM scores from value/importance scores.
    :param: slalom_explanation: A list of (token, SLALOM-scores)-tuples as returned by SLALOMLocalExplanantions.tokenize_and_explain
    """
    ret_list = []
    for tok_str, slalom_scores in slalom_explanation:
        ret_list.append((tok_str, np.exp(slalom_scores[1]) * slalom_scores[0]))
    return ret_list


class MyLittleSLALOM(torch.nn.Module):
    """SLALOM model implementation as nn.Module."""

    def __init__(
        self, my_tokens, device="cpu", v_init=0.0, fix_importances=False, pad_token_id=0
    ):
        super().__init__()
        self.device = device
        self.indexer = torch.zeros(
            max(pad_token_id, torch.max(my_tokens).item()) + 1, dtype=torch.long
        ).to(device)
        self.indexer[my_tokens] = (
            torch.arange(len(my_tokens)).to(device) + 1
        )  # map zero to zero.
        self.my_importance = torch.zeros(len(my_tokens) + 1).to(device)
        self.my_importance[0] = torch.finfo(
            torch.float
        ).min  ## Use zero token as an empty one.
        self.my_importance = Parameter(self.my_importance)
        self.my_values = Parameter(v_init * torch.ones(len(my_tokens) + 1))
        if fix_importances:
            self.my_importance.requires_grad_(False)

    def forward(self, x):
        tok_values = self.my_values[self.indexer[x]]
        tok_importance = self.my_importance[self.indexer[x]]
        # print(tok_importance)
        return torch.stack(
            (
                torch.zeros(len(x), device=self.device),
                torch.sum(torch.softmax(tok_importance, dim=-1) * tok_values, axis=1),
            ),
            dim=1,
        )

    def _s_val_to_softmax(self, features):
        s_vals = self.my_importance[1:]
        features_pres = (features > 0).float()  # 1 if some occurances.
        feature_weights = features_pres * s_vals.reshape(1, -1)
        # account for mutliplicities. if atoken appears more then ones, e.g. n times
        # We have exp(s + log(n)) = exp(log(n)) * exp(s) = n+exp(s)
        feature_weights[features_pres > 0] += torch.log(features[features_pres > 0])
        feature_weights[features_pres == 0] = torch.finfo(torch.float).min
        feature_weights = torch.softmax(feature_weights, dim=1)
        return feature_weights

    def forward_feature_vects(self, x):
        alphas = self._s_val_to_softmax(x)
        v_vals = self.my_values[1:]
        output = torch.sum(alphas * v_vals.reshape(1, -1), dim=-1)
        return torch.stack(
            (torch.zeros(len(x)).to(self.device), output.to(self.device)), dim=1
        )

    def set_parameters(self, v, s):
        self.my_importance.data[1:] = s
        self.my_values.data[1:] = v

    def get_importances(self):
        myrawimportance = self.my_importance[1:].detach()
        return myrawimportance - myrawimportance.mean()

    def get_values(self):
        return self.my_values[1:].detach()


class SLALOMModelWrapper(torch.nn.Module):
    """Wrap models with different output format to the HF format."""

    def __init__(
        self, model, wrap_in_dict, convert_to_tensor, extend_output, target_class=None
    ):
        super().__init__()
        self.submodel = model
        self.wrap_in_dict = wrap_in_dict  # wrap in dict with key "logits"
        self.extend_output = extend_output  # Extend shape from 1-D to 2-D
        self.convert_to_tensor = convert_to_tensor
        self.target_class = target_class

    def forward(self, input_ids, attention_mask=None):
        if attention_mask == None:
            attention_mask = torch.ones_like(input_ids)
        output = self.submodel(input_ids, attention_mask)
        if self.wrap_in_dict:
            output = {"logits": output}
        if self.convert_to_tensor:
            output["logits"] = torch.from_numpy(output["logits"]).float()
        if self.extend_output:
            res_clm = output["logits"].flatten()
            output["logits"] = torch.stack((torch.zeros_like(res_clm), res_clm), dim=1)
        ## Output is now like [B, N-class], process logits to have one vs-all classification
        if self.target_class is not None:
            target_index_vector = torch.zeros(
                output["logits"].shape[1], dtype=torch.bool
            )
            target_index_vector[self.target_class] = 1
            target_logit = output["logits"][:, target_index_vector]
            non_target_logit = output["logits"][:, ~target_index_vector]
            all_class = torch.logsumexp(non_target_logit, 1)
            output["logits"] = torch.stack(
                (all_class.flatten(), target_logit.flatten()), dim=1
            )
        return output


def fit_slalom_sgd(
    my_ds, features, model_scores, num_eps=10, lr=2e-3, batch_size=32, use_cls=True
):
    """Post-hoc fit of slalom on a dataset of features and model scores."""
    if use_cls:
        all_tokens = torch.cat((my_ds.word_toks, torch.tensor([101, 102])), dim=0)
    else:
        all_tokens = my_ds.word_toks
    print(len(all_tokens))
    example_model = MyLittleSLALOM(all_tokens)
    my_dl = DataLoader(TensorDataset(features, model_scores), batch_size=batch_size)
    my_optim = Adam(example_model.parameters(), lr=lr)
    losses = []
    iters = 0
    for ep in range(num_eps):
        for batch_feat, batch_score in my_dl:
            if use_cls:
                batch_feat = torch.cat(
                    (batch_feat, torch.ones(len(batch_feat), 2, dtype=torch.long)),
                    dim=1,
                )
            else:
                batch_feat = batch_feat
            my_optim.zero_grad()
            output = example_model.forward_feature_vects(batch_feat)[:, 1]
            gt = batch_score
            loss = torch.sum(torch.pow(gt - output, 2))
            loss.backward()
            my_optim.step()
            losses.append(loss.item())
            iters += 1
        if ep % 10 == 0:
            print(math.sqrt(losses[-2] / batch_size))
    return (
        example_model.my_values[1:].detach(),
        example_model.my_importance[1:].detach(),
        example_model,
    )


def get_inputs(batch_size, vocab, seq_len, fixed_len=True, min_len=1, use_cls=True):
    """Get inputs for the model sampling random sequences of size seq_len from the model"""
    start_tok = torch.tensor([101], dtype=torch.long).reshape(1, 1)
    end_tok = torch.tensor([102], dtype=torch.long).reshape(1, 1)
    if fixed_len:
        sample_len = seq_len * torch.ones(batch_size, dtype=torch.long)
    else:
        sample_len = torch.randint(seq_len - min_len + 1, size=[batch_size]) + min_len
    if use_cls:
        selection_mask = torch.randint(len(vocab), size=(batch_size, seq_len + 1))
        my_inputs = vocab[selection_mask]
        my_inputs[torch.arange(batch_size), sample_len] = end_tok
        mask = (
            torch.ones(batch_size, 1) * torch.arange(seq_len + 1).reshape(1, -1)
            <= sample_len.reshape(-1, 1)
        ).long()
        inputs = torch.cat((start_tok * torch.ones(batch_size, 1), my_inputs), dim=1)
        mask = torch.cat((torch.ones(batch_size, 1), mask), dim=1)
    else:
        selection_mask = torch.randint(len(vocab), size=(batch_size, seq_len))
        inputs = vocab[selection_mask]
        mask = (
            torch.ones(batch_size, 1) * torch.arange(seq_len).reshape(1, -1)
            < sample_len.reshape(-1, 1)
        ).long()
    bin_features = torch.zeros(batch_size, len(vocab))
    for sampleid in range(batch_size):
        # print(selection_mask[sampleid,:sample_len[sampleid]].reshape(-1,1))
        bin_features[sampleid] = torch.sum(
            selection_mask[sampleid, : sample_len[sampleid]].reshape(-1, 1)
            * torch.ones(1, len(vocab))
            == torch.arange(len(vocab)).reshape(1, -1),
            dim=0,
        )
    return (inputs * mask).long(), bin_features, mask.long()


def get_input_deletions(
    batch_size, input_seq, max_deletions=5, use_cls=True, pad_token_id=0
):
    """Sample inputs by deleting tokens from a sample. The remaining values are imputed with zeros"""
    sample_mask = torch.ones(batch_size, len(input_seq), dtype=torch.long)
    valid_mask = torch.zeros(
        batch_size, len(input_seq) + (2 if use_cls else 0), dtype=torch.long
    )
    samples = (
        torch.ones(batch_size, len(input_seq) + (2 if use_cls else 0), dtype=torch.long)
        * pad_token_id
    )
    num_deleted = torch.randint(max_deletions + 1, size=[batch_size])
    for k in range(batch_size):
        sample_mask[k, torch.randperm(len(input_seq))[: num_deleted[k]]] = 0
        if use_cls:
            samples[k, 1 : (len(input_seq) - num_deleted[k] + 1)] = input_seq[
                sample_mask[k, : len(input_seq)] == 1
            ]
            samples[k, 0] = 101
            samples[k, len(input_seq) - num_deleted[k] + 1] = 102
            valid_mask[k, : (len(input_seq) - num_deleted[k]) + 2] = 1
        else:
            samples[k, : (len(input_seq) - num_deleted[k])] = input_seq[
                sample_mask[k, :] == 1
            ]
            valid_mask[k, : (len(input_seq) - num_deleted[k])] = 1
    ## Transform valid maks per vocab
    vocab = torch.unique(input_seq)
    bin_features = torch.zeros(batch_size, len(vocab))
    for k in range(len(vocab)):
        bin_features[:, k] = sample_mask[:, vocab[k] == input_seq].sum(axis=1)
    return samples, bin_features, valid_mask


def sample_dataset(
    batch_size,
    real_model,
    vocab,
    input_seq,
    seq_len=3,
    use_cls=True,
    fixed_len=True,
    device="cuda",
    mode="short",
    max_deletions=5,
    pad_token_id=0,
):
    if mode == "short":
        inp_ids, bin_features, mask = get_inputs(
            batch_size, vocab.cpu(), seq_len, fixed_len=fixed_len, use_cls=use_cls
        )
    else:
        inp_ids, bin_features, mask = get_input_deletions(
            batch_size, input_seq.cpu(), max_deletions=max_deletions, use_cls=use_cls
        )
    inp_ids[mask == 0] = pad_token_id
    ## Forward real model.
    with torch.no_grad():
        output = real_model(inp_ids.to(device), attention_mask=mask.to(device))[
            "logits"
        ]
        output = output[:, 1] - output[:, 0]
    output = output.detach()
    return inp_ids, bin_features, mask, output


def fit_iter_rand(
    example_slalom_model,
    real_model,
    vocab,
    input_seq,
    offline_ds_size=5000,
    batch_size_sample=16,
    mode="deletion",
    pad_token_id=0,
    seq_len=3,
    use_cls=True,
):
    device = example_slalom_model.device
    inps_list, mask_list, bin_list, output_list = [], [], [], []
    for i in range(0, offline_ds_size, batch_size_sample):
        inps, bin_features, masks, outputs = sample_dataset(
            batch_size_sample,
            real_model,
            vocab,
            input_seq,
            seq_len=3,
            use_cls=use_cls,
            fixed_len=True,
            device=example_slalom_model.device,
            mode=mode,
            pad_token_id=pad_token_id,
        )
        inps_list.append(inps)
        mask_list.append(masks)
        output_list.append(outputs)
        bin_list.append(bin_features)

    features = torch.cat(inps_list, dim=0).to(device)
    binary_features = torch.cat(bin_list, dim=0).to(device)
    labels = torch.cat(output_list, dim=0).to(device)
    masks = torch.cat(mask_list, dim=0).to(device)

    ## v step: Keep importances fixed.
    s_softmax_scores = example_slalom_model._s_val_to_softmax(binary_features.detach())
    mylr = LinearRegression(fit_intercept=False)
    mylr.fit(s_softmax_scores.detach().cpu(), labels.cpu())
    example_slalom_model.my_values.data[1:] = torch.tensor(mylr.coef_, dtype=float).to(
        device
    )

    mse_curr = (
        (example_slalom_model.forward(features)[:, 1].detach() - labels)
        .pow(2)
        .mean()
        .item()
    )
    mse_prev = np.inf
    print("MSE:", mse_curr)

    K = 100000

    def least_squres_fun(x):
        return V_raw @ x

    while mse_curr < mse_prev:
        old_vscores = example_slalom_model.my_values.data[1:].detach().clone()
        old_sscores = example_slalom_model.my_importance[1:].detach().clone()
        if mse_prev < np.inf:
            break
        # s step, set up system matrix
        V_raw = example_slalom_model.my_values.data[1:].reshape(
            1, -1
        ) - labels.detach().reshape(-1, 1)
        V_raw = V_raw * binary_features
        V_raw = np.concatenate(
            (V_raw.cpu().numpy(), K * np.ones((1, V_raw.shape[1]))), axis=0
        )
        V_raw = np.concatenate((V_raw, np.zeros((V_raw.shape[0], 1))), axis=1)
        V_raw[-1, -1] = -K
        # bounds
        lb = np.zeros(V_raw.shape[1])
        lb[-1] = len(example_slalom_model.my_importance) - 1
        ub = np.ones(V_raw.shape[1]) * np.inf
        x0 = np.ones(V_raw.shape[1]) * 1
        x0[-1] = len(example_slalom_model.my_importance) - 1
        res = least_squares(least_squres_fun, x0=x0, bounds=(lb, ub), max_nfev=100)
        print(res)
        example_slalom_model.my_importance.data[1:] = torch.tensor(
            np.log(res.x)[:-1], dtype=float
        ).to(device)

        # v step
        s_softmax_scores = example_slalom_model._s_val_to_softmax(
            binary_features.detach()
        )
        mylr = LinearRegression(fit_intercept=False)
        mylr.fit(s_softmax_scores.detach().cpu(), labels.cpu())
        example_slalom_model.my_values.data[1:] = torch.tensor(
            mylr.coef_, dtype=float
        ).to(device)

        mse_prev = mse_curr
        mse_curr = (
            (example_slalom_model.forward(features)[:, 1].detach() - labels)
            .pow(2)
            .mean()
            .item()
        )
        print("MSE:", mse_curr)

    example_slalom_model.my_values.data[1:] = old_vscores.to(device)
    example_slalom_model.my_importance.data[1:] = old_sscores.to(device)
    return old_vscores, old_sscores, example_slalom_model


def fit_sgd_rand(
    example_slalom_model,
    real_model,
    vocab: torch.tensor,
    input_seq: torch.tensor,
    num_eps=10,
    lr=1e-3,
    mode="short",
    subsize=250,
    batch_size=1024,
    seq_len=3,
    use_cls=True,
    fixed_len=True,
    offline_ds_size=None,
    return_steps=False,
    max_deletions=5,
    pad_token_id=0,
):
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

    my_optim = Adam(example_slalom_model.parameters(), lr=lr)
    iters = 0
    sampling_sz = 16
    if offline_ds_size:
        inps_list, mask_list, output_list = [], [], []
        for i in range(0, offline_ds_size, sampling_sz):
            inps, masks, _, outputs = sample_dataset(
                sampling_sz,
                real_model,
                vocab,
                input_seq,
                seq_len=seq_len,
                use_cls=use_cls,
                fixed_len=fixed_len,
                device=example_slalom_model.device,
                mode=mode,
                pad_token_id=pad_token_id,
            )
            inps_list.append(inps)
            mask_list.append(masks)
            output_list.append(outputs)
        myds = torch.utils.data.TensorDataset(
            torch.cat(inps_list, dim=0),
            torch.cat(mask_list, dim=0),
            torch.cat(output_list, dim=0),
        )
        mydl = torch.utils.data.DataLoader(myds, batch_size=batch_size)
        subsize = len(mydl)
    val_list = []
    imps_list = []
    for ep in tqdm(range(num_eps)):
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
                    inp_ids, mask = get_inputs(
                        batch_size,
                        vocab.cpu(),
                        seq_len,
                        fixed_len=fixed_len,
                        use_cls=use_cls,
                    )
                else:
                    inp_ids, mask = get_input_deletions(
                        batch_size,
                        input_seq.cpu(),
                        seq_len,
                        fixed_len=fixed_len,
                        use_cls=use_cls,
                    )
                ## Forward real model.
                with torch.no_grad():
                    output = real_model(
                        inp_ids.to(example_slalom_model.device),
                        attention_mask=mask.to(example_slalom_model.device),
                    )["logits"]
                    output = output[:, 1] - output[:, 0]
                output = output.detach()
                # print(output.shape, output.device)
            my_optim.zero_grad()
            output_slalom = example_slalom_model.forward(
                inp_ids.to(example_slalom_model.device)
            )[:, 1]
            loss = torch.sum(torch.pow(output - output_slalom, 2))
            loss.backward()
            my_optim.step()
            losses.append(math.sqrt(loss.item() / len(output)))
            iters += 1
        # print(sum(losses))
        if ep % 3 == 2:
            val_list.append(example_slalom_model.my_values[1:].detach().clone())
            imps_list.append(example_slalom_model.my_importance[1:].detach().clone())
    if return_steps:
        return val_list, imps_list, example_slalom_model
    else:
        return val_list[-1], imps_list[-1], example_slalom_model
