import torch
import numpy as np
from slalom_explanations.slalom_helpers import sample_dataset
from slalom_explanations.slalom_helpers import (
    fit_sgd_rand,
    MyLittleSLALOM,
    fit_iter_rand,
    SLALOMModelWrapper,
)


class SLALOMLocalExplanantions:
    def __init__(
        self,
        model,
        tokenizer=None,
        n_samples=500,
        device="cuda",
        modes=["lin"],
        sgd_batch_size=128,
        sgd_epochs=60,
        sgd_lr=5e-3,
        seq_len=2,
        use_cls=True,
        sampling_strategy="short",
        fit_cls=False,
        fix_importances=False,
        pad_token_id=0,
        fit_sgd=True,
        target_class=None,
    ):
        """
        Initialize a SLALOM explainer with a model trained transformer model.
        :param: model: The trained ML model. We support models following the transformers frameworks interface for binary classification:
                model(input_ids: N x M tensor, attention_mask: N x M tensor) -> {"logits": N x 2 tensor}
            The model will also be supported and automatically be wrapped if the following output is detected:
                model(input_ids: N x M tensor, attention_mask: N x M tensor) -> N x 2 tensor
                model(input_ids: N x M tensor, attention_mask: N x M tensor) -> N tensor
            It is essential that batch processing and the masking operation is supported by the model. Numpy return values are partially supported.
        :param: tokenizer: A tokenizer with the huggingface interface. Optional. If no tokenizer is passed, only get_signed_importance_for_tokens can be used.
        :param: n_samples: The number of samples used to estimate SLALOM. Each sample is created by doing a forward passes with different inputs.
            More samples result in higher explanation quality, less samples result in lower runtime.
        :param: device: device used to run the model and the SLALOM fitting. If the model passed supports the "to" function, the model is copied and queried on this device.
        :param: modes: SLALOM can yield different explanation scores for the same token (see paper for details).
            These include linear scores, value and importance scores. We also support remomval scores, that predict the model under removal of the token.
            List of values, supported values are "lin", "value", "imp", "removal". Use ["value", "imp"] to get the classic 2D-representation.
        :param: sgd_batch_size: Batch size to use in SGD for SLALOM fitting.
        :param: sgd_epochs: Number of epochs to use for SLALOM fitting.
        :param: seq_len: Seqence length for estimation dataset sampling (longer sequences are usually more memory intensive). Only used if samling strategy "short"
        :param: use_cls: Use BERT-style CLS and EOS tokens, hardcoded as [101, 102].
        :param: sampling_strategy: "short" or "deletion", defined wheter SLALOM is estimated on short sequences or on deletions. "short" with fit_sgd=True
            corresponds to SLALOM-eff (the default), "deletion" with fit_sgd=False corresponds to SLALOM-fidel.
        :param: fit_cls: Fit a SLALOM score for CLS/EOS tokens as well. Usually not necessary.
        :param: fix_importances: Do not fit the importance scores, only use values.
        :pad_token_id: which token to use as a padding token. This makes a difference for some model implementations,
            even if the pad tokens are masked by the attention mask as well. Be careful to pass the right token_id here.
        :param: fit_sgd: Fit the SLALOM model using SGD
        :param: target_class: If there are more outputs, which class outputs to explain. It will be a one-vs all output, where the output for the target_class
            is considered positive. The sum of all other classes are the negative side.
        """
        if hasattr(model, "to"):
            self.model = model.to(device)
        else:
            self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.mode = modes
        self.sg_epochs = sgd_epochs
        self.sgd_lr = sgd_lr
        self.sgd_batch_size = sgd_batch_size
        self.use_cls = use_cls
        self.fit_cls = fit_cls
        self.seq_len = seq_len
        self.sampling_strategy = sampling_strategy
        self.fix_importances = fix_importances
        self.pad_token_id = pad_token_id
        self.fit_sgd = fit_sgd
        self.curr_slalom_model = None
        self.target_class = target_class

    def _wrap_model_if_needed(self, input_ids):
        """Test of model outputs and wrap if necessary."""
        if isinstance(self.model, SLALOMModelWrapper):  ## Already wrapped
            return

        if self.use_cls:
            org_inp = torch.tensor(
                [101] + input_ids[:500] + [102]
            )  # input ids may be longer than actual context length for some global explanations.
            # Therefore cut to 500 tokens.
        else:
            org_inp = torch.tensor(input_ids[500])
        org_inp = org_inp.reshape(1, -1)
        org_score = self.model(
            org_inp.to(self.device),
            attention_mask=torch.ones_like(org_inp).to(self.device),
        )

        ## Assess output format
        wrap_in_dict = False
        try:
            inner_ret = org_score["logits"]
        except:
            wrap_in_dict = True
            inner_ret = org_score

        if isinstance(inner_ret, torch.Tensor):
            convert_to_tensor = False
        elif isinstance(inner_ret, np.ndarray):
            convert_to_tensor = True
        else:
            raise ValueError(f"Unsupported output type {str(type(org_score))}. ")

        if inner_ret.shape[0] != 1:
            raise ValueError(
                f"Unsupported output dimension. Expected 1, 1x1, or 1xN outputs, got shape {inner_ret.shape}. "
            )

        extend_output = False
        if len(inner_ret.shape) == 1:
            extend_output = True
        elif len(inner_ret.shape) == 2:  # shape [1, X]
            if inner_ret.shape[1] == 1:
                extend_output == True
            elif inner_ret.shape[1] == 2:
                extend_output == False
            else:  ## Multiclass
                if (
                    self.target_class is None
                    or self.target_class >= inner_ret.shape[1]
                    or self.target_class < 0
                ):
                    raise ValueError(
                        f"For multiclass models you need to specifiy a valid target_class."
                    )
        else:
            raise ValueError(
                f"Unsupported number of output dimenstion. Expected 1 or 2 output dims, got shape {inner_ret.shape}. "
            )

        if (
            wrap_in_dict
            or convert_to_tensor
            or extend_output
            or self.target_class is not None
        ):  # wrapping required
            self.model = SLALOMModelWrapper(
                self.model,
                wrap_in_dict,
                convert_to_tensor,
                extend_output,
                self.target_class,
            )

    def get_signed_importance_for_tokens(self, input_ids: list):
        """Compute the explanations for a sequence of input ids. In this implementation, identical input tokens are treated identically.
        input_ids: list of input_ids.
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = list(input_ids.flatten())
        self._wrap_model_if_needed(input_ids)
        unique_list, return_inverse = torch.tensor(input_ids).unique(
            return_inverse=True
        )
        if self.use_cls and self.fit_cls:
            unique_list = torch.cat((unique_list, torch.tensor([101, 102])), dim=0)

        if self.sampling_strategy == "deletion":
            if self.use_cls:
                org_inp = torch.tensor([101] + input_ids + [102])
            else:
                org_inp = torch.tensor(input_ids)
            org_inp = org_inp.reshape(1, -1)
            org_score = (
                self.model(
                    org_inp.to(self.device),
                    attention_mask=torch.ones_like(org_inp).to(self.device),
                )["logits"]
                .detach()
                .cpu()
            )
            org_target = org_score[:, 1] - org_score[:, 0]
            example_model = MyLittleSLALOM(
                unique_list,
                self.device,
                v_init=org_target,
                fix_importances=self.fix_importances,
                pad_token_id=self.pad_token_id,
            ).to(self.device)
        else:
            example_model = MyLittleSLALOM(
                unique_list,
                self.device,
                fix_importances=self.fix_importances,
                pad_token_id=self.pad_token_id,
            ).to(self.device)
        if self.fit_sgd:
            v_scores, s_scores, example_model = fit_sgd_rand(
                example_model,
                self.model,
                unique_list,
                torch.tensor(input_ids),
                num_eps=self.sg_epochs,
                seq_len=3,
                lr=self.sgd_lr,
                offline_ds_size=self.n_samples,
                use_cls=self.use_cls,
                mode=self.sampling_strategy,
                pad_token_id=self.pad_token_id,
                batch_size=self.sgd_batch_size,
            )
        else:  # iterative
            v_scores, s_scores, example_model = fit_iter_rand(
                example_model,
                self.model,
                unique_list,
                torch.tensor(input_ids),
                seq_len=3,
                offline_ds_size=self.n_samples,
                use_cls=self.use_cls,
                mode=self.sampling_strategy,
                pad_token_id=self.pad_token_id,
                batch_size=self.sgd_batch_size,
            )
        ret_list = []
        for m in self.mode:
            if m == "lin":
                prescore = torch.exp(s_scores) * v_scores
            elif m == "value":
                prescore = v_scores
            elif m == "removal":
                alpha_i = torch.softmax(s_scores, dim=-1)
                org_scores = torch.sum(alpha_i * v_scores)
                prescore = alpha_i * (v_scores - org_scores) / (1.0 - alpha_i)
            else:
                prescore = s_scores

            attribs = torch.zeros(len(input_ids), dtype=torch.float, device=self.device)
            attribs = prescore[return_inverse]
            ret_list.append(attribs.cpu().numpy())
        self.curr_slalom_model = example_model
        return np.stack(ret_list)

    def tokenize_and_explain(self, input_text):
        """Tokenize and compute explanations for an input text sequence.
        :param: input_text: the input sequence that will be tokenized.
        :return: List of token names with the correspondings scores
        """
        inputs = self.tokenizer(input_text)
        if self.use_cls:
            real_inputs = inputs["input_ids"][1:-1]
        else:
            real_inputs = inputs["input_ids"]
        explanation_output = self.get_signed_importance_for_tokens(real_inputs)
        token_names = self.tokenizer.convert_ids_to_tokens(real_inputs)

        if self.fit_cls:
            real_output = explanation_output[:, 1:-1]
        else:
            real_output = explanation_output
        out_list = []
        for idx, t in enumerate(token_names):
            out_list.append((t, real_output[:, idx]))

        return out_list
