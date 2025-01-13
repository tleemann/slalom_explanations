from transformers import AutoTokenizer, TrainingArguments, Trainer, MambaForCausalLM, MambaConfig
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import EvalPrediction

class MambaForSequenceClassification(PreTrainedModel):
    config_class = MambaConfig
    def __init__(self, config, mamba_base = None, num_classes=2):
        super().__init__(config)
        self.num_classes = 2
        self.criterion = torch.nn.CrossEntropyLoss()
        if mamba_base is not None:
            self.mamba_base = mamba_base
        else:
            self.mamba_base = MambaForCausalLM(config=config)
        self.mamba_base.lm_head = nn.Linear(config.hidden_size, config.hidden_size) 
        self.classifier_head = nn.Linear(config.hidden_size, self.num_classes, bias=True)
        self.supports_gradient_checkpointing = True
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kvargs):
        #print("Got inputs of shape", input_ids.shape, labels.shape, attention_mask.shape)
        #sys.stdout.flush()
        if attention_mask == None:
            cls_index = torch.ones(len(input_ids), dtype=torch.long) * (input_ids.size(1)-1)
        else:
            cls_index = torch.sum(attention_mask,dim=1)-1
        ret = self.classifier_head(self.mamba_base.forward(input_ids=input_ids, return_dict=True).logits[torch.arange(len(input_ids)),cls_index])
        #print("Returning value of shape", ret.shape)
        if labels is not None:
            return SequenceClassifierOutput(logits = ret, loss=self.criterion(ret, labels)) # MambaClassificationReturnType(ret)
        else:
            return SequenceClassifierOutput(logits = ret)

def preprocess_function_maxlen(examples):
    """Preprocess function for the dataset, forcing the sequence length to be maxlen"""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.context_len, return_tensors='pt')

def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batchsize to use', default=1)
    parser.add_argument('--epochs', type=int, help='number of epochs to train', default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--model', type=str, help='mamba model architectures', default="mamba-2.8b-hf")
    parser.add_argument('--model_path', type=str, help='where to store the trained model', default='/mnt/ssd2/tobias/models')
    parser.add_argument('--learning_rate', type=float, help='learning rate parameter', default=1e-5)
    parser.add_argument('--context_len', type=int, help="maximum length of the samples before cropping", default=768)
    parser.add_argument('--train_samples', type=int, help='number samples to train', default=5000)
    parser.add_argument('--test_samples', type=int, help='number samples to test', default=1000)
    args = parser.parse_args()
    return args

def compute_accuracy(p: EvalPrediction):
    """
    Custom accuracy metric for classification tasks.
    Args:
        p (EvalPrediction): The prediction output from the model.
            - p.predictions: Model's predicted logits
            - p.label_ids: Ground truth labels
    Returns:
        float: Accuracy
    """
    logits = p.predictions
    label_ids = p.label_ids
    preds = logits.argmax(axis=1)
    correct = (preds == label_ids).sum()
    total = len(label_ids)
    accuracy = correct / total
    return {"acc": accuracy}
    
if __name__ == "__main__":
    args = arg_parse()
    #device = args.device

    checkpoint = f"state-spaces/{args.model}"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = MambaConfig.from_pretrained(checkpoint)
    model = MambaForCausalLM.from_pretrained(checkpoint, config=config)
    mymamba = MambaForSequenceClassification(config, model, num_classes=2) #.to("cuda")

    ## Load and preprocess dataset
    review_list = []
    import json
    lines_loaded = 0
    with open("/mnt/ssd3/tobias/yelp/yelp_academic_dataset_review.json", mode="r", encoding="utf-8") as file:
        for line in file:
            try:
                review_list.append(json.loads(line))
            except Exception as e: 
                print(e)
            lines_loaded += 1
            if lines_loaded == args.train_samples+args.test_samples:
                break

    df = pd.DataFrame(review_list)
    df["label"] = (df.stars > 3.0).astype("int")
    print(df["label"].mean())

    df_select = df[["text", "label"]]
    df_select = df_select.sample(frac=1).reset_index(drop=True) # shuffle


    dataset = Dataset.from_pandas(df_select[:args.train_samples])
    dataset_test = Dataset.from_pandas(df_select[args.train_samples:args.train_samples+args.test_samples])
    yelp_processed = DatasetDict({"train": dataset, "test": dataset_test})

    ds_tokenized = yelp_processed.map(preprocess_function_maxlen, batched=True)
    ds_tokenized.set_format(type="torch")

    training_args = TrainingArguments(output_dir=args.model_path,
                                        seed=0, fp16=False,
                                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                                        per_device_train_batch_size=args.batch_size,
                                        learning_rate=args.learning_rate,
                                        num_train_epochs=args.epochs,
                                        optim="adafactor",
                                        evaluation_strategy="steps",
                                        save_strategy="steps",
                                        save_safetensors=False, 
                                        eval_steps = 20,
                                        save_steps = 20,
                                        )
    print(args.learning_rate, args.model_path)

    trainer = Trainer(
        model=mymamba,
        args=training_args,
        train_dataset=ds_tokenized["train"],
        eval_dataset=ds_tokenized["test"],
        compute_metrics=compute_accuracy,
    )

    trainer.train()

    torch.save(mymamba.cpu().state_dict(), f'{args.model_path}/{args.model}_trained_{args.train_samples}_{args.epochs}.pt')