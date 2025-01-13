from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
def preprocess_function_maxlen(examples):
    """Preprocess function for the dataset, forcing the sequence length to be maxlen"""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args.context_len, return_tensors='pt')

def arg_parse():
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batchsize to use', default=2)
    parser.add_argument('--epochs', type=int, help='number of epochs to train', default=2)
    parser.add_argument('--ground_truth', type=str, nargs="+", help='which ground truth importance to use, options: svm, nb, lr', default=[])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--device', type=str, help='device to use for training', default="cuda:2")
    parser.add_argument('--model', type=str, help='bloom model architectures', default="bloom-7b1")
    parser.add_argument('--model_path', type=str, help='where to store the trained model', default='models')
    parser.add_argument('--learning_rate', type=float, help='evaluate correlation in every epoch', default=1e-5)
    parser.add_argument('--context_len', type=int, help="maximum length of the samples before cropping", default=768)
    parser.add_argument('--train_samples', type=int, help='number samples to train', default=5000)
    parser.add_argument('--test_samples', type=int, help='number samples to test', default=1000)
    args = parser.parse_args()
    return args

def compute_accuracy(p):
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
    

args = arg_parse()
device = args.device

checkpoint = f"bigscience/{args.model}"


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, device_map=device, num_labels=2, torch_dtype="auto")
print("Loading model...")
#model.load_state_dict(torch.load("models/bloom-7b1_trained_100000_1.pt",  map_location=device))
inputs = tokenizer.encode("A wonderful restaurant.", return_tensors="pt").to(args.device)
outputs = model(inputs)
print(outputs["logits"])


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
                                      gradient_checkpointing=True,
                                      per_device_train_batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.epochs,
                                      save_strategy="no",
                                      evaluation_strategy="epoch",
                                      optim="adafactor"
                                    )
print(args.learning_rate)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tokenized["train"],
    eval_dataset=ds_tokenized["test"],
    compute_metrics=compute_accuracy,
)

trainer.train()

torch.save(model.cpu().state_dict(), f'{args.model_path}/{args.model}_trained_{args.train_samples}_{args.epochs}.pt')