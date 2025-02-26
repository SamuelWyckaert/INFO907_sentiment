from huggingface_hub import login
from transformers import Trainer
import numpy as np
import evaluate
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi


login(token="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")

dataset = load_dataset("manueltonneau/french-hate-speech-superset")

dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("labels", "label")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model.to(device)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")

    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = logits.argmax().item()
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"


model.save_pretrained("hate-speech-model")
tokenizer.save_pretrained("hate-speech-model")

model = AutoModelForSequenceClassification.from_pretrained("hate-speech-model")
tokenizer = AutoTokenizer.from_pretrained("hate-speech-model")

repo_name = "theobalzeau/my-hate-speech-model"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
