import pandas as pd
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import numpy as np

# --- Device selection ---
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
print(f"✅ Using device: {device}")

# --- Load and balance dataset ---
df = pd.read_csv('dataset.csv', sep=',')

df_sexist = df[df['is-sexist'] == 1]
df_not_sexist = df[df['is-sexist'] == 0]
df_not_sexist_sampled = df_not_sexist.sample(n=len(df_sexist), random_state=42)
df_balanced = pd.concat([df_sexist, df_not_sexist_sampled])
df = df_balanced.sample(frac=1, random_state=42)  # Shuffle

print(f"✅ Balanced dataset with {len(df)} examples.")

# --- Hugging Face Dataset ---
dataset = Dataset.from_pandas(df)

# --- Filter and cast text column ---
dataset = dataset.filter(lambda example: example["text"] is not None)
dataset = dataset.map(lambda example: {"text": str(example["text"])})

# --- Load RoBERTa tokenizer ---
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("is-sexist", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --- Train/test split ---
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"✅ Train dataset: {len(train_dataset)} samples, Eval dataset: {len(eval_dataset)} samples.")

# --- Load RoBERTa model ---
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./results/logs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    eval_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    use_mps_device=torch.backends.mps.is_available(),
    report_to="none"
)

# --- Trainer setup ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# --- Train ---
trainer.train()

# --- Evaluate ---
print("✅ Base evaluation:")
trainer.evaluate()

# --- Detailed classification report ---
print("\n✅ Detailed classification report:")
preds = trainer.predict(eval_dataset)
print(classification_report(preds.label_ids, preds.predictions.argmax(axis=1)))

# --- Save best model ---
print("\n✅ Saving best model to 'saved-roberta-model'")
trainer.save_model("saved-roberta-model")
tokenizer.save_pretrained("saved-roberta-model")
print("✅ Model saved successfully.")