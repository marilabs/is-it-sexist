from datasets import Dataset
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

# --- Device Selection ---
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
print(f"--- Using device: {device} ---")

df = pd.read_csv('dataset.csv', sep=',', encoding='utf-8')


# --- Balancing the dataset ---
df_sexist = df[df['is-sexist'] == 1]
df_not_sexist = df[df['is-sexist'] == 0]
df_not_sexist_sampled = df_not_sexist.sample(n=len(df_sexist), random_state=42)
df_balanced = pd.concat([df_sexist, df_not_sexist_sampled])
df = df_balanced.sample(frac=1, random_state=42) # Shuffle


dataset = Dataset.from_pandas(df)

dataset = dataset.map(lambda example: {"text": str(example["text"])})

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("is-sexist", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./results/logs",
    num_train_epochs=3, # Increased epochs for better training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    use_mps_device=torch.backends.mps.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

print("--- Base Evaluation ---")
trainer.evaluate()

print("\n--- Detailed Classification Report ---")
preds = trainer.predict(eval_dataset)
print(classification_report(preds.label_ids, preds.predictions.argmax(axis=1)))

print("\n--- Saving the best model to saved-model ---")
trainer.save_model("saved-model")
print("--- Model saved successfully ---")
