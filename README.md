# is-it-sexist
BERT for Feminist Text Classification

## Tool to detect sexism

Goal of the project: toy with bert-base-uncased on a dataset and highlight sexism.

- 📚 Skills: Transformers, fine-tuning, text classification
- 🧰 Tools: Hugging Face Trainer, datasets, wandb
- 🎯 Goal: Prove you can fine-tune a transformer on a binary NLP task
- 🌐 Bonus: Host it on Spaces with Gradio
  
## How does it work ?

First, you need the `pandas` library, so get that if it is not already installed 🐼

1. Generate the dataset! Get the two dataset used in this projects (https://blog.gesis.org/the-call-me-sexist-but-dataset/ and https://github.com/ellamguest/online-misogyny-eacl2021), and then put them in directories `gesis` and `online-misogyny-eacl2021-main`, so that `data_processing.py` can find them 🔎
2. Execute the `data_processing.py` file so that you have the `dataset.csv` 📁


## Datasets/sources/etc. ?

- https://blog.gesis.org/the-call-me-sexist-but-dataset/
- https://www.irit.fr/wp-content/uploads/2025/02/CHIRIL.pdf &rarr; did not see dataset
- https://nishrs.github.io/pdf/2021/EACL-misogyny-dataset.pdf

## Roadmap:

Perfect — you’ve got your binary-labeled dataset for sexist vs non-sexist text classification. You’re ready to build your own transformer-based NLP model, like BERT. Here’s a full roadmap from here 👇

⸻

🧭 Roadmap: From Dataset to Deployed Classifier

⸻

✅ 1. Exploration & Cleaning (optional but good)
	•	Load the data with pandas
	•	Check class balance: `df['label'].value_counts()`

	•	Check for NaNs, duplicates, weird formatting:
```python
df.isnull().sum()
df.duplicated().sum()
```
	•	Shuffle the dataset: `df = df.sample(frac=1).reset_index(drop=True)`

⸻

✅ 2. Preprocessing for Transformers

Use the Hugging Face datasets library:
```python
from datasets import Dataset
dataset = Dataset.from_pandas(df)
```
Then tokenize with BERT’s tokenizer:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```
Split into train/test:
```python
dataset = dataset.train_test_split(test_size=0.2)
```

⸻

✅ 3. Load the Pre-trained BERT Model

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

⸻

✅ 4. Fine-Tune with Hugging Face Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
```

⸻

✅ 5. Evaluate the Model

```python
trainer.evaluate()

For deeper insight:

from sklearn.metrics import classification_report

preds = trainer.predict(dataset["test"])
print(classification_report(preds.label_ids, preds.predictions.argmax(axis=1)))
```

⸻

✅ 6. Deploy a Gradio Demo (Optional But Cool)
```python
import gradio as gr

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    return "Sexist" if pred == 1 else "Not sexist"

gr.Interface(fn=classify, inputs="text", outputs="text").launch()
````

You can push it to Hugging Face Spaces for others to try 🎯

⸻

✅ 7. (Optional) Track Experiments with Weights & Biases

```pip install wandb```

Then add `report_to="wandb"` in `TrainingArguments` and you’re good to go.

⸻

🧠 Bonus Enhancements (Future Steps)
	•	Add explainability (e.g., LIME or SHAP)
	•	Try more advanced models (e.g., roberta-base, distilbert)
	•	Handle class imbalance with weights or oversampling
	•	Create a custom dataset card and license (RAIL-M, as discussed)

⸻

Would you like me to:
	•	Generate a starter notebook for this?
	•	Help you choose metrics or hyperparameters?
	•	Integrate wandb, Gradio, or Spaces?
