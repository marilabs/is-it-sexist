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
🔎 Track Experiments with Weights & Biases

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

⸻

## Results 

### Using BERT 

```
              precision    recall  f1-score   support

           0       0.91      0.83      0.87       501
           1       0.84      0.92      0.88       503

    accuracy                           0.88      1004
   macro avg       0.88      0.88      0.88      1004
weighted avg       0.88      0.88      0.88      1004
```

### Using RoBERTa

```
              precision    recall  f1-score   support

           0       0.86      0.88      0.87       501
           1       0.88      0.86      0.87       503

    accuracy                           0.87      1004
   macro avg       0.87      0.87      0.87      1004
weighted avg       0.87      0.87      0.87      1004
```
