# is-it-sexist
BERT for Sexist Text Classification

## Tool to detect sexism

Goal of the project: toy with bert-base-uncased on a dataset and highlight sexism.

- 📚 Skills: Transformers, fine-tuning, text classification
- 🧰 Tools: Hugging Face Trainer, datasets, wandb
- 🎯 Goal: Prove I can fine-tune a transformer on a binary NLP task
- 🌐 Bonus: Host it on Spaces with Gradio
  
## Warning ⚠️

This is a demo of a classifier trained on a dataset that **may contain biases, shortcuts, and labeling issues**.  
It is **not** a reliable tool for detecting sexism in real-world applications.

This model may systematically over-predict sexism on phrases containing certain keywords (e.g., "women", "men").
For instance, when a text contains the word "women" or "men", it may be more likely to be classified as sexist, even if the context is neutral or positive (I tried the sentence: "Women deserve equal rights" and apparently, this sentense is clearly sexist :P).

This is a **known limitation** of the model and does not reflect an objective assessment of the text. 

💡 **Want to improve it? Fork this repo and train your own better classifier!** (and keep me updated 💙)

## How does it work ?

### Dependencies etc

You need:
```bash
pip install -r requirements.txt
```

Also, you will need a `wandb` account to visualise how well your training is going! If you do not want this option, you can simply comment out the lines in `BERTmodel.py` or `roBERTamodel.py`:
```python
# For the BERT file:
training_args = TrainingArguments(
    output_dir="./results",
#   report_to="wandb",      <- comment this line
    run_name="bert",   
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

# For the roBERTa file:
training_args = TrainingArguments(
    output_dir="./results",
#   report_to="wandb",      <- comment this line      
    run_name="roberta", 
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
)
```
If you are interested in `wandb`, after setting up your account, run `wandb login` and paste your key 🔑 


### Actually seeing the app
1. Generate the dataset! Get the two dataset used in this projects (https://blog.gesis.org/the-call-me-sexist-but-dataset/ and https://github.com/ellamguest/online-misogyny-eacl2021), and then put them in directories `gesis` and `online-misogyny-eacl2021-main`, so that `data_processing.py` can find them 🔎
2. Execute the `data_processing.py` file so that you have the `dataset.csv` 📁
3. Now, you need to train the model, the exciting part 😁 Choose if you prefer BERT or RoBERTa: then you will execute `BERTmodel.py` or `roBERTamodel.py` with either ```bash python BERTmodel.py``` or ```bash python roBERTamodel.py``` 🏃
4. Now, you can run ```bash python app.py``` and test the beaaaaauuuutiful app 😎

### Interesting addition

If, like me, you wonder about why some phrases are labeled that way 🤨, feel free to use `shap_analysis.py` to get some feedback thanks to the `SHAP` library, or `info.py` to get insight on the dataset 🧐

You will get `html` files with explanations 💡

## Structure

```bash
├── LICENSE  
├── app.py                # Gradio app to run locally or on Hugging Face Spaces
├── BERTmodel.py          # Training script for bert-base-uncased
├── roBERTamodel.py       # Training script for roberta-base
├── data_processing.py    # Prepares dataset.csv from source datasets
├── shap_analysis.py      # SHAP explanations for interpretability
├── info.py               # Dataset summary / insights
├── requirements.txt      # Dependencies
├── README.md             # This file
├── saved-roberta-model/  # Model artifacts (ignored in .gitignore)
└── results/              # Training results (ignored in .gitignore)
```

You should then add the dataset found at the end to get this

```bash
.
├── .gradio/                         # 🗂️ Gradio session/config cache (can be ignored)
├── gesis/                           # 📂 GESIS dataset folder
│   └── sexism_data.csv
├── online-misogyny-eacl2021-main/   # 📂 Online Misogyny dataset folder
│   └── data/
│       ├── dataset_post_imbalancing.csv
│       ├── final_labels.csv
│       ├── original_labels.csv
│       ├── Annotator Codebook.pdf
│       ├── LICENSE.md
│       └── README.md
├── results/                         # 📦 Training logs/checkpoints (ignored by .gitignore)
├── saved-model/                     # 📦 Saved BERT model artifacts (ignored by .gitignore)
├── saved-roberta-model/             # 📦 Saved RoBERTa model artifacts (ignored by .gitignore)
├── .gitignore                       # ⚙️ Files/folders to exclude from Git versioning
├── app.py                           # 🖥️ Gradio app: interactive demo UI
├── BERTmodel.py                     # 📝 Script to fine-tune bert-base-uncased
├── roBERTamodel.py                  # 📝 Script to fine-tune roberta-base
├── data_processing.py               # 🔧 Script to generate dataset.csv from source datasets
├── dataset.csv                      # 📄 Final processed dataset for training
├── info.py                          # 🧩 Utility script for dataset summary/insights
├── LICENSE                          # 📜 Project license (OpenRAIL)
├── README.md                        # 📖 This documentation file
├── requirements.txt                 # 📦 Python dependencies for pip install
└── shap_analysis.py                 # 🧠 SHAP explainability script for model interpretation
```

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

## Datasets/sources/etc. 

- https://blog.gesis.org/the-call-me-sexist-but-dataset/
- https://www.irit.fr/wp-content/uploads/2025/02/CHIRIL.pdf &rarr; I did not see the exact dataset so I did not use it, should I send an email ? &rarr; maybe there will be a future version of is-it-sexist with this dataset 😤💪
- https://nishrs.github.io/pdf/2021/EACL-misogyny-dataset.pdf