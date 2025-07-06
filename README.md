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