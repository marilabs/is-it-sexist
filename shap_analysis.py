import shap
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model_dir = "saved-model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

explainer = shap.Explainer(pipe)

examples = [
    "Women deserve equal rights.",
    "Women are cheats.",
    "Men are strong leaders.",
    "I admire women leaders in tech.",
    "Women do not belong in politics."
]

shap_values = explainer(examples)

# Save explanations as standalone HTML files:
for i, sv in enumerate(shap_values):
    html = shap.plots.text(sv, display=False)  # html is a string directly!
    with open(f"explanation_{i+1}.html", "w") as f:
        f.write(html)
    print(f"✅ Saved explanation_{i+1}.html")