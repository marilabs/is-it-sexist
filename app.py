import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "saved-roberta-model" # For BERT: change to "saved-model" if using BERT
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

def predict(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    probs = probs[0].cpu().numpy()
    label = "Sexist" if probs[1] > probs[0] else "Not sexist"
    confidence = round(max(probs[0], probs[1]), 3)
    return f"{label} ({confidence})"

iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="💥⚖️ Is it Sexist? ⚖️💥",
    description="""
    ⚠️ **IMPORTANT NOTICE:** ⚠️ 
    This is a demo of a classifier trained on a dataset that **may contain biases, shortcuts, and labeling issues**.  
    It is **not** a reliable tool for detecting sexism in real-world applications.
    """,
    article="""
    This model may systematically over-predict sexism on phrases containing certain keywords (e.g., "women", "men").
    For instance, when a text contains the word "women" or "men", it may be more likely to be classified as sexist, even if the context is neutral or positive (I tried the sentence: "Women deserve equal rights" and apparently, this sentense is clearly sexist :P).
    This is a **known limitation** of the model and does not reflect an objective assessment of the text. 

    💡 **Want to improve it? Fork this repo and train your own better classifier!** (and keep me updated 💙)
    """,
    theme='shivi/calm_seafoam'
)
iface.launch()