import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

model_name = "meta/llama-classification"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(model_name)

def classify_content(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = predictions.numpy().tolist()
    
    return labels[0]
