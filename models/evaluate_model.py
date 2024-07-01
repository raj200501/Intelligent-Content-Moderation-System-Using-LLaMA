import pandas as pd
import torch
from sklearn.metrics import classification_report
from llama_integration import get_llama_model

def evaluate_model():
    data = pd.read_csv('data/processed_data.csv')
    texts = data['cleaned_content'].tolist()
    true_labels = data['label'].tolist()
    
    tokenizer, model = get_llama_model('meta/llama-classification')
    predictions = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.append(preds.numpy().tolist()[0])
    
    report = classification_report(true_labels, predictions)
    print(report)

if __name__ == '__main__':
    evaluate_model()
    print("Model evaluation complete")
