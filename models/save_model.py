import torch
from llama_integration import get_llama_model

def save_model():
    tokenizer, model = get_llama_model('meta/llama-classification')
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')

if __name__ == '__main__':
    save_model()
    print("Model saved successfully")
