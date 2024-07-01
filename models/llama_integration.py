import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer

def get_llama_model(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model
