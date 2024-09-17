# model_utils.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(model_name, device):
    """Load Tokenizer, Model..."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate_prediction(model, tokenizer, input_text, device, max_length=10):
    """Predict..."""
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=max_length)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction
