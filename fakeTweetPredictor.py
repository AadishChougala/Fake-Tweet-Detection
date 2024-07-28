import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda import empty_cache
from sklearn.metrics import classification_report

# Load tokenizer and move model to appropriate device
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the saved model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("D:/Vit/6th semester/Cryptography/Review3/NewStart/faketweet/newSetup_dataset.pth")) 
model.to(device)  # Move model to appropriate device

def predict_fake_news(text, model, tokenizer):
    model.eval()  # Set the model to evaluation mode
    
    # Ensure text is converted to a single string
    text = str(text)
    
    # Tokenize the text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move inputs to appropriate device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get predicted label
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
    # Map label to text
    label_text = "False" if predicted_label == False else "True"
    
    return label_text

# Example usage

def returnPrediction(text):
    return predict_fake_news(text, model, tokenizer)

