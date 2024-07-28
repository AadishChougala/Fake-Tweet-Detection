import torch #for deep neural network
import pandas as pd #for datasets
import numpy as np #for arrays
import torch.nn as nn
import torch.optim as optim
import time #retrieving the current time, waiting during code execution
from sklearn.model_selection import train_test_split #machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, AdamW
from torch.cuda import memory_allocated, empty_cache
from sklearn.metrics import classification_report, accuracy_score

class TokenDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load dataset
dataset = pd.read_csv("suspicioustweets.csv")
dataset = dataset.sample(frac=0.015)

# print(dataset.head())

# Initialize BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Move model to device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Split dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(dataset['message'], dataset['label'], test_size=0.3, random_state=42)

# Set batch size and accumulation steps and epochs
batch_size = 10
accumulation_steps = 2
epochs = 3

# Create DataLoader for training data
train_dataset = TokenDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
model.train()
start_time = time.time()
all_labels = []
all_preds = []

for epoch in range(epochs):
    print(f"Starting epoch {epoch+1}...")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Iterate over batches
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Accumulate gradients
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update total loss
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Update total correct predictions
        total_correct += torch.sum(preds == labels).item()
        
        # Update total samples
        total_samples += labels.size(0)
        
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # Perform optimization step after accumulation_steps
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Print average loss and accuracy
            average_loss = total_loss / accumulation_steps
            accuracy = total_correct / total_samples
            print(f"Batch {i+1}/{len(train_loader)}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            # Clear CUDA cache to free up memory
            empty_cache()

    # Compute metrics at the end of each epoch
    print("Computing metrics...")
    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])
    print(report)

# Calculate total training time
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), "bert_fake_news_model.pth")

def predict_fake_tweet(text, model, tokenizer):
    model.eval()  # Set the model to evaluation mode
    
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
    label_text = "Fake" if predicted_label == 1 else "Real"
    
    return label_text

# Example usage
# text_to_check = "Please click on this link to get twitter blue tick verficiation at just 99 dollars"
# predicted_label = predict_fake_tweet(text_to_check, model, tokenizer)
# print(f"The predicted label for the text '{text_to_check}' is: {predicted_label}")
