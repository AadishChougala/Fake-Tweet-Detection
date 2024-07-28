import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from multiprocessing import Pool

# Define your TokenDataset class and predict_fake_tweet function here...
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


def train_model_batch(model, counter, train_loader, optimizer, batch, device):
    model.train()

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    total_loss = loss.item()

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)

    total_correct = torch.sum(preds == labels).item()
    total_samples = labels.size(0)

    all_labels = labels.cpu().numpy()
    all_preds = preds.cpu().numpy()

    batch_loss = loss.item()
    batch_accuracy = torch.sum(preds == labels).item() / labels.size(0)
    print(f"Batch {counter+1}/{len(train_loader)}, Batch Loss: {batch_loss:.4f}, Batch Accuracy: {batch_accuracy:.4f}")

    return total_loss, total_correct, total_samples, all_labels, all_preds

if __name__ == "__main__":
    dataset = pd.read_csv("tweetsDetectionDataset.csv")
    # dataset = dataset.sample(frac=0.05)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    train_texts, test_texts, train_labels, test_labels = train_test_split(dataset['message'], dataset['label'], test_size=0.3, random_state=42)

    batch_size = 10
    train_dataset = TokenDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    epochs = 15
    accumulation_steps = 2

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}...")
        pool = Pool(processes=4)  # Adjust the number of processes as needed
        counter = 0
        results = []
        for i, batch in enumerate(train_loader):
            result = pool.apply_async(train_model_batch, args=(model, counter, train_loader, optimizer, batch, device))
            counter += 1
            results.append(result)

        pool.close()
        print()
        pool.join()
        print()

        all_labels = []
        all_preds = []
        for result in results:
            result_values = result.get()
            all_labels.extend(result_values[3])
            all_preds.extend(result_values[4])

        total_loss = sum([result.get()[0] for result in results])
        total_correct = sum([result.get()[1] for result in results])
        total_samples = sum([result.get()[2] for result in results])
        # all_labels = sum([result.get()[3] for result in results], [])
        # all_preds = sum([result.get()[4] for result in results], [])

        average_loss = total_loss / (accumulation_steps * len(train_loader))
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    report = classification_report(all_labels, all_preds, target_names=["Real", "Fake"])
    print(report)

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    torch.save(model.state_dict(), "newSetup_dataset.pth")
