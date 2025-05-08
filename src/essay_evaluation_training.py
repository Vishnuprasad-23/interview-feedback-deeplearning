import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define the EssayModel class
class EssayModel(torch.nn.Module):
    def __init__(self, base_model):
        super(EssayModel, self).__init__()
        self.base = base_model
        self.coherence = torch.nn.Linear(768, 1)  # 768 is RoBERTa hidden size, 1 for coherence score
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None, coherence_labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1][:, 0, :]  # CLS token embedding
        coherence_score = self.sigmoid(self.coherence(hidden)) * 100  # Scale to 0-100
        
        loss = None
        if labels is not None and coherence_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            sentiment_loss = loss_fct(logits, labels)
            coherence_loss = torch.nn.MSELoss()(coherence_score.squeeze(), coherence_labels)
            loss = sentiment_loss + 0.5 * coherence_loss
            
        return {
            'loss': loss,
            'logits': logits,
            'coherence_score': coherence_score
        }

# Custom dataset class to handle sentiment and coherence
class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, coherence_scores):
        self.encodings = encodings
        self.labels = labels
        self.coherence_scores = coherence_scores

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['coherence_labels'] = torch.tensor(self.coherence_scores[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Custom data collator
def data_collator(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    coherence_labels = torch.stack([item['coherence_labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'coherence_labels': coherence_labels
    }

# Load dataset (using SST-2 for sentiment and synthetic coherence scores)
dataset = load_dataset('sst2')
train_data = dataset['train']
val_data = dataset['validation']

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
base_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2, output_hidden_states=True)
model = EssayModel(base_model)

# Tokenize the dataset
train_encodings = tokenizer([x['sentence'] for x in train_data], truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer([x['sentence'] for x in val_data], truncation=True, padding='max_length', max_length=128)

# Extract labels and generate synthetic coherence scores
train_labels = [x['label'] for x in train_data]
val_labels = [x['label'] for x in val_data]
train_coherence = [random.uniform(0, 100) for _ in range(len(train_labels))]  # Synthetic coherence scores
val_coherence = [random.uniform(0, 100) for _ in range(len(val_labels))]

# Create datasets
train_dataset = EssayDataset(train_encodings, train_labels, train_coherence)
val_dataset = EssayDataset(val_encodings, val_labels, val_coherence)

# Define training arguments
training_args = TrainingArguments(
    output_dir="../models/essay_model_roberta",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define compute_metrics function
def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()
    return {
        "accuracy": accuracy,
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the entire model
torch.save(model.state_dict(), "../models/essay_model_roberta/pytorch_model.bin")
tokenizer.save_pretrained("../models/essay_model_roberta")
print("Model and tokenizer saved successfully")