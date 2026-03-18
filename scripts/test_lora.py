import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EsmForSequenceClassification, EsmTokenizer, Trainer, TrainingArguments
import pickle

class LLMDataset(Dataset):
    #def __init__(self, sequences, labels, tokenizer, max_length=128):
    def __init__(self, dataset_path, tokenizer, max_length=128):
        with open(dataset_path, 'rb') as f:
            self.dataSet_df = pickle.load(f)
        self.sequences = self.dataSet_df['sequence'].values
        self.labels = self.dataSet_df['positive'].astype(int).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }

id2label = {0: "Not RNA binding", 1: "RNA binding"}
label2id = {v: k for k, v in id2label.items()}
len(id2label)
tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
model = EsmForSequenceClassification.from_pretrained("facebook/esm1b_t33_650M_UR50S", num_labels=len(id2label), label2id=label2id, id2label=id2label)


repository = Path(os.getenv("REPOSITORY", "/path/to/RBP_IG_storage"))
dataset_path = repository / "data" / "data_sets" / "bressin19_human_pre-training.pkl"
dataset = LLMDataset(dataset_path, tokenizer)

len_val = int(len(dataset)*0.2)
len_train = len(dataset)-len_val
train_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[len_train, len_val])

from transformers import Trainer, TrainingArguments

torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset  # Use the test set as the evaluation dataset
)

trainer.train()
