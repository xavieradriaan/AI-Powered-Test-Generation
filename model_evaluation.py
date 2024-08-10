# model_evaluation.py

import torch
from torch.utils.data import Dataset
from transformers import Trainer, BertForSequenceClassification

# Definir una clase de dataset personalizada
class UserStoryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Cargar el modelo entrenado y el dataset de prueba
model = BertForSequenceClassification.from_pretrained('./results')
test_encodings, test_labels = torch.load('test_encodings.pt')

# Crear el objeto de dataset de prueba
test_dataset = UserStoryDataset(test_encodings, test_labels)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset
)

# Evaluar el modelo
eval_result = trainer.evaluate()
print(eval_result)