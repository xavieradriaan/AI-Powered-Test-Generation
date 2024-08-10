import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Cargar los datos tokenizados y las etiquetas
train_encodings, train_labels = torch.load('train_encodings.pt')
test_encodings, test_labels = torch.load('test_encodings.pt')

# Verificar y depurar etiquetas
print("Algunas etiquetas de entrenamiento:", train_labels[:10])
print("Algunas etiquetas de prueba:", test_labels[:10])

# Limpiar etiquetas: Reemplazar NaN con un valor predeterminado (por ejemplo, 0)
train_labels = [0 if np.isnan(label) else int(label) for label in train_labels]
test_labels = [0 if np.isnan(label) else int(label) for label in test_labels]

# Verificar nuevamente las etiquetas después de la limpieza
print("Etiquetas de entrenamiento después de la limpieza:", train_labels[:10])
print("Etiquetas de prueba después de la limpieza:", test_labels[:10])

# Definir una clase de dataset personalizada
class UserStoryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Asegurarse de que las etiquetas sean enteros largos
        return item

    def __len__(self):
        return len(self.labels)

# Crear los objetos de dataset
train_dataset = UserStoryDataset(train_encodings, train_labels)
test_dataset = UserStoryDataset(test_encodings, test_labels)

# Cargar el modelo preentrenado de BERT con el número correcto de clases
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Ajusta num_labels según tus clases

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Entrenar el modelo
trainer.train()