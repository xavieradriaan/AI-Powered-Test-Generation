# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import warnings

# Suprimir advertencias
warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar el dataset
df = pd.read_csv('/home/adrian/Descargas/archive/sentimentdataset.csv')

# Mostrar las primeras filas del dataset
print(df.head())

# Tokenizar y preprocesar los datos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(data):
    return tokenizer(data, padding=True, truncation=True, return_tensors="pt")

# Dividir los datos en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizar las historias de usuario y los casos de prueba
train_encodings = tokenize_data(train_df['Text'].tolist())
test_encodings = tokenize_data(test_df['Text'].tolist())

# Convertir las etiquetas a numéricas
label_mapping = {'Positive': 1, 'Negative': 0, 'Neutral': 2}  # Ajusta según tus etiquetas
train_labels = train_df['Sentiment'].map(label_mapping).tolist()
test_labels = test_df['Sentiment'].map(label_mapping).tolist()

# Guardar los datos tokenizados y las etiquetas
torch.save((train_encodings, train_labels), 'train_encodings.pt')
torch.save((test_encodings, test_labels), 'test_encodings.pt')