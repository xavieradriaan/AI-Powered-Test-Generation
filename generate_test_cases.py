import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Cargar el modelo entrenado y el tokenizador
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Poner el modelo en modo de evaluación
model.eval()

# Mapeo de etiquetas
label_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}  # Ajusta según tus etiquetas

def generate_test_case(user_story):
    inputs = tokenizer(user_story, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        outputs = model(**inputs)
    predicted_test_case = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[predicted_test_case]

# Ejemplo de uso
user_story = "As a user, I want to reset my password so that I can recover my account."
test_case = generate_test_case(user_story)
print(test_case)