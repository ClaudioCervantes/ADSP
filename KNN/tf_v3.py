import os
import pdfplumber
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar el modelo en español de spaCy para procesamiento de texto
nlp = spacy.load('es_core_news_sm')


def extract_text_from_pdf(file_path):
    """Extrae texto de un archivo PDF usando pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""  # Asegúrate de manejar None
            text += page_text + " "  # Agregar espacio para evitar unión de palabras
    return text


def preprocess_text(text):
    """Preprocesa el texto utilizando spaCy para tokenización y eliminación de stopwords."""
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)


# Carga de datos desde CSV y preprocesamiento
data_path = "data.csv"
data = pd.read_csv(data_path, delimiter=';')
data['Processed_Text'] = data['Text'].apply(preprocess_text)

# Vectorización de texto
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Processed_Text'])
y = data['Label']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo RandomForest
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Procesar y predecir con nuevos documentos PDF
test_folder_path = "../Test"
test_files = [f for f in os.listdir(test_folder_path) if f.endswith('.pdf')]
test_texts = [preprocess_text(extract_text_from_pdf(os.path.join(test_folder_path, f))) for f in test_files]

X_new_test = vectorizer.transform(test_texts)
new_y_pred = classifier.predict(X_new_test)

# Evaluar el modelo en el conjunto de prueba
y_pred = classifier.predict(X_test)
print("Evaluación con el conjunto de prueba dividido del CSV:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nPredicciones para nuevos documentos PDF:")
for file_name, prediction in zip(test_files, new_y_pred):
    print(f"Archivo: {file_name}, Predicción: {prediction}")






