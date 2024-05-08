import os
import shutil
import nltk
import spacy
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo en español de spaCy
nlp = spacy.load('es_core_news_sm')


# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# Función para preprocesar texto
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]  # solo palabras alfabéticas
    stop_words = nlp.Defaults.stop_words
    tokens = [word for word in tokens if word not in stop_words]  # eliminar stopwords
    return ' '.join(tokens)


# Función para asignar etiquetas basadas en el contenido analizado
def assign_label_from_content(text):
    lower_text = text.lower()
    if "manual usuario" in lower_text:
        return "Manual de Usuario"
    if "manual instalacion" in lower_text:
        return "Manual de Instalacion"
    return "Indefinido"


folder_path = r"E:\UPC\2024-1\Analitica_de_Datos_y_Sistemas_Predictivos\TF\DataSet"
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

texts = []
labels = []

# Leer cada archivo PDF, extraer texto y preprocesarlo
for file in pdf_files:
    file_path = os.path.join(folder_path, file)
    text = extract_text_from_pdf(file_path)
    preprocessed_text = preprocess_text(text)
    label = assign_label_from_content(preprocessed_text)
    print(f"Etiqueta: {label}")
    texts.append(preprocessed_text)
    labels.append(label)

data = pd.DataFrame({'Text': texts, 'Label': labels})

# Vectorización de texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, data['Label'], test_size=0.2, random_state=42)

# Entrenar el clasificador, usando RandomForest
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Predecir etiquetas para datos de prueba
y_pred = classifier.predict(X_test)

# Calcular precisión y otras métricas
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear carpetas basadas en las etiquetas y mover los archivos PDF
for label in set(labels):
    label_folder = os.path.join(folder_path, label)
    os.makedirs(label_folder, exist_ok=True)

for file, label in zip(pdf_files, labels):
    src_path = os.path.join(folder_path, file)
    dst_path = os.path.join(folder_path, label, file)
    shutil.move(src_path, dst_path)

print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)
print("Confusion Matrix:\n", conf_matrix)
