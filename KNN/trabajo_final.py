import os
import re
import nltk
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from PyPDF2 import PdfReader

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar stemmer para español
stemmer = SnowballStemmer("spanish")

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Función para preprocesar texto
def preprocess_text(text):
    # Tokenización
    tokens = nltk.word_tokenize(text)
    # Eliminación de puntuación y números
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Eliminación de stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Obtener lista de archivos PDF en la carpeta
folder_path = r"D:\Claudio\Dataset"
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

# Listas para almacenar texto y etiquetas
texts = []
labels = []

# Leer cada archivo PDF, extraer texto y preprocesarlo
for file in pdf_files:
    file_path = os.path.join(folder_path, file)
    text = extract_text_from_pdf(file_path)
    preprocessed_text = preprocess_text(text)
    texts.append(preprocessed_text)
    # Asignar etiqueta basada en el nombre del archivo
    if "Especificacion de Servicio" in file:
        label = "Especificacion de Servicio"
    elif "Manual de Usuario" in file:
        label = "Manual de Usuario"
    elif "Informe de Pruebas Unitarias" in file:
        label = "Informe de Pruebas Unitarias"
    elif "Informe de Pruebas Integrales" in file:
        label = "Informe de Pruebas Integrales"
    elif "Diccionario de Datos" in file:
        label = "Diccionario de Datos"
    elif "Troubleshooting" in file:
        label = "Troubleshooting"
    elif "Analisis Sonar" in file:
        label = "Analisis Sonar"
    elif "Manual de Configuracion" in file:
        label = "Manual de Configuracion"
    elif "Manual de Instalacion" in file:
        label = "Manual de Instalacion"
    else:
        label = "Otro"
    labels.append(label)
    # Imprimir información sobre el archivo y su etiqueta asignada
    print(f"Archivo: {file}, Etiqueta: {label}")
    # Puedes agregar más condiciones o cambiar las etiquetas según tus necesidades

# Crear un DataFrame para almacenar los datos
data = pd.DataFrame({'Text': texts, 'Label': labels})

# Vectorización de texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, data['Label'], test_size=0.2, random_state=42)

# Entrenar el clasificador KNN
k = 5  # Puedes ajustar este valor según sea necesario
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Predecir etiquetas para datos de prueba
y_pred = knn_classifier.predict(X_test)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Clasificar un nuevo documento de prueba
def classify_new_document(file_path):
    text = extract_text_from_pdf(file_path)
    preprocessed_text = preprocess_text(text)
    new_X = vectorizer.transform([preprocessed_text])
    prediction = knn_classifier.predict(new_X)
    return prediction[0]

# Ejemplo de clasificación de un nuevo documento
new_document_path = r"D:\Claudio\Manual de Usuario - Servicio Migracion de Encriptacion de Huellas - BANBIF.pdf"  # Ruta del documento de prueba
predicted_label = classify_new_document(new_document_path)
print("Predicted label for the new document:", predicted_label)
