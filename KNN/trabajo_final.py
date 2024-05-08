# import os
# import shutil
# import nltk
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from PyPDF2 import PdfReader
#
# # Descargar recursos de NLTK
# nltk.download('punkt')
# nltk.download('stopwords')
#
# # Inicializar stemmer para español
# stemmer = SnowballStemmer("spanish")
#
#
# # Función para extraer texto de un archivo PDF
# def extract_text_from_pdf(file_path):
#     text = ""
#     with open(file_path, 'rb') as file:
#         pdf_reader = PdfReader(file)
#         for page_num in range(len(pdf_reader.pages)):
#             page_text = pdf_reader.pages[page_num].extract_text()
#             if page_text:
#                 text += page_text
#     return text
#
#
# # Función para preprocesar texto
# def preprocess_text(text):
#     #print(f"Texto procesado: {text[:50]}...")
#     tokens = nltk.word_tokenize(text)
#     tokens = [word.lower() for word in tokens if word.isalpha()]
#     stop_words = set(stopwords.words('spanish'))
#     tokens = [word for word in tokens if word not in stop_words]
#     tokens = [stemmer.stem(word) for word in tokens]
#     return ' '.join(tokens)
#
#
# # Función para asignar etiquetas basadas en el contenido analizado
# def assign_label_from_content(text):
#     lower_text = text.lower()
#     print(f"Texto procesado: {lower_text[:550]}...")  # Imprime los primeros 50 caracteres del texto procesado
#     if "document especificaci on serv ici" in lower_text:
#         print("Etiqueta asignada: ocumento de Especificacion de Servicio")
#         return "Documento de Especificacion de Servicio"
#     if "mu" in lower_text:
#         print("Etiqueta asignada: Manual de Usuario")
#         return "Manual de Usuario"
#     if "mi" in lower_text:
#         print("Etiqueta asignada: Manual de Instalación")
#         return "Manual de Instalación"
#     print("Etiqueta asignada: Indefinido")
#     return "Indefinido"
#
#
# folder_path = r"E:\UPC\2024-1\Analitica_de_Datos_y_Sistemas_Predictivos\TF\DataSet"
# pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
#
# texts = []
# labels = []
#
# # Leer cada archivo PDF, extraer texto y preprocesarlo
# for file in pdf_files:
#     file_path = os.path.join(folder_path, file)
#     text = extract_text_from_pdf(file_path)
#     preprocessed_text = preprocess_text(text)
#     label = assign_label_from_content(preprocessed_text)
#     texts.append(preprocessed_text)
#     labels.append(label)
#
# data = pd.DataFrame({'Text': texts, 'Label': labels})
#
# # Vectorización de texto usando TF-IDF
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(data['Text'])
#
# # Dividir datos en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, data['Label'], test_size=0.2, random_state=42)
#
# # Entrenar el clasificador, usando RandomForest
# classifier = RandomForestClassifier(n_estimators=100)
# classifier.fit(X_train, y_train)
#
# # Predecir etiquetas para datos de prueba
# y_pred = classifier.predict(X_test)
#
# # Calcular precisión y otras métricas
# accuracy = accuracy_score(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
#
# # Crear carpetas basadas en las etiquetas y mover los archivos PDF
# for label in set(labels):
#     label_folder = os.path.join(folder_path, label)
#     os.makedirs(label_folder, exist_ok=True)
#
# for file, label in zip(pdf_files, labels):
#     src_path = os.path.join(folder_path, file)
#     dst_path = os.path.join(folder_path, label, file)
#     shutil.move(src_path, dst_path)
#
# print("Accuracy:", accuracy)
# print("Classification Report:\n", class_report)
# print("Confusion Matrix:\n", conf_matrix)
