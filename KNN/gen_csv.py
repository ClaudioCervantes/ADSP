import csv
import os
import pdfplumber


def extract_text_from_pdf(file_path):
    """Extrae texto de un archivo PDF usando pdfplumber y elimina los saltos de línea."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            # Reemplaza los saltos de línea con un espacio
            cleaned_text = page_text.replace('\n', ' ')
            text += cleaned_text
    return text


def process_folders(base_folder):
    data = []
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):  # Verifica que es un directorio
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.pdf'):
                    file_path = os.path.join(folder_path, file_name)
                    text = extract_text_from_pdf(file_path)
                    # Usa el nombre de la carpeta como etiqueta
                    label = folder_name
                    data.append([text, label])
    return data


base_folder = r'E:\UPC\2024-1\Analitica_de_Datos_y_Sistemas_Predictivos\TF\DataSet'  # Ruta a la carpeta principal
output_csv = 'data_original.csv'

# Procesar todas las carpetas y recopilar los datos
all_data = process_folders(base_folder)

# Escribir los datos en un archivo CSV usando punto y coma como delimitador
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')  # Especificar ';' como delimitador
    writer.writerow(['Text', 'Label'])  # Escribir cabecera
    writer.writerows(all_data)
