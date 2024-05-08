### DataSet

En la ruta /DataSet se deben colocar los PDF que serviran de entrenamiento

### Documento de Prueba

El documento borrame.txt que se encuentra en la raiz del proyecto debe ser reemplazado por un PDF que sera usado como prueba del modelo

### Dependencias

Instala las dependencias necesarias ejecutando el siguiente comando:

```bash
pip install pandas 
pip install scikit-learn 
pip install PyPDF2 
pip install nltk
pip install pdfplumber
pip install spacy
python -m spacy download es_core_news_sm
```
## Instrucciones

1. Ejecutar el archivo gen_csv.py para generar el documento data.csv a partir de los datos en la carpeta DataSet
2. Ejecutar el archivo tf_v3.py para el entrenamiento del modelo y la clasificacion de los documentos de la carpeta Test

