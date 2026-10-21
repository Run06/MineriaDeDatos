# MineriaDeDatos
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

CLASIFICADOR NO SUPERVISADO: K-Means

Marcos Cobo Gutiérrez, Ibai Buján Urien, Keneth Sebastián Campos

Python recomendado: 3.12.

Instalar requirements.txt.
````
pip install -r requirements.txt
````

## Commands
### Main
Dado un csv (-f) y una columna objetivo (-t), ejecuta clasificación NO supervisada (K-Means). Emplea los parámetros del clasficador.json, para tanto el preproceso como el clustering. Asimismo, divide los datos en train test, los datos del test quedan almacenados en un test.csv por si se quiere usar en demo.py.
````
python main --help
````
````
python main.py -f 500_Reddit_users_posts_labels.csv -t Post
````

### Ollama
1. Descargar Ollama: https://ollama.com/download.
2. Descarga el modelo desde una terminal:
    ````
    ollama pull llama3.1
   ````
3. Asegúrate de que Ollama está ejecutándose en segundo plano.
4. Ejecutar main con Ollama:
    ````
    python main.py -f 500_Reddit_users_posts_labels.csv -t Post -o True
    ````

### Demo
La demo carga los centroides y el artefacto de preprocesamiento guardado durante el entrenamiento. Después permite asignar una o varias instancias nuevas a los clusters.

Modo interactivo (Prueba un único mensaje que introduces):
````
python demo.py -m "output/modelos/emotion-mensaje-plusplus-Euclidean-15.pkl"
````
Predicción masiva desde un CSV que tenga una columna Mensaje:
````
python demo.py -m "output/modelos/emotion-mensaje-plusplus-Euclidean-15.pkl" -c test.csv -t Mensaje
````

### Optimizador
Realiza una serie de pruebas (-n) mediante Optuna con distintas combinaciones de parámetros del clasificador. Busca optimizar, el silhouette. Dando como mejor modelo el que obtenga un silhouette máx próximo a uno.
````
python optimizador --help
````
````
python optimizador -n 25 -m 20
````