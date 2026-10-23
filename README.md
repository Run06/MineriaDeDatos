# MineriaDeDatos
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

CLASIFICADOR NO SUPERVISADO: K-Means (creado desde cero)

Marcos Cobo Gutiérrez, Ibai Buján Urien, Keneth Sebastián Campos

Python recomendado: 3.12.

Instalar requirements.txt.
````
pip install -r requirements.txt
````

Datos - Reddit C-SSRS Suicide Dataset: https://zenodo.org/records/2667859

## Commands
### Main
Dado un csv (-f) y una columna objetivo (-t), ejecuta clasificación NO supervisada (K-Means). Emplea los parámetros del clasficador.json, para tanto el preproceso como el clustering. Asimismo, divide los datos en train test, los datos del test quedan almacenados en un test.csv por si se quiere usar en demo.py.

Precondición: el csv de datos debe contener una columna de mensajes (target, -t) y una columna de labels (labels, -l, default="Label") para el cálculo de las métricas de evaluación externas.
````
python main --help
````
````
python main.py -f 500_Reddit_users_posts_labels.csv -t Post -l Label
````

### Ollama
Mediante los centroides del modelo generado y las 5 frases más cercanas a cada centroide, por cada cluster concluye una etiqueta que lo resuma y un breve resumen de por qué.
A veces falla con textos muy explícitos, ya que la Ollama se niega a responder; sin embargo, generalmente adjunta una justificación de porque no puede dar una respuesta.
1. Descargar Ollama: https://ollama.com/download.
2. Descarga el modelo desde una terminal:
    ````
    ollama pull llama3.1
   ````
3. Asegúrate de que Ollama está ejecutándose en segundo plano.
4. Ejecutar Ollama:

A - Ejecutar main.py

   ````
   python main.py -f 500_Reddit_users_posts_labels.csv -t Post -l Label -o True
   ````
B - Ejecutar optimizador.py
   ````
   python optimizador.py -n 100 -m 20 -o True
   ````

### Demo
La demo carga los centroides y el artefacto de preprocesamiento guardado durante el entrenamiento. Después permite asignar una o varias instancias nuevas a los clusters.
Asimismo imprime las 5 palabras del dataset original (train.csv) más cercanas a cada centroide.

Precondición: debe haberse ejecutado main.py antes y tener, un modelo entrenado, train.csv, test.csv.
````
python demo.py --help
````
Modo interactivo (Prueba un único mensaje que introduces):
````
python demo.py -i True -m "output/modelos/emotion-mensaje-plusplus-Euclidean-15.pkl" -r train.csv -t Mensaje
````
Predicción masiva desde un CSV que tenga una columna Mensaje:
````
python demo.py -m "output/modelos/emotion-mensaje-plusplus-Euclidean-15.pkl" -c test.csv -t Mensaje -r train.csv
````

### Optimizador
Realiza una serie de pruebas (-n) mediante Optuna con distintas combinaciones de parámetros del clasificador. Busca optimizar, el silhouette. Dando como mejor modelo el que obtenga un silhouette máx próximo a uno. Por último, grafica los datos del mejor modelo mediante tSNE.

Guarda todo en la carpeta output/mejor_modelo
````
python optimizador.py --help
````
````
python optimizador.py -n 100 -m 20
````

## Configuración Hiperparámetros
### clasificador.json
````
{
  "preprocessing": {
    "text_process": "emotion",               #Tipo de vectorización ["bow", "tf-idf", "embeddings", "doc2vec", "emotion"].
    "unidad": "mensaje"                      #Divide los mensajes en nuevas filas si tienen la estructura [msj1],[msj2],...
  },
  "bow": {
    "max_features": 10000,                   #Limite máximo del vocabulario.
    "ngram_range": [                         #Tamaño de las agrupaciones de palabras (unigrama, bigrama, trigrama,...).
      1,
      2
    ]
  },
  "tf-idf": {
    "max_features": 5000,                    #Limite máximo del vocabulario.
    "ngram_range": [                         #Tamaño de las agrupaciones de palabras (unigrama, bigrama, trigrama,...).
      1,
      3
    ],
    "min_df": 2,                             #Descarta palabras o n-gramas que aparecen en menos de 2 documentos.
    "max_df": 0.9,                           #Descarta palabras o n-gramas que aparecen en más del 90% de los documentos.
    "sublinear_tf": true                     #Escala logarítmica a la frecuencia (Evita que un doc que usa mucho la misma palabra opaque al resto).
  },
  "KMeans": {
    "n_clusters": 15,                        #Número de grupos en los que el algoritmo intentará dividir todos los mensajes.
    "random_state": 42,                      #Una semilla matemática para asegurar que si se vuelve a ejecutar, el resultado sea exactamente el mismo.
    "distancia": "Euclidean",                #Mide la similitud o distancia geométrica entre los puntos usando ["Euclidean", "Cosine","Manhattan"].
    "modo": "plusplus",                      #Utiliza el método K-Means o K-Means++ ["kmeans", "plusplus"].
    "max_iter": 300,                         #Número máximo de iteraciones que hará el algoritmo para acomodar los grupos antes de detenerse.
    "tol": 0.0001                            #Criterio de parada; si en una iteración los grupos casi no cambian, el algoritmo finaliza.
  }
}
````
