# MineriaDeDatos
Proyecto grupal para la asignatura de Minería de Datos del grado en Ingeniería Informática de Gestión y Sistemas de Información en la EHU.

CLASIFICADOR NO SUPERVISADO: K-Means

Marcos Cobo Gutiérrez, Ibai Buján Urien, Keneth Sebastián Campos

## Commands
````
python main --help
````
````
python main.py -f 500_Reddit_users_posts_labels.csv -t Post
````

## Ollama
ANTES DE EJECUTAR!!!

Descargar Ollama: https://ollama.com/download, después la versión 3.1 (en terminal del PC: ollama pull llama3.1).

Dejar abierto Ollama en segundo plano.
````
python main.py -f 500_Reddit_users_posts_labels.csv -t Post -o True
````

## Optimizador
Se puede ejecutar tanto desde el botón de play como desde terminal (si se quieren especificar parámetros).
````
python optimizador --help
````
````
python optimizador -n 25 -m 20
````