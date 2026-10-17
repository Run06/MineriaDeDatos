import json
import sys
import subprocess
import pandas as pd

def cargar_config(ruta="clasificador.json"):
  with open(ruta, "r") as f:
    return json.load(f)

def guardar_config(config, ruta="clasificador.json"):
  with open(ruta, "w") as f:
    json.dump(config, f, indent=2)

if __name__ == "__main__": #En proceso, todavía falta busqueda ternaria
    text_process = ['tf-idf','bow','embeddings']
    distancia = ['Euclidean', 'Cosine']
    modo = ['kmeans','plusplus']
    args = None
    try:
        raw_data = pd.read_csv('500_Reddit_users_posts_labels.csv')
    except FileNotFoundError:
        print(f"Error: El archivo 500_Reddit_users_posts_labels.csv no existe o la ruta es incorrecta.")
        sys.exit(-1)

    for t in text_process:
        for d in distancia:
            for m in modo:
                print("Procesando: "+t+" "+d+""+m)
                conf = cargar_config("clasificador.json")
                conf["preprocessing"]["text_process"] = t
                conf["KMeans"]["distancia"] = d
                conf["KMeans"]["modo"] = m
                guardar_config(conf, "clasificador.json")
                subprocess.run([
                    sys.executable,
                    "main.py",
                    "-f", "500_Reddit_users_posts_labels.csv",
                    "-t", "Post"
                ])