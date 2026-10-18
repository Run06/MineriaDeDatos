import json
import os
import sys
import subprocess
import pandas as pd

def cargar_config(ruta="clasificador.json"):
    with open(ruta, "r") as f:
        return json.load(f)

def guardar_config(config, ruta="clasificador.json"):
    with open(ruta, "w") as f:
        json.dump(config, f, indent=2)

def cargar_metricas():
    if not os.path.exists("output/metricas_modelos.csv"):
        print("No se pudo encontrar el archivo output/metricas_modelos.csv")
        sys.exit(-1)
    try:
        df = pd.read_csv("output/metricas_modelos.csv")
        if df.empty:
            print("El fichero está vacio: output/metricas_modelos.csv")
            sys.exit(-1)
        return df
    except Exception as e:
        print(f"Error leyendo el CSV de métricas: {e}")
        sys.exit(-1)

def busqueda_ternaria_k(k_min=1, k_max=500):
    cache = {}

    def evaluar_k(k):
        if k < 2 or k >= 500:  #Límite operativo seguro
            print("El valor de k debe ser mayor a 1 y menor o igual al número de instancias")
            sys.exit(-1)
        if k in cache:
            return cache[k]

        conf = cargar_config("clasificador.json")
        conf["KMeans"]["n_clusters"] = k
        guardar_config(conf, "clasificador.json")

        subprocess.run([
            sys.executable,
            "main.py",
            "-f", "500_Reddit_users_posts_labels.csv",
            "-t", "Post",
            "-l", "Label"
        ])
        df = cargar_metricas()
        score = float(df["Silhouette_Score"].iloc[-1])
        cache[k] = score
        return score

    left = max(2, k_min)
    right = k_max

    best_score = -1.0
    best_k = left

    #Búsqueda ternaria
    while right - left > 2:
        m1 = left + (right - left) // 3
        m2 = right - (right - left) // 3

        score1 = evaluar_k(m1)
        score2 = evaluar_k(m2)

        if score1 > best_score:
            best_score, best_k = score1, m1
        if score2 > best_score:
            best_score, best_k = score2, m2

        if score1 < score2:
            left = m1
        else:
            right = m2

    #Exploración fina final en el rango restante
    for k in range(left, right + 1):
        score = evaluar_k(k)
        if score > best_score:
            best_score, best_k = score, k

    return best_k, best_score

if __name__ == "__main__":
    text_process = ['tf-idf','bow','embeddings']
    distancia = ['Euclidean', 'Cosine']
    modo = ['kmeans','plusplus']
    for t in text_process:
        for d in distancia:
            for m in modo:
                print("Realizando búsqueda ternaria: "+t+" "+d+""+m)
                conf = cargar_config("clasificador.json")
                conf["preprocessing"]["text_process"] = t
                conf["KMeans"]["distancia"] = d
                conf["KMeans"]["modo"] = m
                guardar_config(conf, "clasificador.json")
                k, silhouette = busqueda_ternaria_k(k_min=2, k_max=500)
                print(f"Mejor k: {k} con Silhouette {silhouette:.4f}")
    print("-"*60)
    df = cargar_metricas()
    df_sorted = df.sort_values(by="Silhouette_Score", ascending=False)
    primero = df_sorted.iloc[0]
    modelo = primero["ArchivoPKL"]
    score = primero["Silhouette_Score"]
    print(f"Mejor modelo {modelo} según Silhouette {score:.4f}")