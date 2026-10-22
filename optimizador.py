import argparse
import json
import os
import sys
import subprocess

import optuna
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Obtimizador de combinaciones del clasificador no supervisado K-Means")
    parser.add_argument("-n", "--num", default=25, type=int, help="Número de combinaciones/iteraciones inteligentes de Optuna")
    parser.add_argument("-m", "--max", default=20, type=int, help="Número máximo de clusters para la prueba")
    args = parser.parse_args()
    return args

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

def objetivo(trial):
    t = trial.suggest_categorical("text_process", ["tf-idf", "bow", "embeddings", "doc2vec", "emotion"])
    d = trial.suggest_categorical("distancia", ["Euclidean", "Cosine","Manhattan"])
    m = trial.suggest_categorical("modo", ["kmeans", "plusplus"])
    k = trial.suggest_int("n_clusters", 2, args.max)
    print(f"Probando: {t} {d} {m} k={k}")
    conf = cargar_config("clasificador.json")
    conf["preprocessing"]["text_process"] = t
    conf["KMeans"]["distancia"] = d
    conf["KMeans"]["modo"] = m
    conf["KMeans"]["n_clusters"] = k
    guardar_config(conf, "clasificador.json")
    subprocess.run([
            sys.executable,
            "main.py",
            "-f", "500_Reddit_users_posts_labels.csv",
            "-t", "Post",
            "-l", "Label",
        ])
    df = cargar_metricas()
    score = float(df["Silhouette_Score"].iloc[-1])
    return score

if __name__ == "__main__":
    args = parse_args()
    study = optuna.create_study(direction="maximize")
    N_TRIALS = args.num #cuántas combinaciones/iteraciones inteligentes (Default 25)
    print("Ejecutando estudio Bayesiano con Optuna...")
    study.optimize(objetivo, n_trials=N_TRIALS)
    print("-"*60)
    df = cargar_metricas()
    df_sorted = df.sort_values(by="Silhouette_Score", ascending=False)
    primero = df_sorted.iloc[0]
    modelo = primero["ArchivoPKL"]
    score = primero["Silhouette_Score"]
    print(f"Mejor modelo {modelo} según Silhouette {score:.4f}")