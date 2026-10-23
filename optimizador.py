import argparse
import json
import os
import sys
import subprocess
import optuna
import pandas as pd
import time
import shutil

from py import ollama_pred
from py.visualizar_instancias import visualizar_tsne

def parse_args():
    """
    - Precondición: Ninguna.
    - Inputs: Argumentos introducidos por la línea de comandos.
    - Outputs: Objeto con los argumentos analizados.
    """
    parser = argparse.ArgumentParser(description="Optimizador de combinaciones del clasificador no supervisado K-Means")
    parser.add_argument("-n", "--num", default=100, type=int, help="Número de combinaciones/iteraciones inteligentes de Optuna")
    parser.add_argument("-m", "--max", default=20, type=int, help="Número máximo de clusters para la prueba")
    parser.add_argument("-f", "--file", default="500_Reddit_users_posts_labels.csv", help="CSV de entrada")
    parser.add_argument("-t", "--target", default="Post", help="Columna de texto")
    parser.add_argument("-l", "--label", default="Label", help="Columna de etiquetas externas")
    parser.add_argument("-o", "--ollama", default=False, help="Aplicar predicción de Ollama sobre los clusters del mejor modelo")
    args = parser.parse_args()
    return args

def cargar_config(ruta="clasificador.json"):
    """
    - Precondición: El archivo JSON de configuración debe existir.
    - Inputs: Ruta del archivo JSON ruta.
    - Outputs: Diccionario con la configuración cargada.
    """
    with open(ruta, "r") as f:
        return json.load(f)

def guardar_config(config, ruta="clasificador.json"):
    """
    - Precondición: El diccionario de configuración debe ser válido para serializar a JSON.
    - Inputs: Diccionario config y ruta del archivo ruta.
    - Outputs: Ninguno (actualiza el archivo físico de configuración).
    """
    with open(ruta, "w") as f:
        json.dump(config, f, indent=2)

def cargar_metricas():
    """
    - Precondición: El archivo output/metricas_modelos.csv debe existir y contener registros.
    - Inputs: Ninguno.
    - Outputs: Un pd.DataFrame con las métricas acumuladas.
    """
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

def objetivo(trial, args_globales):
    """
    - Precondición: Estar ejecutándose dentro de un estudio activo de Optuna.
    - Inputs: Objeto trial de Optuna.
    - Outputs: Retorna un valor float correspondiente al Silhouette Score de la prueba o float("-inf") si ocurre un fallo.
    """
    try:
        t = trial.suggest_categorical("text_process", ["tf-idf", "bow", "embeddings", "doc2vec", "emotion"])
        d = trial.suggest_categorical("distancia", ["Euclidean", "Cosine","Manhattan"])
        m = trial.suggest_categorical("modo", ["kmeans", "plusplus"])
        k = trial.suggest_int("n_clusters", 2, args_globales.max)
        print(f"Probando: {t} {d} {m} k={k}")
        conf = cargar_config("clasificador.json")
        conf["preprocessing"]["text_process"] = t
        conf["KMeans"]["distancia"] = d
        conf["KMeans"]["modo"] = m
        conf["KMeans"]["n_clusters"] = k
        guardar_config(conf, "clasificador.json")
        resultado = subprocess.run([
                sys.executable,
                "main.py",
                "-f", args_globales.file,
                "-t", args_globales.target,
                "-l", args_globales.label,
            ], capture_output=True, text=True)

        if resultado.returncode != 0:
            print(f"Error en main.py: {resultado.stderr}")
            return float("-inf") # Penaliza la prueba si falla la ejecución

        df = cargar_metricas()
        score = float(df["Silhouette_Score"].iloc[-1])
        return score

    except Exception as e:
        print(f"Falló la iteración de Optuna: {e}")
        return float("-inf")

#Si se imprimen los logs entremezclados con los prints de los índices de las pruebas, es normal, es por como se gestionan los buffers de print y logs en la terminal.
#No afecta al funcionamiento. Ejecutar por línea de comandos para ver mejor.
if __name__ == "__main__":
    args = parse_args()
    study = optuna.create_study(study_name="optimización_kmeans",direction="maximize")
    N_TRIALS = args.num #cuántas combinaciones/iteraciones inteligentes (Default 25)
    tiempo_inicio_opt = time.time()
    print("Ejecutando estudio Bayesiano con Optuna...")
    study.optimize(lambda trial: objetivo(trial, args), n_trials=N_TRIALS)
    tiempo_fin_opt = time.time()
    duracion_opt = tiempo_fin_opt - tiempo_inicio_opt
    minutos = int(duracion_opt//60)
    segundos = int(duracion_opt%60)
    print("-"*60)
    print(f"Tiempo total optimización: {minutos} min {segundos} s")
    df = cargar_metricas()
    df_sorted = df.sort_values(by="Silhouette_Score", ascending=False)
    primero = df_sorted.iloc[0]
    modelo = primero["ArchivoPKL"]
    score = primero["Silhouette_Score"]
    print(f"Mejor modelo {modelo} según Silhouette {score:.4f}")

    #Guardar el mejor modelo en una carpeta independiente
    carpeta_mejor = os.path.join("output", "mejor_modelo")
    os.makedirs(carpeta_mejor, exist_ok=True)
    nombre_base = modelo.replace(".pkl","")
    ruta_modelo = os.path.join("output","modelos", modelo)
    ruta_memoria = os.path.join("output","usoDeMemoria", f"{nombre_base}.png")
    ruta_tiempo = os.path.join("output","tiemposEjecucion", f"{nombre_base}_tiempo.png")
    if os.path.exists(ruta_modelo):
        shutil.copy2(ruta_modelo, carpeta_mejor)
    if os.path.exists(ruta_memoria):
        shutil.copy2(ruta_memoria, carpeta_mejor)
    if os.path.exists(ruta_tiempo):
        shutil.copy2(ruta_tiempo, carpeta_mejor)
    print(f"Archivos del mejor modelo guardados en: {carpeta_mejor}")
    print("-"* 60)
    visualizar_tsne(args.file)
    #Ejecutar Ollama
    if args.ollama:
        subprocess.run([
            sys.executable,
            "main.py",
            "-f", args.file,
            "-t", args.target,
            "-l", args.label,
            "-o", "True",
        ])