import argparse
import os
import json
import sys
import numpy as np
import pandas as pd
from colorama import Fore
import psutil

from py.preproceso import preproceso
from py.k_means import kmeans
from py.evaluacion import evaluar
from py.ollama_pred import predecir_clusters

#CLASIFICADOR NO SUPERVISADO: K-Means
#Marcos Cobo Gutiérrez, Ibai Buján Urien, Keneth Sebastián Campos

def parse_args():
    parser = argparse.ArgumentParser(description="Clasificación no supervisada K-Means")
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    parser.add_argument("-o", "--ollama", default=False, help="Aplicar predicción de Ollama sobre los clusters")
    parser.add_argument("-t", "--target", required=True, help="Columna de la tabla a clasificar")
    parser.add_argument("-l", "--label", default='Label', help="Columna del Label para métricas externas")
    parser.add_argument("-s", "--sample", type=float, default=1.0, help="Porcentaje de datos a usar (0.0 a 1.0)")

    args = parser.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args

def imprimir_memoria(fase):
    #Memoria del script
    proceso = psutil.Process(os.getpid())
    mem_proceso_mb = proceso.memory_info().rss / (1024 * 1024)

    #Memoria completa del ordenador
    mem_sistema = psutil.virtual_memory()
    mem_sistema_gb = mem_sistema.used / (1024 * 1024 * 1024)

    print(Fore.CYAN + f"[{fase}]" + Fore.RESET)
    print(f"  ├─ RAM del script Python: {mem_proceso_mb:.2f} MB")
    print(f"  └─ RAM total del sistema: {mem_sistema_gb:.2f} GB ({mem_sistema.percent}%)\n")

if __name__ == "__main__":
    args = parse_args()
    imprimir_memoria("Inicio del programa")
    #Cargar csv
    try:
        raw_data = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: El archivo '{args.file}' no existe o la ruta es incorrecta.")
        sys.exit(-1)

    #División train test, 0.7
    p_train = 0.7
    np.random.seed(42)
    raw_data['is_train'] = np.random.uniform(0, 1, len(raw_data)) <= p_train
    train, test = raw_data[raw_data['is_train'] == True], raw_data[raw_data['is_train'] == False]

    imprimir_memoria("Carga CSV")

    print(Fore.MAGENTA + f"Preprocesando datos del train..." + Fore.RESET)
    train_prep = preproceso(train[args.target], args)
    print(Fore.GREEN + f"Datos preprocesados!" + Fore.RESET)
    imprimir_memoria("Preprocesado")
    print("-"*60)

    X_train = train_prep

    #Ejecución K-means
    print(Fore.MAGENTA + f"Ejecutando K-Means..." + Fore.RESET)
    clusters, centroides, nombre_archivo = kmeans(X_train, **args.KMeans)
    nombre_archivo_final = args.preprocessing.get("text_process")+'-'+nombre_archivo
    if os.path.exists('output/'+nombre_archivo_final):
        os.remove('output/'+nombre_archivo_final)
    os.rename('output/'+nombre_archivo, 'output/'+nombre_archivo_final)
    print(Fore.GREEN + f"Modelo guardado en output/'{nombre_archivo_final}'" + Fore.RESET)
    print(f"Centroides finales calculados: {centroides.shape[0]}")
    imprimir_memoria("K-Means")
    print("-"*60)

    #Evaluación
    y_train = train[args.label].to_numpy()
    evaluar(X_train, y_train, nombre_archivo_final)
    imprimir_memoria("Evaluación")
    print("-"*60)

    #Predicción Ollama
    #ANTES DE EJECUTAR!!!
    #Descargar Ollama: https://ollama.com/download, después la versión 3.1 (en terminal del PC: ollama pull llama3.1)
    #Dejar abierto Ollama en segundo plano
    if args.ollama:
        predecir_clusters(nombre_archivo_final, X_train, train[args.target], args.preprocessing.get("text_process"))
        imprimir_memoria("Ollama")