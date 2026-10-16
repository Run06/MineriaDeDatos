import argparse
import os
import json
import numpy as np
import pandas as pd
from colorama import Fore

from preproceso import preproceso
from k_means import kmeans
from evaluacion import evaluar
from ollama_pred import predecir_clusters

#CLASIFICADOR NO SUPERVISADO: K-Means
#Marcos Cobo Gutiérrez, Ibai Buján Urien, Keneth Sebastián Campos

def parse_args():
    parser = argparse.ArgumentParser(description="Clasificación no supervisada K-Means")
    parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    #parser.add_argument("-c", "--cpu", default=-1, type=int)
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

if __name__ == "__main__":
    args = parse_args()
    #Cargar csv
    raw_data = pd.read_csv(args.file)

    #División train test, 0.7
    p_train = 0.7
    np.random.seed(42)
    raw_data['is_train'] = np.random.uniform(0, 1, len(raw_data)) <= p_train
    train, test = raw_data[raw_data['is_train'] == True], raw_data[raw_data['is_train'] == False]

    print(Fore.MAGENTA + f"Preprocesando datos del train..." + Fore.RESET)
    train_prep = preproceso(train[args.target], args)
    print(Fore.GREEN + f"Datos preprocesados!" + Fore.RESET)

    print("-"*60)

    #Preparación de la matriz para K-means
    if hasattr(train_prep, 'toarray'):
        X_train = train_prep.toarray()
    elif isinstance(train_prep, pd.DataFrame):
        X_train = train_prep.values
    else:
        X_train = np.array(train_prep)

    print(Fore.MAGENTA + f"Matriz preparada con forma: {X_train.shape}" + Fore.RESET)

    #Ejecución K-means
    print(Fore.MAGENTA + f"Ejecutando K-Means..." + Fore.RESET)
    clusters, centroides, nombre_archivo = kmeans(X_train, **args.KMeans)
    nombre_archivo_final = args.preprocessing.get("text_process")+'-'+nombre_archivo
    if os.path.exists(nombre_archivo_final):
        os.remove(nombre_archivo_final)
    os.rename(nombre_archivo, nombre_archivo_final)
    print(Fore.GREEN + f"Modelo guardado en '{nombre_archivo_final}'" + Fore.RESET)
    print(f"Centroides finales calculados: {centroides.shape[0]}")

    print("-"*60)

    #Evaluación
    y_train = train[args.label].to_numpy()
    evaluar(X_train, y_train, nombre_archivo_final)

    print("-"*60)

    #Predicción Ollama
    #ANTES DE EJECUTAR!!!
    #Descargar Ollama: https://ollama.com/download, después la versión 3.1 (en terminal del PC: ollama pull llama3.1)
    #Dejar abierto Ollama en segundo plano
    predecir_clusters(nombre_archivo_final, X_train, train[args.target], args.preprocessing.get("text_process"))