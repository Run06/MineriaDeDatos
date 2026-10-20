import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from colorama import Fore, Style

from py.preproceso import preproceso
from py.k_means import predict

class ArgsWrapper:
    pass
def cargar_entorno(ruta_pkl,ruta_json='clasificador.json'):
    if not os.path.exists(ruta_json):
        print(f"{Fore.RED}Error: No se encuentra {ruta_json}{Fore.RESET}")
        sys.exit(-1)

    if not os.path.exists(ruta_pkl):
        print(f"{Fore.RED}Error: No se encuentra el modelo {ruta_pkl}{Fore.RESET}")
        sys.exit(-1)

    args = ArgsWrapper()
    with open(ruta_json, 'r') as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    with open(ruta_pkl, 'rb') as f:
        datos_modelo = pickle.load(f)

    centroides = datos_modelo['cluster_centers_']
    distancia_metrica = args.KMeans.get('distancia', 'Euclidean')

    return args, centroides, distancia_metrica


def modo_interactivo(args, centroides, metrica):
    print(f"{Fore.CYAN}\n=== MODO DEMOSTRACIÓN INTERACTIVO ==={Fore.RESET}")
    print("Escribe un mensaje para evaluarlo. Escribe 'salir' para terminar.\n")

    while True:
        texto = input(f"{Fore.YELLOW}Mensaje: {Fore.RESET}")
        if texto.lower().strip() == 'salir':
            break
        if not texto.strip():
            continue

        serie_texto = pd.Series([texto])
        X_nuevo = preproceso(serie_texto, args)

        # Llamada directa a vuestra nueva función en k_means.py
        cluster_asignado = predict(X_nuevo, centroides, metrica)[0]

        print(f"{Fore.GREEN}  └─> Asignado al Clúster: {cluster_asignado}\n{Fore.RESET}")


def modo_csv(ruta_csv, args, centroides, metrica, col_texto):
    print(f"{Fore.CYAN}\n=== MODO PREDICCIÓN MASIVA ({ruta_csv}) ==={Fore.RESET}")
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"{Fore.RED}Archivo CSV no encontrado.{Fore.RESET}")
        sys.exit(-1)

    if col_texto not in df.columns:
        print(f"{Fore.RED}La columna '{col_texto}' no existe en el CSV.{Fore.RESET}")
        sys.exit(-1)

    print(f"Procesando {len(df)} mensajes nuevos...")

    X_nuevos = preproceso(df[col_texto], args)

    # Llamada a vuestra función
    predicciones = predict(X_nuevos, centroides, metrica)

    df['Cluster_Predicho'] = predicciones
    ruta_salida = ruta_csv.replace('.csv', '_predicciones.csv')
    df.to_csv(ruta_salida, index=False)

    print(f"{Fore.GREEN}Predicciones finalizadas. Resultados guardados en: {ruta_salida}{Fore.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de predicción para demostración en vivo.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo .pkl del mejor modelo")
    parser.add_argument("-c", "--csv", default=None, help="Ruta a un archivo .csv opcional")
    parser.add_argument("-t", "--texto", default="Mensaje", help="Nombre de la columna de texto en el CSV")
    sys_args = parser.parse_args()

    print("Cargando el cerebro del modelo...")
    args_modelo, centroides_fijos, metrica = cargar_entorno(sys_args.model)
    print(f"Modelo cargado: {centroides_fijos.shape[0]} clusters fijados usando distancia {metrica}.")

    if sys_args.csv:
        modo_csv(sys_args.csv, args_modelo, centroides_fijos, metrica, sys_args.texto)
    else:
        modo_interactivo(args_modelo, centroides_fijos, metrica)