import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from colorama import Fore, Style

from py.preproceso import preproceso
from py.k_means import predict, calcular_distancias

class ArgsWrapper:
    pass
def cargar_entorno(ruta_pkl,ruta_json='clasificador.json'):
    """
    - Precondición: Deben existir los archivos de configuración JSON y el modelo .pkl.
    - Inputs: Ruta del archivo pickle (ruta_pkl) y ruta del archivo JSON (ruta_json).
    - Outputs: Tupla con argumentos, centroides, métrica de distancia y vectorizador del modelo.
    - Postcondición: Carga y retorna los parámetros de configuración, centroides, la métrica empleada y el vectorizador entrenado necesarios para realizar la predicción.
    """
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
    distancia_metrica = datos_modelo.get('distancia', 'Euclidean')
    vectorizador_modelo = datos_modelo.get('vectorizador',None)
    metodo_original = datos_modelo.get('text_process','none')
    if hasattr(args,'preprocessing'):
        args.preprocessing['text_process'] = metodo_original

    return args, centroides, distancia_metrica, vectorizador_modelo


def modo_interactivo(args, centroides, metrica,vectorizador_modelo, textos_ref, dist_ref, pred_ref):
    """
    - Precondición: El entorno y el modelo deben estar cargados correctamente.
    - Inputs: Objeto args, matriz centroides, string metrica, vectorizador_modelo y datos de referencia del train (textos_ref, dist_ref, pred_ref).
    - Outputs: Ninguno (ejecuta un bucle interactivo evaluando mensajes por consola y muestra las 5 instancias más cercanas de su clúster asignado).
    - Postcondición: Permite la interacción por consola en bucle hasta recibir la orden de parada, mostrando la asignación de clúster e instancias cercanas del conjunto de entrenamiento.
    """
    print(f"{Fore.CYAN}\n=== MODO DEMOSTRACIÓN INTERACTIVO ==={Fore.RESET}")
    print("Escribe un mensaje para evaluarlo. Escribe 'salir' para terminar.\n")

    while True:
        texto = input(f"{Fore.YELLOW}Mensaje: {Fore.RESET}")
        if texto.lower().strip() == 'salir':
            break
        if not texto.strip():
            continue

        serie_texto = pd.Series([texto])
        X_nuevo = preproceso(serie_texto, args,fase="predict",vectorizador_entrenado=vectorizador_modelo)

        # Llamada directa a vuestra nueva función en k_means.py
        cluster_asignado = predict(X_nuevo, centroides, metrica)[0]

        print(f"{Fore.GREEN}  |--> Asignado al Clúster: {cluster_asignado}{Fore.RESET}")

        #Mostrar las 5 instancias más cercanas del train para este clúster
        if textos_ref is not None and dist_ref is not None and pred_ref is not None:
            indices_cluster = np.flatnonzero(pred_ref == cluster_asignado)
            if len(indices_cluster) > 0:
                top_indices = indices_cluster[np.argsort(dist_ref[indices_cluster, cluster_asignado])[:5]]
                print(f"{Fore.CYAN}  |--> 5 instancias más cercanas del clúster (Train):{Fore.RESET}")
                for idx in top_indices:
                    print(f"      |-- [Distancia: {dist_ref[idx, cluster_asignado]:.4f}] {textos_ref.iloc[idx]}")
        print()


def modo_csv(ruta_csv, args, centroides, metrica, col_texto,vectorizador_modelo, textos_ref, dist_ref, pred_ref):
    """
    - Precondición: El archivo CSV especificado debe existir y contener la columna de texto indicada.
    - Inputs: Ruta del CSV (ruta_csv), args, centroides, metrica, nombre de columna (col_texto), vectorizador_modelo y datos de referencia del train.
    - Outputs: Ninguno (genera y guarda un nuevo archivo CSV con las predicciones y las 5 instancias más cercanas de los clústeres).
    - Postcondición: Exporta un nuevo fichero CSV con una nueva columna que contiene la predicción asignada a cada fila por el modelo K-Means.
    """
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

    X_nuevos = preproceso(df[col_texto], args,fase="predict",vectorizador_entrenado=vectorizador_modelo)

    # Llamada a vuestra función
    predicciones = predict(X_nuevos, centroides, metrica)
    df['Cluster_Predicho'] = predicciones

    #Mostrar las 5 instancias más cercanas del train por clúster
    print(f"\n{Fore.CYAN}5 instancias más cercanas al centroide por clúster (Train){Fore.RESET}")
    n_clusters = centroides.shape[0]

    if textos_ref is not None and dist_ref is not None and pred_ref is not None:
        for i in range(n_clusters):
            indices_cluster = np.flatnonzero(pred_ref == i)
            print(f"{Fore.YELLOW}Clúster {i} (Total elementos en train: {len(indices_cluster)}){Fore.RESET}")
            if len(indices_cluster) == 0:
                print("  |-- Clúster vacío en el conjunto de referencia.\n")
                continue
            top_indices = indices_cluster[np.argsort(dist_ref[indices_cluster, i])[:5]]
            for idx in top_indices:
                print(f"  |-- [Distancia: {dist_ref[idx, i]:.4f}] {textos_ref.iloc[idx]}")
            print()
    else:
        print(f"{Fore.RED}No se pudo cargar el archivo de referencia para mostrar los textos del train.{Fore.RESET}")

    ruta_salida = ruta_csv.replace('.csv', '_predicciones.csv')
    df.to_csv(ruta_salida, index=False)

    print(f"{Fore.GREEN}Predicciones finalizadas. Resultados guardados en: {ruta_salida}{Fore.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de predicción para demostración en vivo.")
    parser.add_argument("-i", "--interactive", default=False, help="Activar modo interactivo")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo .pkl del mejor modelo")
    parser.add_argument("-c", "--csv", default=None, help="Ruta a un archivo .csv opcional")
    parser.add_argument("-r", "--reference", help="Ruta al archivo CSV de entrenamiento/referencia")
    parser.add_argument("-t", "--texto", default="Mensaje", help="Nombre de la columna de texto en el CSV")
    sys_args = parser.parse_args()

    print("Cargando el cerebro del modelo...")
    args_modelo, centroides_fijos, metrica,vectorizador_modelo = cargar_entorno(sys_args.model)
    print(f"Modelo cargado: {centroides_fijos.shape[0]} clusters fijados usando distancia {metrica}.")

    textos_ref, dist_ref, pred_ref = None, None, None
    if sys_args.reference and os.path.exists(sys_args.reference):
        print(f"Cargando conjunto de referencia desde '{sys_args.reference}'...")
        try:
            df_ref = pd.read_csv(sys_args.reference)
            textos_ref = df_ref[sys_args.texto].astype(str)
            X_ref = preproceso(textos_ref, args_modelo, fase="predict", vectorizador_entrenado=vectorizador_modelo)

            # Calcular distancias y predicciones con los centroides fijos
            dist_ref = calcular_distancias(X_ref, centroides_fijos, metrica)
            pred_ref = predict(X_ref, centroides_fijos, metrica)

            print(
                f"{Fore.GREEN}¡Conjunto de referencia procesado con éxito! ({len(textos_ref)} instancias){Fore.RESET}\n")
        except Exception as e:
            print(f"{Fore.RED}Error procesando el archivo de referencia: {e}{Fore.RESET}")
    else:
        print(f"{Fore.YELLOW}Aviso: No se encontró o no se especificó el archivo de referencia.{Fore.RESET}")

    if sys_args.interactive:
        modo_interactivo(args_modelo, centroides_fijos, metrica, vectorizador_modelo, textos_ref, dist_ref, pred_ref)
    else:
        modo_csv(sys_args.csv, args_modelo, centroides_fijos, metrica, sys_args.texto, vectorizador_modelo, textos_ref, dist_ref, pred_ref)