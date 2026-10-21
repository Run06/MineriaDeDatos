import argparse
import os
import json
import sys
import numpy as np
import pandas as pd
from colorama import Fore
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from py.preproceso import preproceso, preparar_instancias, COL_TEXTO
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

    args = parser.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args
#Lista global
historial_memoria = []
def registrar_memoria(fase):
    #Obtener memoria actual
    proceso = psutil.Process(os.getpid())
    mem_proceso_mb = proceso.memory_info().rss / (1024 * 1024)

    #Memoria completa del ordenador
    mem_sistema = psutil.virtual_memory()
    mem_sistema_gb = mem_sistema.used / (1024 * 1024 * 1024)

    historial_memoria.append({
        'fase': fase,
        'ram_python_mb': mem_proceso_mb,
        'ram_sistema_gb': mem_sistema_gb
    })

    # Calcular los picos históricos iterando sobre la lista
    pico_python = max(registro['ram_python_mb'] for registro in historial_memoria)
    pico_sistema = max(registro['ram_sistema_gb'] for registro in historial_memoria)

    # Imprimir por consola
    print(Fore.CYAN + f"[{fase}]" + Fore.RESET)
    print(f"  ├─ RAM Python:  Actual {mem_proceso_mb:.2f} MB | Pico: {pico_python:.2f} MB")
    print(f"  └─ RAM Sistema: Actual {mem_sistema_gb:.2f} GB | Pico: {pico_sistema:.2f} GB\n")
def graficar_memoria(nombre_modelo="grafico_memoria"):
    if not historial_memoria:
        return

    fases = [h['fase'] for h in historial_memoria]
    ram_python = [h['ram_python_mb'] for h in historial_memoria]
    ram_sistema = [h['ram_sistema_gb'] for h in historial_memoria]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Eje Y Izquierdo: RAM Python (MB) ---
    color1 = 'tab:blue'
    ax1.set_xlabel('Fases del Proceso', fontweight='bold')
    ax1.set_ylabel('RAM Script Python (MB)', color=color1, fontweight='bold')
    ax1.plot(fases, ram_python, marker='o', color=color1, linewidth=2, label='Python (MB)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(range(len(fases)))
    ax1.set_xticklabels(fases, rotation=45, ha="right")

    max_python = max(ram_python)
    limite_sup_python = max_python * 1.1
    ax1.set_ylim(0, limite_sup_python)
    ax1.set_xticks(range(len(fases)))
    ax1.set_yticks(np.arange(0, limite_sup_python + 50, 50))

    # Añadir los valores numéricos debajo de cada punto de Python
    for i, valor in enumerate(ram_python):
        ax1.annotate(f"{valor:.0f} MB",
                     (fases[i], valor),
                     textcoords="offset points",
                     xytext=(0, -15),  # Desplazamiento hacia abajo
                     ha='center', va='top',
                     fontsize=9, color=color1, fontweight='bold')

    # --- Eje Y Derecho: RAM Total Sistema (GB) ---
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('RAM Total Sistema (GB)', color=color2, fontweight='bold')
    ax2.plot(fases, ram_sistema, marker='s', color=color2, linewidth=2, linestyle='--', label='Sistema (GB)')
    ax2.tick_params(axis='y', labelcolor=color2)

    max_sistema = max(ram_sistema)
    limite_sup_sistema = max_sistema * 1.1
    ax2.set_ylim(0, limite_sup_sistema)
    ax2.set_yticks(np.arange(0, limite_sup_sistema + 1.0, 1.0))

    # Añadir los valores numéricos debajo de cada punto del Sistema
    for i, valor in enumerate(ram_sistema):
        ax2.annotate(f"{valor:.1f} GB",
                     (fases[i], valor),
                     textcoords="offset points",
                     xytext=(0, 10),  # Desplazamiento hacia abajo
                     ha='center', va='top',
                     fontsize=9, color=color2, fontweight='bold')

    # Estética y guardado
    fig.tight_layout()
    plt.title('Perfil de Consumo de Memoria durante la Ejecución', fontsize=14, pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)

    nombre_limpio = nombre_modelo.replace('.pkl', '')
    ruta_grafico = os.path.join('output', 'UsoDeMemoria', f"{nombre_limpio}.png")
    os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)
    plt.savefig(ruta_grafico, dpi=300)
    print(Fore.GREEN + f"\nGráfico guardado en '{ruta_grafico}'" + Fore.RESET)

if __name__ == "__main__":
    args = parse_args()
    registrar_memoria("Inicio del programa")
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

    registrar_memoria("Carga CSV")

    #Unidad de análisis: "usuario" (una fila por usuario) o "mensaje" (una fila por mensaje)
    #Se divide ANTES por usuario (arriba) para que los mensajes de un mismo usuario no caigan en train y test
    unidad = args.preprocessing.get("unidad", "usuario")
    train = preparar_instancias(train, args)
    test = preparar_instancias(test, args)
    print(f"Unidad de análisis: {unidad} -> {len(train)} instancias en train, {len(test)} en test")

    print(Fore.MAGENTA + f"Preprocesando datos del train..." + Fore.RESET)
    train_prep = preproceso(train[COL_TEXTO], args)
    print(Fore.GREEN + f"Datos preprocesados!" + Fore.RESET)
    registrar_memoria("Preprocesado")
    print("-"*60)

    X_train = train_prep

    #Ejecución K-means
    print(Fore.MAGENTA + f"Ejecutando K-Means..." + Fore.RESET)
    clusters, centroides, nombre_archivo = kmeans(X_train, **args.KMeans)
    nombre_archivo_final = args.preprocessing.get("text_process")+'-'+unidad+'-'+nombre_archivo
    os.makedirs(os.path.join('output', 'modelos'), exist_ok=True)
    ruta_origen = os.path.join('output', nombre_archivo)
    ruta_destino = os.path.join('output', 'modelos', nombre_archivo_final)
    if os.path.exists(ruta_destino):
        os.remove(ruta_destino)
    os.rename(ruta_origen, ruta_destino)
    print(Fore.GREEN + f"Modelo guardado en '{ruta_destino}'" + Fore.RESET)
    print(f"Centroides finales calculados: {centroides.shape[0]}")
    registrar_memoria("K-Means")
    print("-"*60)

    #Evaluación
    y_train = train[args.label].to_numpy()
    evaluar(X_train, y_train, nombre_archivo_final)
    registrar_memoria("Evaluación")
    print("-"*60)

    #Predicción Ollama
    #ANTES DE EJECUTAR!!!
    #Descargar Ollama: https://ollama.com/download, después la versión 3.1 (en terminal del PC: ollama pull llama3.1)
    #Dejar abierto Ollama en segundo plano
    if args.ollama:
        predecir_clusters(nombre_archivo_final, X_train, train[COL_TEXTO], args.preprocessing.get("text_process"))
        registrar_memoria("Ollama")
    graficar_memoria(nombre_archivo_final)