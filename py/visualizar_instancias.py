import os
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from colorama import Fore

from py.preproceso import preproceso, preparar_instancias, COL_TEXTO
from py.k_means import predict

def parse_args():
    """
        - Precondición: Ninguna.
        - Inputs: Argumentos introducidos por la línea de comandos al ejecutar el script directamente.
        - Outputs: Objeto con los argumentos analizados.
        - Postcondición: Define y extrae el parámetro `-f` / `--file` permitiendo inyectar dinámicamente el nombre del archivo CSV.
    """
    parser = argparse.ArgumentParser(description="Generar t-SNE del mejor modelo")
    parser.add_argument("-f", "--file", default="500_Reddit_users_posts_labels.csv", help="CSV de entrada")
    args = parser.parse_args()
    return args

def visualizar_tsne(archivo_csv="500_Reddit_users_posts_labels.csv"):
    """
        - Precondición: Debe existir un archivo modelo (.pkl) válido en el directorio 'output/mejor_modelo/'. Los archivos 'clasificador.json' y el CSV objetivo deben estar accesibles en la raíz del proyecto.
        - Inputs: archivo_csv (str) indicando el nombre del archivo de datos a procesar y evaluar.
        - Outputs: Ninguno.
        - Postcondición: Reconstruye el entorno del mejor modelo, vectoriza el texto congelando los parámetros de entrenamiento (fase='predict'), agrupa los datos e invoca t-SNE para la reducción dimensional a 2D. Guarda físicamente un panel comparativo (.png) mostrando las etiquetas originales frente a los clústeres descubiertos.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_mejor = os.path.join(root_dir,"output", "mejor_modelo")
    archivos_pkl = glob.glob(os.path.join(carpeta_mejor, "*.pkl"))

    if not archivos_pkl:
        print(f"{Fore.RED}No se encontró ningún modelo en {carpeta_mejor}{Fore.RESET}")
        return

    ruta_pkl = archivos_pkl[0]
    print(f"{Fore.CYAN}Cargando el mejor modelo: {os.path.basename(ruta_pkl)}{Fore.RESET}")

    class ArgsWrapper:
        pass
    args = ArgsWrapper()
    ruta_json = os.path.join(root_dir, "clasificador.json")
    with open(ruta_json, "r") as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    args.target = getattr(args, 'target', 'Post')
    args.label = getattr(args, 'label', 'Label')

    with open(ruta_pkl, "rb") as f:
        datos_modelo = pickle.load(f)

    metodo_original = datos_modelo.get('text_process', 'none')
    if hasattr(args, 'preprocessing'):
        args.preprocessing['text_process'] = metodo_original

    centroides = datos_modelo['cluster_centers_']
    vectorizador = datos_modelo.get('vectorizador', None)
    metrica = datos_modelo.get('distancia', 'Euclidean')
    ruta_csv = os.path.join(root_dir, archivo_csv)
    df = pd.read_csv(ruta_csv)
    df_prep = preparar_instancias(df, args)
    print(f"Vectorizando {len(df_prep)} instancias con {metodo_original}...")
    X = preproceso(df_prep[COL_TEXTO], args, fase="predict", vectorizador_entrenado=vectorizador)

    clusters_predichos = predict(X, centroides, metrica)

    print(f"{Fore.YELLOW}Calculando t-SNE (esto puede tardar unos segundos)...{Fore.RESET}")
    if hasattr(X, "toarray"):
        X = X.toarray()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Extraer y numerar las etiquetas reales para darles color
    etiquetas_reales = df_prep[args.label]
    etiquetas_numericas, nombres_etiquetas = pd.factorize(etiquetas_reales)

    # Crear lienzo con dos subgráficos (1 fila, 2 columnas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Gráfico 1: Etiquetas Reales (Ground Truth) ---
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=etiquetas_numericas, cmap='Set1', alpha=0.7, edgecolors='k', s=50)
    ax1.set_title('Ground Truth (Etiquetas Originales)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dimensión 1 t-SNE')
    ax1.set_ylabel('Dimensión 2 t-SNE')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Leyenda con los nombres reales
    legend1 = ax1.legend(handles=scatter1.legend_elements()[0], labels=list(nombres_etiquetas), title="Clases Reales",
                         loc='best')
    ax1.add_artist(legend1)

    # --- Gráfico 2: Clústeres de K-Means ---
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters_predichos, cmap='tab20', alpha=0.7, edgecolors='k', s=50)
    ax2.set_title(f'Predicción K-Means (K={len(np.unique(clusters_predichos))})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimensión 1 t-SNE')
    ax2.grid(True, linestyle=':', alpha=0.5)

    handles, _ = scatter2.legend_elements(num=len(np.unique(clusters_predichos)))
    etiquetas_clusters = [str(c) for c in np.unique(clusters_predichos)]
    handles = handles[:len(etiquetas_clusters)]
    legend2 = ax2.legend(handles, etiquetas_clusters, title="Clústeres", loc='best')
    ax2.add_artist(legend2)

    # Estética general
    fig.suptitle(f'Comparación t-SNE: Real vs Clustering (Método: {metodo_original})', fontsize=18, fontweight='bold')
    plt.tight_layout()

    ruta_grafico = os.path.join(carpeta_mejor, "tsne_comparacion.png")
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    print(f"{Fore.GREEN}Gráfico t-SNE comparativo guardado con éxito en: {ruta_grafico}{Fore.RESET}")


if __name__ == "__main__":
    args = parse_args()
    visualizar_tsne(args.file)