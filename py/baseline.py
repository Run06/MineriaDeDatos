import time
import tracemalloc
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from k_means import kmeans

def medir_rendimiento(func, *args, **kwargs):
    """
    - Precondición: La función de entrada func debe ser un objeto invocable.
    - Inputs: Una función func, argumentos posicionales *args y argumentos de palabras clave **kwargs.
    - Outputs: Una tupla que contiene el resultado de la función, el tiempo total de ejecución en segundos y el pico de memoria en MB.
    """
    tracemalloc.start()
    inicio_tiempo = time.perf_counter()

    resultado = func(*args, **kwargs)

    fin_tiempo = time.perf_counter()
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tiempo_total = fin_tiempo - inicio_tiempo
    pico_mb = pico / (1024 * 1024)

    return resultado, tiempo_total, pico_mb


def baseline():
    """
    - Precondición: Ninguna.
    - Inputs: Ninguno.
    - Outputs: Ninguno (ejecuta una comparativa de rendimiento entre Sklearn KMeans y el modelo propio, imprimiendo el consumo de memoria y el tiempo por consola.).
    """
    #Generamos un dataset de prueba
    print("Generando datos de prueba...")
    X, _ = make_blobs(n_samples=5000, n_features=128, centers=14, random_state=42)
    k = 14

    print(f"\nProbando Sklearn KMeans (k={k})")

    def run_sklearn():
        model = SklearnKMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        return model

    _, tiempo_sk, mem_sk = medir_rendimiento(run_sklearn)
    print(f"Tiempo: {tiempo_sk:.4f} s")
    print(f"Pico de Memoria: {mem_sk:.2f} MB")

    print(f"\nProbando Nuestro KMeans (k={k})")

    def run_custom():
        #Nuestro modelo
        clusters, centroides, nombre_archivo = kmeans(
            X, n_clusters=k, modo='plusplus', distancia='Euclidean'
        )
        return centroides

    _, tiempo_custom, mem_custom = medir_rendimiento(run_custom)
    print(f"Tiempo: {tiempo_custom:.4f} s")
    print(f"Pico de Memoria: {mem_custom:.2f} MB")


if __name__ == "__main__":
    print("Comparativa entre nuestro modelo frente al de Sklearn")
    baseline()