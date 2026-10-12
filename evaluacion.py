from colorama import Fore
import pickle
import numpy as np
from sklearn.metrics import silhouette_score, rand_score, fowlkes_mallows_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import contingency_matrix


def evaluar(train):
    with open('kmeans_model.pkl', 'rb') as f:
        data = pickle.load(f)

    kmeans = data['model']
    labels_pred = kmeans.labels_
    X = train['Post']
    y_true = train['Label']

    #---
    print(Fore.YELLOW + f"Empleando métodos de evaluación Internos..." + Fore.RESET)
    #Cohesión interna
    cohesion = kmeans.inertia_
    print(f"Cohesión Interna (Inercia): {cohesion:.4f} (Menor es más coherente cada cluster)")
    #Separabilidad externa
    centroides = kmeans.cluster_centers_
    dist_matrix = euclidean_distances(centroides)
    np.fill_diagonal(dist_matrix, np.inf)  # Ignorar la distancia de un centroide consigo mismo
    separabilidad = np.min(dist_matrix)
    print(f"Separabilidad Externa (Mín. Dist. Centroides): {separabilidad:.4f} (Mayor es más separado)")
    #Silhouette
    sil_score = silhouette_score(X, labels_pred)
    print(f"Silhouette Score: {sil_score:.4f} (Rango [-1, 1], ideal cercano a 1)")

    #---
    print(Fore.YELLOW + f"Empleando métodos de evaluación Externos..." + Fore.RESET)
    conf_matrix = contingency_matrix(y_true, labels_pred)
    print("Tabla de Comparación por Parejas (Matriz de Contingencia):")
    print(conf_matrix)
    #Jaccard
    print(f"Jaccard: {jaccard_clustering(conf_matrix):.4f}")
    #Rand Index
    rand = rand_score(y_true, labels_pred)
    print(f"Rand Index: {rand:.4f}")
    #Fowlkes-Mallows
    fmi = fowlkes_mallows_score(y_true, labels_pred)
    print(f"Fowlkes-Mallows: {fmi:.4f}")


def jaccard_clustering(conf_matrix):
    #Función auxiliar, calcular combinaciones de parejas: n*(n-1)/2
    n_comb = lambda n: n * (n - 1) // 2 if n >= 2 else 0
    #Calcular N11 (pares en el mismo clúster en ambas particiones)
    N11 = sum(n_comb(n) for n in conf_matrix.flatten())
    #Calcular N10 y N01 basados en las sumas marginales de la matriz
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)

    N11_plus_N10 = sum(n_comb(n) for n in row_sums)
    N11_plus_N01 = sum(n_comb(n) for n in col_sums)

    N10 = N11_plus_N10 - N11
    N01 = N11_plus_N01 - N11
    #Jaccard N11 / (N11 + N10 + N01)
    denominador = N11 + N10 + N01
    jaccard = N11 / denominador if denominador > 0 else 0.0
    return jaccard
