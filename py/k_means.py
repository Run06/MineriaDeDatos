import os
import numpy as np
import pickle
import scipy.sparse as sp

#Validación de los parametros
####################################3
def validar_parametros(X, n_clusters,distancia):
    n_muestras = X.shape[0]
    if n_clusters > n_muestras:
        raise ValueError(f"Error: El número de clusters ({n_clusters}) no puede ser mayor que {n_muestras})")
    #Aseguramos el formato CSR para operaciones matriciales rápidas en caso de que sea dispersa
    if sp.issparse(X) and not sp.isspmatrix_csr(X):
        X = X.tocsr()
    #En caso de distancia Manhattan se convierte a denso
    if distancia == 'Manhattan' and sp.issparse(X):
        X = X.toarray()
    return X

def inicializar_centroides(X, n_clusters,modo,random_state,distancia):
    np.random.seed(random_state)
    n_muestras = X.shape[0]

    if modo == 'kmeans':
        indices = np.random.choice(n_muestras, n_clusters, replace=False)
        centroides = X[indices]
        return centroides.toarray() if sp.issparse(centroides) else centroides
    elif modo == 'plusplus':
        centroides = []
        idx = np.random.choice(n_muestras)
        primer_c = np.ravel(X[idx].toarray()) if sp.issparse(X) else np.ravel(X[idx])
        centroides.append(primer_c)

        for _ in range(1, n_clusters):
            dist_a_centroides = calcular_distancias(X, np.array(centroides), distancia)
            distancias_minimas = np.min(dist_a_centroides, axis=1)

            dist_sq = distancias_minimas ** 2
            suma_distancias = np.sum(dist_sq)
            probabilidades = (dist_sq / suma_distancias) if suma_distancias > 0 else (np.ones(n_muestras) / n_muestras)

            siguiente_idx = np.random.choice(n_muestras, p=probabilidades)
            nuevo_centroide = np.ravel(X[siguiente_idx].toarray()) if sp.issparse(X) else np.ravel(X[siguiente_idx])
            centroides.append(nuevo_centroide)
        return np.array(centroides)
    else:
        raise ValueError('Valor no valido de modo introducido')

def calcular_distancias(X,centroides,distancia):
    n_muestras = X.shape[0]
    n_clusters = centroides.shape[0]
    distancias = np.zeros((n_muestras, n_clusters))

    if distancia == 'Cosine':
        if sp.issparse(X):
            norm_X = np.sqrt(np.ravel(X.power(2).sum(axis=1)))[:, np.newaxis]
            producto_escalar = X.dot(centroides.T)
        else:
            norm_X = np.linalg.norm(X, axis=1, keepdims=True)
            producto_escalar = X @ centroides.T

        norm_C = np.linalg.norm(centroides, axis=1)
        denominador = np.maximum(norm_X * norm_C, 1e-10)
        distancias = 1.0 - (producto_escalar / denominador)

    elif distancia == 'Euclidean':
        if sp.issparse(X):
            X_sq = np.ravel(X.power(2).sum(axis=1))
            prod = X.dot(centroides.T)
        else:
            X_sq = np.sum(X ** 2, axis=1)
            prod = X @ centroides.T

        C_sq = np.sum(centroides ** 2, axis=1)

        for i in range(n_clusters):
            dist_sq = X_sq + C_sq[i] - 2 * prod[:, i]
            distancias[:, i] = np.sqrt(np.maximum(dist_sq, 0))

    elif distancia == 'Manhattan':
        for i in range(n_clusters):
            distancias[:, i] = np.sum(np.abs(X - centroides[i]), axis=1)
    else:
        raise ValueError('Distancia no válida')
    return distancias

def actualizar_centroides(X,clusters,n_clusters,n_caract,distancia):
    centroides = np.zeros((n_clusters, n_caract))
    for i in range(n_clusters):
        indices = np.where(clusters == i)[0]
        if len(indices)>0:
            puntos_cluster = X[indices]

            if sp.issparse(puntos_cluster):
                # .mean() en matriz dispersa devuelve una matriz de Scipy, con .A1 la aplanamos a vector normal
                centroides[i] = puntos_cluster.mean(axis=0).A1
            else:
                centroides[i] = np.mean(puntos_cluster, axis=0)

            # RENORMALIZACIÓN (Spherical K-means)
            if distancia == 'Cosine':
                norma = np.linalg.norm(centroides[i])
                if norma > 0:
                    centroides[i] = centroides[i] / norma
        else:
            idx_random = np.random.choice(X.shape[0])
            centroides[i] = np.ravel(X[idx_random].toarray()) if sp.issparse(X) else np.ravel(X[idx_random])
            if distancia == 'Cosine':
                norma = np.linalg.norm(centroides[i])
                if norma > 0:
                    centroides[i] = centroides[i] / norma
    return centroides

        #Implementación de clustering sin uso de librerias externas
def kmeans(X,n_clusters,random_state=42,distancia='Euclidean',modo='kmeans',max_iter=300,tol=1e-4):
    X = validar_parametros(X,n_clusters,distancia)
    n_muestras, n_caracteristicas = X.shape

    #Inicializar
    centroides = inicializar_centroides(X,n_clusters,modo,random_state,distancia)

    #Bucle principal
    for iteracion in range(max_iter):
        centroides_anteriores = np.copy(centroides)
        #Asignación
        distancias = calcular_distancias(X,centroides,distancia)
        clusters = np.argmin(distancias, axis=1)
        #Actualizacion
        centroides = actualizar_centroides(X,clusters,n_clusters,n_caracteristicas,distancia)
        #Criterio de convergencia
        if np.linalg.norm(centroides - centroides_anteriores) <= tol:
            break
    #---
    #Almacenar pkl
    nombre_archivo = f"{modo}-{distancia}-{n_clusters}.pkl"

    datos_modelo = {
        'labels_': clusters,
        'cluster_centers_': centroides,
        'n_clusters': n_clusters,
        'modo': modo,
        'distancia': distancia
    }

    file_path = os.path.join('output/', nombre_archivo)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(datos_modelo, f)


    return clusters, centroides, nombre_archivo


def predict(X_nuevos, centroides, metrica='Euclidean'):
    """
    Fase de inferencia: Asigna nuevos datos a los clústeres existentes
    calculando la distancia a los centroides fijos
    """

    distancias = calcular_distancias(X_nuevos, centroides, metrica)

    return np.argmin(distancias, axis=1)

