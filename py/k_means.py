import os
import numpy as np
import pickle
import scipy.sparse as sp

#Validación de los parametros
def validar_parametros(X, n_clusters,distancia):
    """
        - Precondición: X debe ser una matriz de datos (Numpy ndarray o Scipy sparse).
        - Inputs: X (matriz N x M), n_clusters (int), distancia (str).
        - Outputs: Matriz X validada y formateada (CSR o densa).
        - Postcondición: Lanza ValueError si hay más clústeres que muestras. Convierte la matriz a formato CSR (si es dispersa) para eficiencia, o a matriz densa si se exige la distancia Manhattan.
    """
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
    """
        - Precondición: La matriz X ya debe estar validada y formateada.
        - Inputs: X (matriz de datos), n_clusters (int), modo (str: 'kmeans' o 'plusplus'), random_state (int), distancia (str).
        - Outputs: Array Numpy 2D con los centroides inicializados.
        - Postcondición: Fija la semilla aleatoria global. Retorna los centroides extraídos aleatoriamente ('kmeans') o distribuidos de forma probabilística basada en distancias ('plusplus'). Lanza ValueError si el modo es inválido.
    """
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
    """
        - Precondición: X y centroides deben tener exactamente la misma dimensionalidad M (características).
        - Inputs: X (matriz de muestras), centroides (array Numpy de centroides actuales), distancia (str: 'Cosine', 'Euclidean', 'Manhattan').
        - Outputs: Matriz Numpy 2D de dimensiones (N_muestras, K_clusters) con las distancias.
        - Postcondición: Calcula matricialmente (sin bucles explícitos por muestra) la distancia requerida soportando tanto estructuras densas como dispersas (sparse).
    """
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
    """
        - Precondición: El vector clusters debe contener asignaciones válidas (de 0 a K-1) para todas las muestras en X.
        - Inputs: X (matriz de muestras), clusters (array 1D de asignaciones), n_clusters (int), n_caract (int, nº de columnas), distancia (str).
        - Outputs: Array Numpy 2D con los nuevos centroides actualizados.
        - Postcondición: Reposiciona los centroides calculando la media espacial de los puntos asignados. Si un clúster queda vacío, se reasigna a un punto aleatorio. Aplica renormalización L2 si la métrica es Cosine (Spherical K-Means).
    """
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
def kmeans(X,n_clusters,vectorizador=None,text_process="none",random_state=42,distancia='Euclidean',modo='kmeans',max_iter=300,tol=1e-4):
    """
        - Precondición: El entorno debe tener permisos de escritura en la carpeta 'output/' para serializar el modelo final.
        - Inputs: Matriz X, hiperparámetros del algoritmo (n_clusters, vectorizador, etc.), criterios de parada (max_iter, tol).
        - Outputs: Tupla (clusters, centroides, nombre_archivo).
        - Postcondición: Ejecuta el bucle hasta alcanzar la tolerancia o max_iter. Empaqueta el modelo resultante, sus hiperparámetros y el vectorizador preentrenado en un archivo binario (.pkl) y retorna los resultados.
    """
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
        'distancia': distancia,
        'vectorizador': vectorizador,
        'text_process': text_process
    }

    file_path = os.path.join('output/', nombre_archivo)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(datos_modelo, f)


    return clusters, centroides, nombre_archivo


def predict(X_nuevos, centroides, metrica='Euclidean'):
    """
    - Precondición: X_nuevos debe haber sido vectorizado con el mismo modelo y dimensionalidad que originó los centroides.
    - Inputs: X_nuevos (matriz de datos no vistos), centroides (array Numpy del modelo entrenado), metrica (str).
    - Outputs: Array Numpy 1D con las etiquetas numéricas de los clústeres asignados.
    - Postcondición: Calcula la distancia matemática entre las nuevas instancias y los centroides fijos en el espacio latente, asignando a cada muestra el clúster con distancia mínima (argmin).
    """

    distancias = calcular_distancias(X_nuevos, centroides, metrica)

    return np.argmin(distancias, axis=1)

