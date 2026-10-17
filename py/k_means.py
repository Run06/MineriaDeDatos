import os
import numpy as np
import pickle

#Validación de los parametros
####################################3
def validar():
    return NotImplemented

#Implementación de clustering sin uso de librerias externas
def kmeans(X,n_clusters,random_state=42,distancia='Euclidean',modo='kmeans',max_iter=300,tol=1e-4):
    np.random.seed(random_state)
    n_muestras = X.shape[0]
    #Eleccion de los centroides iniciales
    if modo=='plusplus':
        centroides = []
        idx = np.random.choice(n_muestras)
        centroides.append(X[idx])
        for _ in range(1, n_clusters):
            distancias_a_centroides = np.zeros((n_muestras, len(centroides)))
            for j, c in enumerate(centroides):
                #Usamos la distancia Euclidiana al cuadrado (Estandár K-means++)
                distancias_a_centroides[:, j] = np.sum((X - c) ** 2, axis=1)

            distancias_minimas = np.min(distancias_a_centroides, axis=1)

            suma_distancias = np.sum(distancias_minimas)
            if suma_distancias == 0:
                probabilidades = np.ones(n_muestras) / n_muestras
            else:
                probabilidades = distancias_minimas / suma_distancias

            siguiente_idx = np.random.choice(n_muestras, p=probabilidades)
            centroides.append(X[siguiente_idx])

        centroides = np.array(centroides)

    elif modo=='kmeans':
        indices_aleatorios = np.random.choice(n_muestras, n_clusters, replace=False)
        centroides = X[indices_aleatorios]
    else:
        raise ValueError('Valor no valido de modo introducido')

    #Bucle Principal K-means
    for iteracion in range(max_iter):
        centroides_anteriores = np.copy(centroides)
        distancias = np.zeros((n_muestras,n_clusters))

        if distancia == 'Euclidean':
            for i in range(n_clusters):
                distancias[:,i] = np.linalg.norm(X-centroides[i],axis=1)
        elif distancia == 'Manhattan':
            for i in range(n_clusters):
                distancias[:, i] = np.sum(np.abs(X - centroides[i]), axis=1)
        elif distancia == 'Cosine':
            # Calculamos las normas de cada vector
            # keepdims=True mantiene la forma (N, 1)
            norm_X = np.linalg.norm(X,axis=1,keepdims=True)
            norm_C = np.linalg.norm(centroides,axis=1)
            #Multiplicamos las normas, 1e-10 es para evitar la division por 0 en caso de post vacio
            denominador = np.maximum(norm_X * norm_C, 1e-10)
            #Producto escalar de todos contra todos a la vez
            producto_escalar = X @ centroides.T
            similitud = producto_escalar / denominador
            # K-Means asigna el punto a la distancia MÍNIMA.
            # La similitud máxima es 1. Por lo tanto, Distancia = 1 - Similitud.
            # (Distancia 0 = idénticos, Distancia 2 = opuestos)
            distancias = 1.0 - similitud
        else:
            raise ValueError('Distancia no válida')
        clusters = np.argmin(distancias, axis=1)
        for i in range(n_clusters):
            puntos_cluster = X[clusters == i]
            if len(puntos_cluster) > 0:
                centroides[i] = np.mean(puntos_cluster, axis=0)
        #Criterio de parada
        desplazamiento = np.linalg.norm(centroides - centroides_anteriores)
        if desplazamiento <= tol:
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


