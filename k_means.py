import numpy as np

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
    return clusters, centroides


