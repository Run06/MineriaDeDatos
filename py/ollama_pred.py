import os
import pickle
import numpy as np
from colorama import Fore
import ollama
from py.k_means import calcular_distancias

def predecir_clusters(pkl, X_train, textos_train, text_process):
    file_path = os.path.join('output/modelos/', pkl)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    centroides = data['cluster_centers_']
    n_clusters = data['n_clusters']

    etiquetas = np.asarray(data["labels_"])
    metrica = data.get("distancia", "Euclidean")

    if len(etiquetas) != X_train.shape[0]:
        raise ValueError(
            "Las etiquetas guardadas no coinciden con las filas de entrenamiento."
        )

    distancias = calcular_distancias(
        X_train,
        centroides,
        metrica,
    )

    print(Fore.CYAN + f"Interpretación de clústeres con Ollama..." + Fore.RESET)

    for i in range(n_clusters):
        if text_process in ["bow", "tf-idf", "embeddings", "doc2vec", "emotion"]:
            print(Fore.CYAN + f"Clúster {i}" + Fore.RESET)
            indices = np.flatnonzero(etiquetas == i)
            if len(indices) == 0:
                print("Clúster vacío: no hay ejemplos para interpretar.")
                print("-" * 20)
                continue
            # Elegir los cinco miembros más próximos al centroide.
            indices = indices[
                np.argsort(distancias[indices, i])[:5]
            ]
            if hasattr(textos_train, "iloc"):
                ejemplos_textos = [f"- {textos_train.iloc[idx]}" for idx in indices]
            else:
                ejemplos_textos = [f"- {textos_train[idx]}" for idx in indices]

            info_contexto = ("Ejemplos representativos del clúster:\n" + "\n".join(ejemplos_textos))
        else:
            raise ValueError(f"text_process no soportado: {text_process}")

        #Definir prompt
        prompt = (
            f"{info_contexto}\n\n"
            f"Basándote exclusivamente en estos textos, responde estrictamente con (Si no pudieras responder, di brevemente la razón, dos o tres palabras.):\n"
            f"1. Una etiqueta que resuma todos los textos, máximo tres palabras."
            f"2. Una explicación de una sola línea de por qué."
        )

        #Llamada a Ollama
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {
                    'role': 'system',
                    'content': 'Eres un analista experto en minería de datos, '
                               'que está realizando un análisis para detectar síntomas de malestar emocional en la gente en redes sociales.'
                               'Sé muy conciso.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={'temperature': 0.1}
        )

        print(Fore.GREEN + f"Respuesta Ollama:" + Fore.RESET)
        print(response['message']['content'])
        print("-"*20)