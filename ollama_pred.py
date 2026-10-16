import pickle
import numpy as np
from colorama import Fore
import ollama

def predecir_clusters(pkl, X_train, textos_train, text_process):
    with open(pkl, 'rb') as f:
        data = pickle.load(f)

    centroides = data['cluster_centers_']
    n_clusters = data['n_clusters']

    print(Fore.CYAN + f"Interpretación de clústeres con Ollama..." + Fore.RESET)

    for i in range(n_clusters):
        print(Fore.CYAN + f"Clúster {i}" + Fore.RESET)
        if text_process in ["tf-idf", "embeddings"]:
            distancias = np.linalg.norm(X_train - centroides[i], axis=1)
            indices_cercanos = np.argsort(distancias)[:5]

            #Asegurarnos de que funcione tanto con Series de Pandas como con listas/arrays
            if hasattr(textos_train, 'iloc'):
                ejemplos_textos = [f"- {textos_train.iloc[idx]}" for idx in indices_cercanos]
            else:
                ejemplos_textos = [f"- {textos_train[idx]}" for idx in indices_cercanos]

            info_contexto = "Ejemplos de textos que pertenecen a este clúster:\n" + "\n".join(ejemplos_textos)
        else:
            raise ValueError(f"text_process no soportado: {text_process}")

        #Definir prompt
        prompt = (
            f"{info_contexto}\n\n"
            f"Basándote exclusivamente en esta información, responde estrictamente con:\n"
            f"1. Una etiqueta de entre (Supportive, Ideation, Behavior, Attempt) según la probabilidad de cometer suicidio."
            f"2. Una explicación de una sola línea de por qué."
        )

        #Llamada a Ollama
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {
                    'role': 'system',
                    'content': 'Eres un analista de datos experto en minería de datos. Sé muy conciso.'
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