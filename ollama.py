import pickle
from colorama import Fore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def predecir_clusters():
    with open('kmeans_model.pkl', 'rb') as f:
        data = pickle.load(f)

    kmeans = data['model']
    vectorizer = data['vectorizer']

    #Inicializar Ollama (llama3.1)
    ollama = ChatOllama(model="llama3.1", temperature=0.1)

    #Definir prompt, analice palabras e infiera tema
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista de datos experto en minería de textos y clustering."),
        ("human",
         "Estas son las 10 palabras que más definen a un clúster de documentos según sus pesos:\n\n{keywords}\n\n"
         "Basándote exclusivamente en estas palabras:\n"
         "1. Enumera las 10 palabras.\n"
         "2. Explica brevemente de qué trata o cuál es la temática común de este clúster.")
    ])

    chain = prompt_template | ollama

    print(Fore.CYAN + f"Interpretación de clústeres con Ollama..." + Fore.RESET)

    #Extraer 10 palabras por clúster usando los centroides del modelo
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(kmeans.n_clusters):
        # Obtener las 10 palabras principales del clúster i
        top_indices = order_centroids[i, :10]
        top_words = [terms[ind] for ind in top_indices]
        keywords_str = ", ".join(top_words)

        print(Fore.CYAN + f"Clúster {i}" + Fore.RESET)
        print(f"Palabras clave: {keywords_str}")

        #Respuesta Ollama
        print(Fore.CYAN + f"Respuesta Ollama" + Fore.RESET)
        response = chain.invoke({"keywords": keywords_str})
        print(response.content)
        print("-"*50)