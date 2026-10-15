import ast
import hashlib
import os
import re
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_PALABRAS = 80   # ~100-110 tokens, por debajo del límite de 128
MAX_CHUNKS = 50       # tope de trozos por usuario (None = sin tope)

# Preproceso de los datos
def preproceso(text_cols, args):
    text_process = args.preprocessing.get("text_process", "none")
    nltk.download('punkt_tab')

    if text_process == "tf-idf":
        textos = text_cols.apply(clean_text)
        vec = TfidfVectorizer(max_features=5000,
                              ngram_range=(1, 3),
                              min_df=2,
                              max_df=0.9,
                              sublinear_tf=True)
        return vec.fit_transform(textos)

    elif text_process == "embeddings":
        return vectorizar_embeddings(
            text_cols,
            cache_dir=args.preprocessing.get("cache_dir", "cache"),
            chunk_palabras=args.preprocessing.get("chunk_palabras", CHUNK_PALABRAS),
            max_chunks=args.preprocessing.get("max_chunks", MAX_CHUNKS),
        )

    else:
        raise ValueError(f"text_process no soportado: {text_process}")

# Embeddings con caché
def vectorizar_embeddings(text_cols, cache_dir="cache",
                          chunk_palabras=CHUNK_PALABRAS, max_chunks=MAX_CHUNKS):
    textos = [str(t) for t in text_cols]   # un mensaje por elemento

    firma = f"{MODELO_EMBEDDINGS}|msg|{chunk_palabras}|{max_chunks}||"
    hash_datos = hashlib.md5((firma + "\x1f".join(textos)).encode("utf-8")).hexdigest()[:12]

    os.makedirs(cache_dir, exist_ok=True)
    ruta_cache = os.path.join(cache_dir, f"vectores_{hash_datos}.npy")
    if os.path.exists(ruta_cache):
        print(f"Cargando vectores desde caché: {ruta_cache}")
        return np.load(ruta_cache)

    from sentence_transformers import SentenceTransformer
    print("Cargando modelo...")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)

    todos_chunks, inicios = [], []
    for t in textos:
        inicios.append(len(todos_chunks))
        todos_chunks.extend(trocear(t, chunk_palabras, max_chunks))
    fines = inicios[1:] + [len(todos_chunks)]
    print(f"{len(textos)} mensajes -> {len(todos_chunks)} trozos")

    print("Vectorizando textos...")
    emb = modelo.encode(todos_chunks, batch_size=64,
                        show_progress_bar=True, normalize_embeddings=True)

    X_raw = np.vstack([emb[i:f].mean(axis=0) for i, f in zip(inicios, fines)])
    normas = np.linalg.norm(X_raw, axis=1, keepdims=True)
    normas[normas == 0] = 1
    X = X_raw / normas

    np.save(ruta_cache, X)
    print(f"Vectores guardados en {ruta_cache} con forma {X.shape}")
    return X

def parsear_mensajes(crudo):
    """Convierte "['msg1', 'msg2']" en ['msg1', 'msg2'].
    Si la lista está truncada (cadena sin cerrar), recupera los mensajes
    separando por los delimitadores ', ' entre comillas."""
    try:
        lista = ast.literal_eval(crudo)
        if isinstance(lista, (list, tuple)):
            return [str(m) for m in lista]
        return [str(lista)]
    except (ValueError, SyntaxError):
        s = str(crudo).strip()
        s = re.sub(r"^\[\s*['\"]", "", s)
        s = re.sub(r"['\"]\s*\]\s*$", "", s)
        return re.split(r"['\"]\s*,\s*['\"]", s)

def explotar_mensajes(df, col_post="Post", col_user="User",
                      col_label="Label", min_palabras=1):
    """De una fila por usuario a una fila por mensaje.
    Cada mensaje hereda el usuario y la etiqueta de su autor."""
    filas = []
    for user, post, label in zip(df[col_user], df[col_post], df[col_label]):
        for m in parsear_mensajes(post):
            m = m.strip()
            if len(m.split()) >= min_palabras:
                filas.append((user, label, m))
    return pd.DataFrame(filas, columns=[col_user, col_label, "Mensaje"])


def trocear(texto, chunk_palabras, max_chunks):
    """Divide un texto en trozos de `chunk_palabras` palabras.
    Si hay más de `max_chunks`, se queda con trozos repartidos uniformemente
    a lo largo del texto (no solo los primeros)."""
    palabras = texto.split()
    if not palabras:
        return [""]
    chunks = [" ".join(palabras[i:i + chunk_palabras])
              for i in range(0, len(palabras), chunk_palabras)]
    if max_chunks and len(chunks) > max_chunks:
        idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in idx]
    return chunks

#Limpiar texto (solo para TF-IDF)
def clean_text(text):
    nltk.download(['stopwords', 'punkt'], quiet=True)
    #Función de limpieza con manejo de negaciones
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    negaciones = {'no', 'not', 'neither', 'nor', 'none', 'never'}
    stop_words = stop_words - negaciones
    #Texto minúsculas
    text = str(text).lower()
    #Limpieza texto
    clean_text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    tokens = word_tokenize(clean_text, language='english')
    cleaned = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned)
