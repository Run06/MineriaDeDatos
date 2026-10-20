import ast
import hashlib
import os
import re
import time
import zlib
import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_PALABRAS = 80   # ~100-110 tokens, por debajo del límite de 128
MAX_CHUNKS = 50       # tope de trozos por instancia cuando la instancia es un usuario
MAX_CHUNKS_MENSAJE = 10   # tope cuando la instancia es un mensaje (None = sin tope)
COL_TEXTO = "Mensaje"     # nombre de la columna de texto que generan las instancias
UNIDADES = ("usuario", "mensaje")
MODELO_EMOCIONES = "pysentimiento-emotion-en"   # BERTweet fine-tuned en emociones (inglés)

# Preproceso de los datos
def preproceso(text_cols, args):
    cfg = args.preprocessing
    text_process = cfg.get("text_process", "none")
    unidad = cfg.get("unidad", "usuario")
    max_chunks_defecto = MAX_CHUNKS_MENSAJE if unidad == "mensaje" else MAX_CHUNKS

    if text_process == "tf-idf":
        textos = text_cols.apply(clean_text)
        vec = TfidfVectorizer(max_features=5000,
                              ngram_range=(1, 3),
                              min_df=2,
                              max_df=0.9,
                              sublinear_tf=True)
        return vec.fit_transform(textos)

    elif text_process == "bow":
        textos = text_cols.apply(clean_text)
        vec = CountVectorizer(max_features=10000, ngram_range=(1, 2))
        return vec.fit_transform(textos)

    elif text_process == "embeddings":
        return vectorizar_embeddings(
            text_cols,
            cache_dir=cfg.get("cache_dir", "cache"),
            chunk_palabras=cfg.get("chunk_palabras", CHUNK_PALABRAS),
            max_chunks=cfg.get("max_chunks", max_chunks_defecto),
        )

    elif text_process == "doc2vec":
        return vectorizar_doc2vec(
            text_cols,
            vector_size=cfg.get("vector_size", 100),
            epochs=cfg.get("epochs", 40),
            min_count=cfg.get("min_count", 2),
            window=cfg.get("window", 5),
            dm=cfg.get("dm", 0),
            dbow_words=cfg.get("dbow_words", 0),
            workers=cfg.get("workers", 1),
        )

    elif text_process == "emotion":
        return vectorizar_emociones(
            text_cols,
            cache_dir=cfg.get("cache_dir", "cache"),
            chunk_palabras=cfg.get("chunk_palabras", CHUNK_PALABRAS),
            max_chunks=cfg.get("max_chunks", max_chunks_defecto),
        )

    else:
        raise ValueError(f"text_process no soportado: {text_process}")

# Núcleo común de las aproximaciones densas basadas en un modelo preentrenado:
# trocear -> codificar -> promediar los trozos de cada instancia -> (normalizar) -> caché
def _vectorizar_por_trozos(textos, firma, prefijo, cache_dir, chunk_palabras, max_chunks,
                           codificar, normalizar=True):
    hash_datos = hashlib.md5((firma + "\x1f".join(textos)).encode("utf-8")).hexdigest()[:12]

    os.makedirs(cache_dir, exist_ok=True)
    ruta_cache = os.path.join(cache_dir, f"{prefijo}_{hash_datos}.npy")
    if os.path.exists(ruta_cache):
        print(f"Cargando vectores desde caché: {ruta_cache}")
        return np.load(ruta_cache, mmap_mode='r')

    todos_chunks, inicios = [], []
    for t in textos:
        inicios.append(len(todos_chunks))
        todos_chunks.extend(trocear(t, chunk_palabras, max_chunks))
    fines = inicios[1:] + [len(todos_chunks)]
    print(f"{len(textos)} mensajes -> {len(todos_chunks)} trozos")

    emb = codificar(todos_chunks)

    X = np.vstack([emb[i:f].mean(axis=0) for i, f in zip(inicios, fines)])
    if normalizar:
        normas = np.linalg.norm(X, axis=1, keepdims=True)
        normas[normas == 0] = 1
        X = X / normas

    np.save(ruta_cache, X)
    print(f"Vectores guardados en {ruta_cache} con forma {X.shape}")
    return X

# Embeddings contextuales (sentence-transformers) con caché
def vectorizar_embeddings(text_cols, cache_dir="cache",
                          chunk_palabras=CHUNK_PALABRAS, max_chunks=MAX_CHUNKS):
    textos = [str(t) for t in text_cols]   # un mensaje por elemento
    # La firma NO ha cambiado: las cachés de embeddings que ya tengas se siguen reutilizando
    firma = f"{MODELO_EMBEDDINGS}|msg|{chunk_palabras}|{max_chunks}||"

    def codificar(chunks):
        from sentence_transformers import SentenceTransformer
        print("Cargando modelo...")
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
        print("Vectorizando textos...")
        return modelo.encode(chunks, batch_size=64,
                             show_progress_bar=True, normalize_embeddings=True)

    return _vectorizar_por_trozos(textos, firma, "vectores", cache_dir,
                                  chunk_palabras, max_chunks, codificar, normalizar=True)

# Embedding emocional (pysentimiento): cada instancia -> vector de 7 probabilidades de emoción
# (orden alfabético: anger, disgust, fear, joy, others, sadness, surprise)
def vectorizar_emociones(text_cols, cache_dir="cache",
                         chunk_palabras=CHUNK_PALABRAS, max_chunks=MAX_CHUNKS):
    textos = [str(t) for t in text_cols]
    firma = f"{MODELO_EMOCIONES}|emo|{chunk_palabras}|{max_chunks}||"

    def codificar(chunks):
        # Silenciar avisos y barras de progreso internas de Hugging Face (una por llamada a predict)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from pysentimiento import create_analyzer
        try:
            import datasets
            datasets.disable_progress_bars()
        except Exception:
            pass

        print("Cargando analizador de emociones...")
        analizador = create_analyzer(task="emotion", lang="en", batch_size=64)

        # pysentimiento rellena (padding) cada lote hasta el texto más largo y NO ordena por longitud:
        # mezclar textos cortos y largos desperdicia cálculo. Se procesan de más largo a más corto
        # (así, además, la estimación de tiempo restante es pesimista, no optimista) y luego se
        # restaura el orden original.
        orden = np.argsort([-len(c) for c in chunks], kind="stable")
        salida = [None] * len(chunks)
        bloque = 512   # cada predict() crea un Dataset y un Trainer: mejor pocas llamadas grandes
        t0 = time.time()
        for ini in range(0, len(orden), bloque):
            idx = orden[ini:ini + bloque]
            for j, r in zip(idx, analizador.predict([chunks[j] for j in idx])):
                salida[j] = [r.probas[k] for k in sorted(r.probas)]
            hechos = ini + len(idx)
            t = time.time() - t0
            resto = t / hechos * (len(chunks) - hechos)
            print(f"  emociones: {hechos}/{len(chunks)} trozos | {t / 60:.1f} min | "
                  f"quedan como mucho ~{resto / 60:.1f} min", flush=True)
        return np.asarray(salida, dtype=np.float32)

    # Sin normalización L2: las probabilidades ya suman 1 y conservan su significado
    return _vectorizar_por_trozos(textos, firma, "emociones", cache_dir,
                                  chunk_palabras, max_chunks, codificar, normalizar=False)

# Doc2Vec (gensim): se ENTRENA con los propios textos (no hay modelo preentrenado)
def _hash_estable(s):
    # El hash por defecto de Python cambia en cada ejecución; con este el entrenamiento es reproducible
    return zlib.crc32(s.encode("utf-8") if isinstance(s, str) else s)

def vectorizar_doc2vec(text_cols, vector_size=100, epochs=40, min_count=2,
                       window=5, dm=0, dbow_words=0, workers=1, seed=42):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.utils import simple_preprocess

    # Tokenización ligera: minúsculas y palabras (sin stemming ni stopwords: Doc2Vec las aprovecha)
    docs = [TaggedDocument(simple_preprocess(str(t), min_len=2, max_len=30), [i])
            for i, t in enumerate(text_cols)]
    print(f"Entrenando Doc2Vec ({'PV-DM' if dm else 'PV-DBOW'}) con {len(docs)} documentos...")

    modelo = Doc2Vec(vector_size=vector_size, min_count=min_count, window=window,
                     dm=dm, dbow_words=0 if dm else dbow_words,   # PV-DBOW: dbow_words=1 entrena también las palabras (~5x más lento, sin mejora apreciable aquí)
                     epochs=epochs, workers=workers, seed=seed, hashfxn=_hash_estable)
    modelo.build_vocab(docs)
    modelo.train(docs, total_examples=modelo.corpus_count, epochs=modelo.epochs)

    X = np.vstack([modelo.dv[i] for i in range(len(docs))])
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    normas[normas == 0] = 1
    X = X / normas
    print(f"Doc2Vec listo: matriz {X.shape} (vocabulario: {len(modelo.wv)} palabras)")
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
    return pd.DataFrame(filas, columns=[col_user, col_label, COL_TEXTO])


def unir_mensajes(crudo):
    """Convierte "['msg1', 'msg2']" en 'msg1 msg2' (un único texto por usuario)."""
    return " ".join(m.strip() for m in parsear_mensajes(crudo))


def preparar_instancias(df, args):
    """Convierte el DataFrame (una fila por usuario) en el DataFrame de instancias
    según args.preprocessing["unidad"]:
      - "usuario": una fila por usuario, con todos sus mensajes unidos en un texto.
      - "mensaje": una fila por mensaje; cada mensaje hereda User y Label de su autor.
    Siempre devuelve columnas [col_user, col_label, COL_TEXTO] y NO modifica df."""
    cfg = args.preprocessing
    unidad = cfg.get("unidad", "usuario")
    col_user = cfg.get("col_user", "User")
    col_post, col_label = args.target, args.label

    if unidad not in UNIDADES:
        raise ValueError(f"unidad no soportada: {unidad!r} (usa una de {UNIDADES})")
    if col_user not in df.columns:           # dataset sin columna de usuario
        df = df.assign(**{col_user: df.index})

    if unidad == "mensaje":
        return explotar_mensajes(df, col_post, col_user, col_label,
                                 cfg.get("min_palabras", 1))

    out = df[[col_user, col_label]].copy()
    out[COL_TEXTO] = [unir_mensajes(p) for p in df[col_post]]
    return out.reset_index(drop=True)


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

#Recursos de NLTK: se preparan UNA vez (antes se descargaban y se recreaban en cada llamada,
#lo que con una fila por mensaje se repetiría miles de veces)
_STEMMER = SnowballStemmer('english')
_STOP_WORDS = None

def _stop_words():
    global _STOP_WORDS
    if _STOP_WORDS is None:
        nltk.download(['stopwords', 'punkt', 'punkt_tab'], quiet=True)
        negaciones = {'no', 'not', 'neither', 'nor', 'none', 'never'}
        _STOP_WORDS = set(stopwords.words('english')) - negaciones
    return _STOP_WORDS

#Limpiar texto (solo para Aproximaciones dispersas)
def clean_text(text):
    stop_words = _stop_words()
    #Texto minúsculas
    text = str(text).lower()
    #Limpieza texto
    limpio = re.sub(r'[^a-záéíóúñ\s]', '', text)
    tokens = word_tokenize(limpio, language='english')
    cleaned = [_STEMMER.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned)