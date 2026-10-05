import nltk
import re
from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#Preproceso de los datos
def preproceso(text_cols, args):
    #Limpieza de texto
    text_cols = text_cols.apply(clean_text)
    #Vectorización
    text_process = args.preprocessing.get("text_process", "none")
    if text_process == "tf-idf":
        vec = TfidfVectorizer(max_features=5000,
                              ngram_range=(1, 3),
                              min_df=2,
                              max_df=0.9,
                              sublinear_tf=True)
    else:
        print("Otro preproceso")

    matriz = vec.fit_transform(text_cols)
    return matriz

#Limpiar texto
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
