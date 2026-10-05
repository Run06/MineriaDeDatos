import argparse
import os
import json
import nltk
import numpy as np
import pandas as pd
from colorama import Fore

from preproceso import preproceso


def parse_args():
    parser = argparse.ArgumentParser(description="Clasificación no supervisada K-Means")
    #parser.add_argument("-f", "--file", required=True, help="Archivo CSV de entrada")
    #parser.add_argument("-c", "--cpu", default=-1, type=int)
    #parser.add_argument("-l", "--column", required=True, help="Columna de la tabla a clasificar")
    #parser.add_argument("-s", "--sample", type=float, default=1.0, help="Porcentaje de datos a usar (0.0 a 1.0)")

    args = parser.parse_args()

    if os.path.exists('clasificador.json'):
        with open('clasificador.json') as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)
    return args

if __name__ == "__main__":
    args = parse_args()
    #cargar csv
    #raw_data = pd.read_csv(parse_args().file)
    raw_data = pd.read_csv('500_Reddit_users_posts_labels.csv')
    #división train test, 0.7
    p_train = 0.7
    raw_data['is_train'] = np.random.uniform(0, 1, len(raw_data)) <= p_train
    train, test = raw_data[raw_data['is_train'] == True], raw_data[raw_data['is_train'] == False]
    print(Fore.MAGENTA + f"Preprocesando datos del train..." + Fore.RESET)
    #train_prep = preproceso(train[parse_args().column], args)
    train_prep = preproceso(train['Post'], args)
    print(Fore.GREEN + f"Datos preprocesados!" + Fore.RESET)