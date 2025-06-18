import csv
import re
import numpy as np
from numpy import array
from pickle import dump
from unicodedata import normalize
from numpy.random import shuffle

def clean_text(text):
    # Normaliza y limpia el texto
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9¿?¡! ]', ' ', text)  # Mantiene signos y espacios
    text = re.sub(r'\s+', ' ', text)
    return text

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    pairs = []
    for line in lines:
        if ',' in line:
            pregunta, respuesta = line.split(',', 1)
            pregunta = clean_text(pregunta)
            respuesta = clean_text(respuesta)
            if pregunta and respuesta:
                pairs.append((pregunta, respuesta))
    return array(pairs)

def save_clean_data(data, filename):
    dump(data, open(filename, 'wb'))
    print(f'Datos guardados: {filename}')

# ========== EJECUCIÓN PRINCIPAL ==========
if __name__ == '__main__':
    dataset = load_dataset('sample_conversations.csv')

    # Barajar aleatoriamente
    shuffle(dataset)

    # Dividir en entrenamiento y prueba
    train, test = dataset[:100], dataset[100:]

    # Guardar los archivos
    save_clean_data(dataset, 'both.pkl')
    save_clean_data(train, 'train.pkl')
    save_clean_data(test, 'test.pkl')

    # Guardar las preguntas/respuestas para fallback
    qa_pairs = [(q, r) for q, r in dataset]
    save_clean_data(qa_pairs, 'qa_pairs.pkl')

