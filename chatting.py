from pickle import load
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from difflib import SequenceMatcher
import gradio as gr

# ========== CARGA DE MODELO Y CONFIGURACIÓN ==========
def load_assets():
    model = load_model('model.h5')
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    config = load(open('config.pkl', 'rb'))
    contextual_memory = load(open('qa_pairs.pkl', 'rb'))  # <-- renombrado
    return model, tokenizer, config, contextual_memory

# ========== PREDICCIÓN CON EL MODELO PRINCIPAL ==========
def generate_response(input_text, model, tokenizer, config):
    input_text = input_text.lower().strip()
    if not input_text.endswith('?'):
        input_text += '?'

    sequence = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(sequence, maxlen=config['max_question_len'], padding='post')
    prediction = model.predict(padded, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]

    words = []
    for i in integers:
        word = tokenizer.index_word.get(i, '')
        if word in ['', '<OOV>']:
            continue
        words.append(word)
        if word == '?':
            break

    return ' '.join(words).capitalize()

# ========== MÓDULO DE MEMORIA CONTEXTUAL ==========
def compute_relevance(a, b):
    return SequenceMatcher(None, a, b).ratio()

def neural_memory_query(user_input, contextual_memory):
    user_input = user_input.lower().strip()
    best_match = max(contextual_memory, key=lambda pair: compute_relevance(user_input, pair[0]))
    relevance_score = compute_relevance(user_input, best_match[0])
    
    if relevance_score > 0.5:  # Umbral ajustable
        return best_match[1]
    return None

# ========== INTERFAZ ==========
model, tokenizer, config, contextual_memory = load_assets()

def chat_interface(message, history):
    respuesta = generate_response(message, model, tokenizer, config)

    if not respuesta or len(respuesta.split()) < 3:
        fallback = neural_memory_query(message, contextual_memory)
        return fallback if fallback else "Lo siento, no encontré una respuesta clara."

    return respuesta

# ========== LANZAR ==========
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🤖 Chaski UMSS - Asistente Académico",
    description="Pregúntame sobre la UMSS, Ingeniería de Sistemas, trámites y requisitos"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
