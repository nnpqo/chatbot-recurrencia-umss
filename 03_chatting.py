from pickle import load
from numpy import array, argmax
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import gradio as gr

# ========== FUNCIONES BASE ==========

def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

def create_tokenizer(lines):
    tokenizer = Tokenizer(char_level=False)
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = []
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def generate_response(message, history):
    # Tokenizamos la entrada del usuario
    X = all_tokenizer.texts_to_sequences([message])
    X = pad_sequences(X, maxlen=all_length, padding='post')

    # Predecimos la respuesta
    X = X.reshape((1, X.shape[1]))
    respuesta = predict_sequence(model, all_tokenizer, X)
    return respuesta

# ========== CARGA DEL MODELO Y TOKENIZER ==========

dataset = load_clean_sentences("both.pkl")
dataset = dataset.reshape(-1, 1)

all_tokenizer = create_tokenizer(dataset[:, 0])
all_vocab_size = len(all_tokenizer.word_index) + 1
all_length = max_length(dataset[:, 0])
model = load_model("model1.h5")

# ========== CHAT CON GRADIO (ESTILO CHATBOT) ==========

chat = gr.ChatInterface(fn=generate_response, title="🤖 Chatbot Académico UMSS")

if __name__ == '__main__':
    chat.launch(server_name="0.0.0.0", server_port=7860)

