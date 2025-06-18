from pickle import load
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

def create_tokenizer(lines):
    tokenizer = Tokenizer(filters='', oov_token='<OOV>')
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines) + 1  # +1 para token de inicio/fin

def encode_sequences(tokenizer, length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units=256):
    model = Sequential([
        Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True),
        GRU(n_units),
        RepeatVector(tar_timesteps),
        GRU(n_units, return_sequences=True),
        TimeDistributed(Dense(tar_vocab, activation='softmax'))
    ])
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    return model

# Carga de datos
dataset = load_clean_sentences('both.pkl')
train = load_clean_sentences('train.pkl')
test = load_clean_sentences('test.pkl')

# Prepara tokenizer con todas las preguntas y respuestas
all_text = array([d[0] for d in dataset] + [d[1] for d in dataset])
tokenizer = create_tokenizer(all_text)
vocab_size = len(tokenizer.word_index) + 1

# Longitudes máximas
max_question_len = max_length([d[0] for d in dataset])
max_answer_len = max_length([d[1] for d in dataset])

# Prepara datos de entrenamiento
trainX = encode_sequences(tokenizer, max_question_len, [d[0] for d in train])
trainY = encode_sequences(tokenizer, max_answer_len, [d[1] for d in train])

# Prepara datos de test
testX = encode_sequences(tokenizer, max_question_len, [d[0] for d in test])
testY = encode_sequences(tokenizer, max_answer_len, [d[1] for d in test])

# Define modelo
model = define_model(vocab_size, vocab_size, max_question_len, max_answer_len)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint('model.h5', save_best_only=True)
]

# Entrenamiento
history = model.fit(
    trainX, trainY,
    epochs=300,
    batch_size=16,
    validation_data=(testX, testY),
    callbacks=callbacks,
    verbose=1
)

# Guarda tokenizer y configuraciones
# Guarda tokenizer y configuraciones
from pickle import dump
dump(tokenizer, open('tokenizer.pkl', 'wb'))
dump({'max_question_len': max_question_len, 'max_answer_len': max_answer_len}, 
     open('config.pkl', 'wb'))
