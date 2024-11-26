import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

phoneme_data = pd.read_csv('phoneme_data.csv')

phoneme_data['phoneme'] = phoneme_data['phoneme'].apply(lambda x: x.split())

flat_phonemes = [phoneme for sublist in phoneme_data['phoneme'] for phoneme in sublist]

phoneme_encoder = LabelEncoder()
phoneme_encoder.fit(flat_phonemes + ['<unk>'])
word_encoder = LabelEncoder()
word_encoder.fit(phoneme_data['word'].tolist() + ['<start>', '<unk>'])

phoneme_data['encoded_phonemes'] = phoneme_data['phoneme'].apply(lambda x: phoneme_encoder.transform(x))
phoneme_data['encoded_words'] = phoneme_data['word'].apply(lambda x: word_encoder.transform([x])[0])

max_phoneme_len = max(phoneme_data['encoded_phonemes'].apply(len))
X = pad_sequences(phoneme_data['encoded_phonemes'], maxlen=max_phoneme_len, padding='post')

y = phoneme_data['encoded_words']

input_phonemes = Input(shape=(max_phoneme_len,))
embedding_phonemes = Embedding(input_dim=len(phoneme_encoder.classes_), output_dim=128)(input_phonemes)

encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(embedding_phonemes)

decoder_inputs = Input(shape=(1,))
embedding_words = Embedding(input_dim=len(word_encoder.classes_), output_dim=128)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embedding_words, initial_state=[state_h, state_c])

decoder_dense = Dense(len(word_encoder.classes_), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([input_phonemes, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

decoder_input_data = np.expand_dims(y, axis=1)
decoder_output_data = np.expand_dims(y, axis=-1)

model.fit([X, decoder_input_data], decoder_output_data, epochs=1000, batch_size=64, validation_split=0.2)

def predict_word_sequence(phoneme_sequence):
    encoded_phonemes = [phoneme_encoder.transform([phoneme])[0] if phoneme in phoneme_encoder.classes_ else phoneme_encoder.transform(['<unk>'])[0] for phoneme in phoneme_sequence]
    encoded_phonemes = pad_sequences([encoded_phonemes], maxlen=max_phoneme_len, padding='post')

    decoder_input = np.array([[word_encoder.transform(['<start>'])[0]]])

    predicted_words = []
    for _ in range(len(phoneme_sequence)):
        output = model.predict([encoded_phonemes, decoder_input])
        predicted_word = np.argmax(output[0, -1, :])

        predicted_words.append(word_encoder.inverse_transform([predicted_word])[0])
        decoder_input = np.array([[predicted_word]])

    return predicted_words

phoneme_sequence = ["DH", "EH", "R", "DH", "EH", "R"]
predicted_words = predict_word_sequence(phoneme_sequence)
print(predicted_words)
