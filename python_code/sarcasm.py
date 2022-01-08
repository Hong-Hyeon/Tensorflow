# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# NLP QUESTION
#
# For this task you will build a classifier for the sarcasm dataset
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown
# It will be tested against a number of sentences that the network hasn't previously seen
# And you will be scored on whether sarcasm was correctly detected in those sentences

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 4 - sarcasm
# val_loss: 0.3650
# val_acc: 0.83
# =================================================== #
# =================================================== #


import json
import tensorflow as tf
import numpy as np
import urllib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    with open('sarcasm.json') as f:
        datas = json.load(f)

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    for data in datas:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOv>')

    # 단어사전 만들기
    tokenizer.fit_on_texts(train_sentences)

    for key,value in tokenizer.word_index.items():
        print('{}\t=======>\t{}'.format(key,value))
        if value == 25:
            break

    word_index = tokenizer.word_index

    train_sequences=tokenizer.texts_to_sequences(train_sentences)
    validation_sequences=tokenizer.texts_to_sequences(validation_sentences)

    train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    # model_1 valloss 0.368
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        # many to one방식, many to many는 return_sequences=True를 넣어주면됨.
        Bidirectional(LSTM(64)),
        Dense(32,activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # model_2 valloss 0.376
    # model = Sequential([
    #     Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     Bidirectional(LSTM(64, return_sequences=True)),
    #     # many to one방식, many to many는 return_sequences=True를 넣어주면됨.
    #     Bidirectional(LSTM(128)),
    #     Dense(32,activation='relu'),
    #     Dense(16, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_3 valloss 0.367
    # model = Sequential([
    #     Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     # Bidirectional(LSTM(32, return_sequences=True)),
    #     Bidirectional(LSTM(64, return_sequences=True)),
    #     # many to one방식, many to many는 return_sequences=True를 넣어주면됨.
    #     Bidirectional(LSTM(128)),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_4 valloss 0.366
    # model = Sequential([
    #     Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     Bidirectional(LSTM(64, return_sequences=True)),
    #     Dropout(0.25),
    #     # many to one방식, many to many는 return_sequences=True를 넣어주면됨.
    #     Bidirectional(LSTM(128)),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_5 valloss 0.3677
    # model = Sequential([
    #     Embedding(vocab_size, embedding_dim, input_length=max_length),
    #     Bidirectional(LSTM(64, return_sequences=True)),
    #     # many to one방식, many to many는 return_sequences=True를 넣어주면됨.
    #     Bidirectional(LSTM(128)),
    #     Dense(1, activation='sigmoid')
    # ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    check_path = 'check.ckpt'
    checkpoint = ModelCheckpoint(filepath=check_path,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)

    model.fit(train_padded, train_labels,
              validation_data=(validation_padded, validation_labels),
              epochs=10, callbacks=[checkpoint])

    model.load_weights(check_path)
    print(model.evaluate(validation_padded, validation_labels))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF4-sarcasm.h5")