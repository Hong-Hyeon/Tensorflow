# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# This task requires you to create a classifier for horses or humans using
# the provided dataset.
#
# Please make sure your final layer has 2 neurons, activated by softmax
# as shown. Do not change the provided output layer, or tests may fail.
#
# IMPORTANT: Please note that the test uses images that are 300x300 with
# 3 bytes color depth so be sure to design your input layer to accept
# these, or the tests will fail.
#

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - Horses Or Humans type A
# val_loss: 0.028
# val_acc: 0.98
# =================================================== #
# =================================================== #


import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def preprocess(data):
    x=data['image']/255
    y=data['label']
    x=tf.image.resize(x,size=(300,300))
    return x,y


def solution_model():
    dataset_name = 'horses_or_humans'

    train_dataset = tfds.load(dataset_name,split='train[:80%]')
    valid_dataset = tfds.load(dataset_name,split='train[80%:]')

    batch_size = 32

    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    # model_1 valloss 0.00024
    model = Sequential([
        Conv2D(64,(3,3),input_shape=(300,300,3),activation='relu'),
        MaxPool2D(3,3),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(3,3),
        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(3,3),
        Flatten(),
        Dropout(0.4),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])

    # model_2 valloss 0.0017
    # model = Sequential([
    #     Conv2D(64,(3,3),input_shape=(300,300,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Conv2D(64,(3,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Conv2D(128,(3,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dense(256, activation='relu'),
    #     Dense(128, activation='relu'),
    #     Dense(2, activation='softmax')
    # ])

    # model_3 valloss 0.00029(batch_size = 30)
    # model = Sequential([
    #     Conv2D(64,(3,3),input_shape=(300,300,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Conv2D(64,(3,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dense(256, activation='relu'),
    #     Dense(128, activation='relu'),
    #     Dense(2, activation='softmax')
    # ])

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

    check_path='check.ckpt'
    checkpoint = ModelCheckpoint(filepath=check_path,
                                save_weights_only=True,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)

    model.fit(train_data,validation_data=(valid_data), callbacks=[checkpoint], epochs=20)

    model.load_weights(check_path)

    print(model.evaluate(valid_data))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-A.h5")