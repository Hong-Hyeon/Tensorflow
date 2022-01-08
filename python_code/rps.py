# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this task you will build a classifier for Rock-Paper-Scissors
# based on the rps dataset.
#
# IMPORTANT: Your final layer should be as shown, do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail.
#
# NOTE THAT THIS IS UNLABELLED DATA.
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - rps
# val_loss: 0.0871
# val_acc: 0.97
# =================================================== #
# =================================================== #

import urllib.request
import zipfile
import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    TRAINING_DIR = "tmp/rps/"

    # model_valloss0.036
    training_datagen = ImageDataGenerator(rescale=1/255,
                                          rotation_range=20,
                                          # width_shift_range=10,
                                          # height_shift_range=10,
                                          # shear_range=0.5,
                                          zoom_range=0.3,
                                          # horizontal_flip=True,
                                          fill_mode='nearest',
                                          validation_split=0.2)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                             target_size=(150,150),
                                                             batch_size=128,
                                                             class_mode='categorical',
                                                             subset='training')

    valid_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                             target_size=(150,150),
                                                             batch_size=128,
                                                             class_mode='categorical',
                                                             subset='validation')

    model = Sequential([
        Conv2D(64,(3,3),input_shape=(150,150,3),activation='relu'),
        MaxPool2D(3,3),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(3,3),
        Conv2D(128,(3,3), activation='relu'),
        MaxPool2D(3,3),
        Flatten(),
        Dense(1028, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # model_valloss0.10
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=20,
    #                                       # width_shift_range=10,
    #                                       # height_shift_range=10,
    #                                       shear_range=0.3,
    #                                       zoom_range=0.3,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest',
    #                                       validation_split=0.2)
    #
    # train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                          target_size=(150,150),
    #                                                          batch_size=128,
    #                                                          class_mode='categorical',
    #                                                          subset='training')
    #
    # valid_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                          target_size=(150,150),
    #                                                          batch_size=128,
    #                                                          class_mode='categorical',
    #                                                          subset='validation')
    #
    # model = Sequential([
    #     Conv2D(64,(3,3),input_shape=(150,150,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Conv2D(64,(3,3),activation='relu'),
    #     MaxPool2D(3,3),
    #     Conv2D(128,(3,3), activation='relu'),
    #     MaxPool2D(3,3),
    #     Flatten(),
    #     Dense(1028, activation='relu'),
    #     Dense(512, activation='relu'),
    #     Dense(256, activation='relu'),
    #     Dense(3, activation='softmax')
    # ])

    # model_valloss0.09
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=20,
    #                                       # width_shift_range=10,
    #                                       # height_shift_range=10,
    #                                       shear_range=0.2,
    #                                       zoom_range=0.2,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest',
    #                                       validation_split=0.2)

    # model_valloss0.02
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=20,
    #                                       # width_shift_range=10,
    #                                       # height_shift_range=10,
    #                                       # shear_range=0.2,
    #                                       zoom_range=0.2,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest',
    #                                       validation_split=0.2)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

    check_path = "TF3-rps.h5"
    checkpoint=ModelCheckpoint(filepath=check_path,
                               save_weights_only=True,
                               save_best_only=True,
                               monitor='val_loss',
                               verbose=1)

    model.fit(train_generator, validation_data=(valid_generator),
              callbacks=[checkpoint], epochs=50)
    print(model.evaluate(valid_generator))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.load_weights("TF3-rps.h5")