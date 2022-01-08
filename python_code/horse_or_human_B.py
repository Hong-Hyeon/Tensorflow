# Question
#
# This task requires you to create a classifier for horses or humans using
# the provided data. Please make sure your final layer is a 1 neuron, activated by sigmoid as shown.
# Please note that the test will use images that are 300x300 with 3 bytes color depth so be sure to design your neural network accordingly

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - Horses Or Humans (Type B)
# val_loss: 0.51 (더 낮아도 안 좋고, 높아도 안 좋음!)
# val_acc: 관계없음
# =================================================== #
# =================================================== #



import tensorflow as tf
import urllib
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('horse-or-human/')
    zip_ref.close()

    urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
    local_zip = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('validation-horse-or-human/')
    zip_ref.close()

    TRAINING_DIR = 'horse-or-human/'
    VALIDATION_DIR = 'validation-horse-or-human'

    # model_1 valloss 0.4536
    training_datagen = ImageDataGenerator(rescale=1/255,
                                          # rotation_range=50,
                                          # width_shift_range=0.4,
                                          # height_shift_range=0.4,
                                          # shear_range=0.4,
                                          zoom_range=0.2,
                                          # horizontal_flip=True,
                                          fill_mode='nearest'
                                          )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                           batch_size=50,
                                                           target_size=(300, 300),
                                                           class_mode='binary'
                                                           )
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=20,
                                                                  target_size=(300, 300),
                                                                  class_mode='binary'
                                                                  )

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(200, activation='relu'),
        Dense(40, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # model_2 valloss 0.54
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=20,
    #                                       # width_shift_range=0.4,
    #                                       # height_shift_range=0.4,
    #                                       # shear_range=0.4,
    #                                       zoom_range=0.2,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest'
    #                                       )
    # validation_datagen = ImageDataGenerator(rescale=1/255)
    #
    # train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                        batch_size=50,
    #                                                        target_size=(300, 300),
    #                                                        class_mode='binary'
    #                                                        )
    # validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
    #                                                               batch_size=20,
    #                                                               target_size=(300, 300),
    #                                                               class_mode='binary'
    #                                                               )
    #
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    #     MaxPool2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dense(200, activation='relu'),
    #     Dense(40, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_3 valloss 0.53
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=30,
    #                                       # width_shift_range=0.4,
    #                                       # height_shift_range=0.4,
    #                                       # shear_range=0.4,
    #                                       zoom_range=0.2,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest'
    #                                       )
    # validation_datagen = ImageDataGenerator(rescale=1/255)
    #
    # train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                        batch_size=50,
    #                                                        target_size=(300, 300),
    #                                                        class_mode='binary'
    #                                                        )
    # validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
    #                                                               batch_size=20,
    #                                                               target_size=(300, 300),
    #                                                               class_mode='binary'
    #                                                               )
    #
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    #     MaxPool2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dense(200, activation='relu'),
    #     Dense(40, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_3 valloss 0.52
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       rotation_range=20,
    #                                       # width_shift_range=0.4,
    #                                       # height_shift_range=0.4,
    #                                       # shear_range=0.4,
    #                                       zoom_range=0.3,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest'
    #                                       )
    # validation_datagen = ImageDataGenerator(rescale=1/255)
    #
    # train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                        batch_size=50,
    #                                                        target_size=(300, 300),
    #                                                        class_mode='binary'
    #                                                        )
    # validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
    #                                                               batch_size=20,
    #                                                               target_size=(300, 300),
    #                                                               class_mode='binary'
    #                                                               )
    #
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    #     MaxPool2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dense(200, activation='relu'),
    #     Dense(40, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model_5 valloss 0.50
    # training_datagen = ImageDataGenerator(rescale=1/255,
    #                                       # rotation_range=20,
    #                                       # width_shift_range=0.2,
    #                                       height_shift_range=0.2,
    #                                       # shear_range=0.4,
    #                                       zoom_range=0.3,
    #                                       # horizontal_flip=True,
    #                                       fill_mode='nearest'
    #                                       )
    # validation_datagen = ImageDataGenerator(rescale=1/255)
    #
    # train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
    #                                                        batch_size=50,
    #                                                        target_size=(300, 300),
    #                                                        class_mode='binary'
    #                                                        )
    # validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
    #                                                               batch_size=20,
    #                                                               target_size=(300, 300),
    #                                                               class_mode='binary'
    #                                                               )
    #
    # model = Sequential([
    #     Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    #     MaxPool2D(2, 2),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPool2D(2, 2),
    #     Flatten(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dense(200, activation='relu'),
    #     Dense(40, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    check_path = 'check.ckpt'
    checkpoint = ModelCheckpoint(filepath=check_path,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_generator, validation_data=(validation_generator), callbacks=[checkpoint], epochs=20)
    model.load_weights(check_path)

    # NOTE: If training is taking a very long time, you should consider setting the batch size appropriately on the generator, and the steps per epoch in the model.fit#
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-B.h5")