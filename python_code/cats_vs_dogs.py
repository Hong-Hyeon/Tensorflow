import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16

def preprocess(data):
    x=data['image']/255
    y=data['label']
    x=tf.image.resize(x,size=(224,224))
    return x,y


def solution_model():
    dataset_name = 'cats_vs_dogs'

    train_dataset = tfds.load(dataset_name,split='train[:80%]')
    valid_dataset = tfds.load(dataset_name,split='train[80%:]')

    batch_size = 32

    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    transfer_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
    transfer_model.trainable=False

    model = Sequential([
        transfer_model,
        Flatten(),
        Dense(450, activation='relu'),
        Dense(225, activation='relu'),
        Dense(117, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

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

if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-cats-vs-dogs.h5")
