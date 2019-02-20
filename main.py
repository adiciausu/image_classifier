import os

import tensorflow
from keras import backend as K
from keras import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator

batch_size = 128
epochs = 30
height = int(128 * 1.5)
width = 128

train_data_dir = "input/train"
validation_data_dir = "input/validation"
model_path = "car_classifier_model.h5"


def create_model(classes_count):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, width, height)
    else:
        input_shape = (width, height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_count, activation='softmax'))

    return model

def build_callbacks():
    checkpoint_path = "./data/checkpoints/image_classifier-epoch{epoch:03d}-" \
                      "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=100)
    tb_callback = TensorBoard(os.path.join('data', 'logs'))

    return [checkpoint_callback, early_stopping, tb_callback]


def predict():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical"
    )
    model = create_model(len(train_generator.class_indices))
    model.load_weights(model_path)


def train():
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists(os.path.join('data', 'logs')):
        os.mkdir(os.path.join('data', 'logs'))
    if not os.path.exists(os.path.join('data', 'checkpoints')):
        os.mkdir(os.path.join('data', 'checkpoints'))


    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tensorflow.Session(config=config))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical"
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical"
    )

    model = create_model(len(train_generator.class_indices))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks = build_callbacks()
    )

    model.evaluate_generator(generator=validation_generator)
    model.save_weights(model_path)


train()
predict()
