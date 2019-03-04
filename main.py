import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras import backend as K, Model, Sequential
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator

batch_size = 32
epochs = 10
height = 128
width = 128

train_data_dir = "input/train"
validation_data_dir = "input/validation"
test_data_dir = "input/test"
model_path = "data/car_classifier_model.h5"


def create_model(classes_count):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(width, height, 3))

    for layer in base_model.layers[:15]:
        layer.trainable = False

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten())
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(classes_count, activation="softmax"))

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model = Model(input=base_model.input, output=top_model(base_model.output))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(base_model.summary())
    print(top_model.summary())
    print(model.summary())

    return model


def build_callbacks():
    checkpoint_path = "./data/checkpoints/image_classifier-epoch{epoch:03d}-" \
                      "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=100)
    tb_callback = TensorBoard(os.path.join('data', 'logs'))

    return [checkpoint_callback, early_stopping, tb_callback]


def predict():
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode="categorical"
    )
    model = create_model(len(train_generator.class_indices))
    model.load_weights(model_path)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(width, height),
        batch_size=1,
        class_mode="categorical"
    )

    predictions = model.predict_generator(test_generator, steps=len(test_generator.filenames))
    predicted_class_indices = predictions.argmax(axis=1)
    label_map = (train_generator.class_indices)
    label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
    predicted_class_indices = [label_map[k] for k in predicted_class_indices]
    for i in range(len(test_generator.filenames)):
        img = load_img(path=os.path.join("input", "test", test_generator.filenames[i]))
        img = img_to_array(img)
        plt.imshow(np.uint8(img))
        plt.title(predicted_class_indices[i])
        plt.show()


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

    if (Path(model_path).exists()):
        print("Loaded previous saved model")
        model.load_weights(model_path)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks=build_callbacks()
    )

train()
predict()
