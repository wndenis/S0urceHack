import numpy as np
import cv2
from os import listdir, rmdir
from os.path import isfile, join, isdir
import random
from Config import path_to_dataset
from time import time

from keras.models import Sequential, load_model, save_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard

labels_str = list("23abcdefghijklmnopqrstuvwxyz")
labels_onehot = [str(bin(2 ** i))[2:] for i in range(len(labels_str))]
labels_onehot = ['0' * (len(labels_str) - len(elem)) + elem for elem in labels_onehot]
labels_onehot = [np.array(list(map(float, elem))) for elem in labels_onehot]

if __name__ == "__main__":
    # read full dataset
    DATASET = []
    directories = [d for d in listdir(path_to_dataset) if isdir(join(path_to_dataset, d))]
    for directory in directories:
        if directory in labels_str:
            oh_label = labels_onehot[labels_str.index(directory)]
            local_dir = path_to_dataset + directory + "/"
            files = sorted([f for f in listdir(local_dir) if isfile(join(local_dir, f))])
            files = [f for f in files if f[-4:] == ".png"]

            # label each img
            for file in files:
                img = cv2.imread(local_dir + file, cv2.IMREAD_GRAYSCALE)
                DATASET.append([np.array(img), oh_label])

    print("Loaded %s images" % len(DATASET))
    random.shuffle(DATASET)

    tr_img_data = np.array([i[0] for i in DATASET]).reshape(-1, 16, 32, 1)
    tr_lbl_data = np.array([i[1] for i in DATASET])

    model = Sequential()

    model.add(InputLayer(input_shape=[16, 32, 1]))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=80, kernel_size=1, strides=1, activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(len(labels_str), activation='softmax'))
    optimizer = Adam(lr=1e-5)

    print("Start training...")
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit(x=tr_img_data, y=tr_lbl_data, epochs=600, batch_size=5, callbacks=[tensorboard])
    # model.fit(x=tr_img_data, y=tr_lbl_data, epochs=700, batch_size=32)
    model.summary()
    print("Saving model to disc...")
    save_model(model, "model.h5")
    print("Done.")


class Thinker:
    def __init__(self):
        self.model = load_model("model.h5")

    def think(self, img, top=2):
        formatted_img = np.array(img).reshape(-1, 16, 32, 1)
        # cv2.imshow(formatted_img)
        # cv2.waitKey()
        prediction = list(self.model.predict(formatted_img)[0])
        sortedpred = sorted(prediction, reverse=True)[:top]
        candidates = []
        prev_pred = 1e0
        for pred in sortedpred:
            if pred <= prev_pred:
                d = prev_pred - pred
                p = prev_pred * 0.9
                # if d > p:
                #     break
            prev_pred = pred
            i = prediction.index(pred)
            candidates.append(list(reversed(labels_str))[i])
        return candidates

