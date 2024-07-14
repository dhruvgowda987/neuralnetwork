import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
import csv, random, numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, RandomFlip, RandomTranslation, Input, Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

def model(load, shape, checkpoint=None):
    if load and checkpoint: return load_model(checkpoint)

    cl, dl = [32, 32, 32, 64, 128], [1024, 512]

    model = Sequential()
    for layers in cl:
        input_layer = Input(shape=shape)
        model.add(Conv2D(layers, (3, 3), activation='elu')) # learn elu
        model.add(MaxPooling2D())
    model.add(Flatten())
    for layers in dl:
        model.add(Dense(layers, activation='elu'))
        model.add(Dropout(.4)) # prevents overfitting by randomly deactivating 40% of the neurons during training
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer="adam") # research
    return model

def get_angles(data):
    frames, angles = [], []
    str_offset = 0.2
    with open(data) as fin:
        for _, l_img, r_img, str_angle, _, _, speed in csv.reader(fin):
            frames += [l_img.strip(), r_img.strip()]
            angles += [float(str_angle) + str_offset, float(str_angle) - str_offset]
    return frames, angles

def data_aug(path, str_angle, augment, shape=(100,100)):
    img = load_img(path, target_size=shape)

    if augment and random.random() > .5: # randomly darkens to generalize for shadows
        x, y = img.size
        x1, y1 = random.randint(0, x), random.randint(0, y)
        x2, y2 = random.randint(x1, x), random.randint(y1, y)

        for x in range(x1, x2):
            for y in range(y1, y2):
                darkened = tuple([int(x * .5) for x in img.getpixel((x, y))])
                img.putpixel((x, y), darkened)

    img = img_to_array(img) # numpy array

    if augment:
        rand_shift = RandomTranslation(height_factor=0.15, width_factor=0)
        img = rand_shift(img) # vertical shift
        if random.random() > .5:
            rand_flip = RandomFlip(mode="horizontal")
            img = rand_flip(img) # horizontal flip
            str_angle = -str_angle

    img = tf.cast(img / 255. - 0.5, dtype=tf.float32) # needed to continue processing the numpy array
    return img, str_angle

def generator(size, frames, angles):
    while True:
        new_frames, new_angles = [], []
        for i in range(size):
            index = random.randint(0, len(frames) - 1)
            str_angle = angles[index]
            img, str_angle = data_aug(frames[index], str_angle, augment=True)
            new_frames.append(img)
            new_angles.append(str_angle)
        yield np.array(new_frames), np.array(new_angles)
 
def train():
    nn = model(load=False, shape=(100, 100, 3))
    frames, angles = get_angles('data\driving_log.csv')
    nn.fit(generator(64, frames, angles), steps_per_epoch=int(len(frames) / 256), epochs=2)
    print("before")
    nn.save('saved/net2.keras')
    print("after")

if __name__ == '__main__':
    train()



