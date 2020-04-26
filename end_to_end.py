# coding: UTF-8
import numpy as np
import csv
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard

# データの読み込み
path = "./data/"

imgs, left_imgs, right_imgs, angles = [], [], [], []
# 角度の補正
correction = 0.2

# ヒストグラム平坦化
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 16))

with open(path + "/driving_log.csv", "r") as f:
	reader = csv.reader(f)
	for row in reader: 
		row[0] = row[0].split('/')[-1]
		row[1] = row[1].split('/')[-1]
		row[2] = row[2].split('/')[-1]
		# 正面からの写真
		center_image = cv2.imread(path + '/IMG/' + row[0])
		center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
		center_image[:, :, 0] = clahe.apply(center_image[:, :, 0])
		imgs.append(center_image)
		angles.append(float(row[3]))
		# 左からの写真
		left_image = cv2.imread(path + '/IMG/' + row[1])
		left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
		left_image[:, :, 0] = clahe.apply(left_image[:, :, 0])
		imgs.append(left_image)
		angles.append(float(row[3]) + correction)
		# 右からの写真
		right_image = cv2.imread(path + '/IMG/' + row[2])
		right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
		right_image[:, :, 0] = clahe.apply(right_image[:, :, 0])
		imgs.append(right_image)
		angles.append(float(row[3]) - correction)
		# 左右反転した写真
		center_flipped_image = cv2.flip(center_image, 1) 
		imgs.append(center_flipped_image)
		angles.append(float(row[3])*(-1))

X_train = np.array(imgs).astype('float32')
Y_train = np.array(angles)

# シミュレータで取れる画像は160*320
# print(X_train.shape)


#画像のサイズ
img_rows, img_cols = 160, 320


#学習パラメータ
batch_size = 128
nb_epoch = 10


#チャンネル数
nb_filters = [24, 36, 48, 64, 64]
nb_conv = [5, 3]


#ネットワークの構成
model = Sequential()

# トリミング
model.add(Cropping2D(cropping = ((50, 20),(10, 10)), input_shape = (img_rows, img_cols, 3)))
# cropping = ((top, bottom),(left, right))

# YUV形式のデータを正規化
model.add(Lambda(lambda x: x/255.0))

model.add(Conv2D(nb_filters[0], (nb_conv[0], nb_conv[0]), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters[1], (nb_conv[0], nb_conv[0]), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters[2], (nb_conv[0], nb_conv[0]), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters[3], (nb_conv[1], nb_conv[1]), strides = (1,1), padding = 'valid'))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters[4], (nb_conv[1], nb_conv[1]), strides = (1,1), padding = 'valid'))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')


#学習
tb_cb = TensorBoard(log_dir="./LOG", histogram_freq=1)
cbks = [tb_cb]
model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_split = 0.2, shuffle = True, callbacks = cbks)

# model.fit(X_train, Y_train, batch_size = batch_size, epochs = nb_epoch, verbose = 1, validation_split = 0.2)
model.save('model.h5')

#推論

