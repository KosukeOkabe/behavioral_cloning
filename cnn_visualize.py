# coding: UTF-8

import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model, Model
from keras.preprocessing.image import array_to_img, image

model = load_model("model.h5")

# データのパス
path = "./data/"

# ヒストグラム平坦化
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 16))

# 画像の読み込み
# 1つのディレクトリに画像を入れておく
files = os.listdir(path + "/IMG/")
img_list = [file for file in files if os.path.isfile(os.path.join(path + "/IMG/", file))]

# print(img_list)

# 画像の取得
def get_img(img_path):
	img = cv2.imread(path + "/IMG/" + img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img[:, :, 0] = clahe.apply(img[:, :, 0])
	return img

# Conv2D層からの出力の抽出
def get_feature(layer, img):
	intermediate_layer_model = Model(inputs = model.input, outputs = model.get_layer("conv2d_" + str(layer)).output)
	img = img[np.newaxis, :, :, :]
	y = intermediate_layer_model.predict(img)
	return y

# 画像に変換
def make_img(y):
	feature_imgs = []
	for feature_img in np.split(y, y.shape[3], axis = 3):
		feature_img = np.squeeze(feature_img, axis  = 0)
		feature_img = np.array(image.array_to_img(feature_img))
		feature_img = cv2.resize(feature_img, None, fx = 36.0, fy = 36.0)
		feature_imgs.append(feature_img)
	return feature_imgs

# 全て描画
def imshow_all():
	for img_path in img_list:
		for layer in range(5):
			img = get_img(img_path)
			y = get_feature(layer + 1, img)
			feature_imgs = make_img(y)
			plt.figure()

			cols = 8
			rows = int(len(feature_imgs) / cols)

			fig, ax = plt.subplots(rows, cols, figsize = (36,12))
			for r in range(rows):
				for c in range (cols):
					i = r*cols + c
					f_ax = ax[r, c]

					f_ax.imshow(feature_imgs[i], cmap='gray')
					f_ax.set_title('Feature' + str(i+1))
					f_ax.set_axis_off()
			# 保存
			plt.savefig(img_path.split('.')[0] + '_conv2d_' + str(layer + 1) + '_all' + '.png')
# 1枚だけ描画
def imshow_single():
	for img_path in img_list:
		for layer in range(5):
			img = get_img(img_path)
			y = get_feature(layer + 1, img)
			feature_imgs = make_img(y)
			plt.figure()

			plt.imshow(feature_imgs[0], cmap='gray')
			# 保存
			plt.savefig(img_path.split('.')[0] + '_conv2d_' + str(layer + 1) + '_single' + '.png')


# 出力形式の決定
while(True):
	print("output all feature map : 0, output one feature map : 1")
	img_num = int(input())
	if img_num == 0 or img_num == 1:
		break
	else:
		print("please input 0 or 1")

# 描画
if img_num == 0:
	imshow_all()
else:
	imshow_single()
