import os

import numpy as np
import pandas as pd

import CV_Functions
import cv2
# file_path = "D:/桌面/test/"
# new_file_list = []
# for root, sub_folder, file_list in os.walk(file_path):
#     print(file_list)
# [x, y, z] = os.walk(file_path)
# print(z)
# t = 0

# file_Path = "D:/桌面/test/5-两只猫猫壁纸堆糖美图壁纸兴趣社区.jpg"
# img1 = CV_Functions.cv_imread(file_Path)
# img2 = cv2.resize(img1, (400, 400), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('image',img1)
# cv2.imshow('image2', img2)
# cv2.waitKey(0)
# t = 2

# import pandas as pd
# x = pd.read_pickle("D:/桌面/test2/resize_imgs")
# for i in range(len(x)):
#     cv2.imshow("pic" + str(i), x[i])
# cv2.waitKey(0)

# x = "cec.4"
# y = x.split(".")[1]
# yy = ["234", "DVC订单", '3dd']
# print("DVC订单" in yy)

# print("第%d个" % 4)

# from Net import AlexNet
# model = AlexNet()
# model.build(input_shape=(None, 227, 227, 3))
# model.summary()
# model.encoder.summary()

# import pickle
# x1 = np.array([[[103, 112, 86], [109, 116, 95]], [[123, 100, 34], [163, 112, 90]], [[153, 122, 16], [123, 54, 33]]])
# x2 = np.array([[[23, 34, 23], [234, 75, 34]], [[15, 55, 23], [87, 45, 12]], [[98, 25, 52], [211, 101, 43]]])
# n1 = []
# n1.append(x1)
# n1.append(x2)
# xx1 = x1.shape
# xx2 = x2.shape
# n2 = np.zeros([2] + list(xx1))
# for i in range(len(n1)):
#     n2[i] = n1[1]
# with open("D:/桌面/test_list", 'wb') as f:
#     pickle.dump(n2, f)
# print("已成功存入指定文件！")
# n3 = pd.read_pickle("D:/桌面/test_list")
# t = 8

import cv2
from CV_Functions import cv_imread
# file_path = "D:/桌面/test/png_cat.png"
# img = cv_imread(file_path)
# img = pd.read_pickle("D:/桌面/relatedFile/cat_and_dog/测试集2/test_dog_imgs")[66]
# img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("img", img)
# for i in range(len(img)):
#     np.delete(img, 3, axis=1)
# cv2.imshow("img2", img)
# cv2.waitKey(0)

# x1 = np.array([[[103, 112, 86], [109, 116, 95]], [[123, 100, 34], [163, 112, 90]], [[153, 122, 16], [123, 54, 33]]])
# x2 = x1[0:2]
# t = 2

img = pd.read_pickle("D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs")[87]
img2 = pd.read_pickle("D:/桌面/relatedFile/cat_and_dog/训练集2/train_cat_imgs")[86]
# img2 = np.delete(img, 3, 2)
cv2.imshow("img", img)
cv2.imshow("img2", img2)
cv2.waitKey(0)

print(img.dtype)
print(img2.dtype)

# x = np.array([[0, 2, 3], [4,5,6], [2,6,8]])
# y = x[0:2]
# print(x)
# print(y)

