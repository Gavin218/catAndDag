import pandas as pd
import numpy as np
# file_path1 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
# file_path2 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_dog_imgs"
# f1 = pd.read_pickle(file_path1)
# f2 = pd.read_pickle(file_path2)
# f3 = np.vstack((f1, f2))
# t = 2

# import tensorflow as tf
# y = [1,3,2,2]
# y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为整形张量
# y = tf.one_hot(y, depth=3) # one-hot 编码
# print(y)

x = []
y = [1,3,2]
z = [4,4,4]
x += y
print(x)
x += z
print(x)
