import pandas as pd
import numpy as np
file_path1 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
file_path2 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_dog_imgs"
f1 = pd.read_pickle(file_path1)
f2 = pd.read_pickle(file_path2)
f3 = np.vstack((f1, f2))
t = 2
