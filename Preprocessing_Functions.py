
# input_path为文件夹 后缀需为/，suffix为支持的后缀格式
def readPicturesAndResizeThenOutput(input_path, suffix_list, X_size, Y_size, output_path):
    import os
    import cv2
    import pickle
    from CV_Functions import cv_imread
    new_file_list = []
    i = 1
    for root, sub_folders, file_list in os.walk(input_path):
        for file in file_list:
            if file.split('.')[1] in suffix_list:
                print("正在读取第%d张图片" % i)
                img = cv_imread(input_path + file)
                if img is not None:
                    new_img = cv2.resize(img, (X_size, Y_size), interpolation=cv2.INTER_CUBIC)
                    new_file_list.append(new_img)
                    print("第%d张图片已成功转换" % i)
                    i += 1
    with open(output_path, 'wb') as f:
        pickle.dump(new_file_list, f)  # 以list格式存储

        # 若想存储为ndarray格式，则用以下函数
        # dim = [len(new_file_list), X_size, Y_size, 3]  # 若彩色图片默认为3
        # pickle.dump(list2Array(new_file_list, dim))
    print("已成功存入指定文件！")
    return 0

# input_path = "D:/桌面/relatedFile/宠物猫/"
# suffix = ["jpg", "jpeg", "png"]
# X_size = 227
# Y_size = 227
# output_path = "D:/桌面/relatedFile/cat_and_dog/测试集/test_cat_imgs"
# readPicturesAndResizeThenOutput(input_path, suffix, X_size, Y_size, output_path)


def png_to_jpg(darry_data):
    import numpy as np
    for i in range(len(darry_data)):
        np.delete(darry_data, 3, axis=1)
    return


def list2Array(origin_list):
    import numpy as np
    num = len(origin_list)
    dim = [num] + list(origin_list[0].shape)
    data_array = np.zeros(dim)
    for i in range(num):
        origin_data = origin_list[i]
        data_shape = list(origin_data.shape)
        if len(data_shape) != 3:
            continue
        if data_shape[2] == 4:  # 如果为png格式
            origin_data = png_to_jpg(origin_data)
        data_array[i] = origin_data
    return data_array


def list2ArrayIO(input_path, output_path):
    import pandas as pd
    import pickle
    origin_list = pd.read_pickle(input_path)
    array_data = list2Array(origin_list)
    with open(output_path, 'wb') as f:
        pickle.dump(array_data, f)
    print("已成功存入指定文件！")
    return 0

input_path = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
output_path = "D:/桌面/relatedFile/cat_and_dog/训练集2/train_cat_imgs"
list2ArrayIO(input_path, output_path)


