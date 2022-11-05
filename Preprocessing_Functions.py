
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
        # pickle.dump(list2Array(new_file_list))
    print("已成功存入指定文件！")
    return 0

# input_path = "D:/桌面/relatedFile/宠物猫/"
# suffix = ["jpg", "jpeg", "png"]
# X_size = 227
# Y_size = 227
# output_path = "D:/桌面/relatedFile/cat_and_dog/测试集/test_cat_imgs"
# readPicturesAndResizeThenOutput(input_path, suffix, X_size, Y_size, output_path)


def list2Array(origin_list):
    import numpy as np
    num = len(origin_list)
    dim = [num] + list(origin_list[0].shape)
    data_array = np.zeros(dim, dtype=origin_list[0].dtype)
    i = 0 # 新array的索引
    i1 = -1 # 原list的索引
    while i1 < num - 1:
        i1 += 1
        origin_data = origin_list[i1]
        data_shape = list(origin_data.shape)
        if len(data_shape) != 3:
            continue
        if data_shape[2] == 4:  # 如果为png格式
            data_array[i] = np.delete(origin_data, 3, 2)
        else:
            data_array[i] = origin_data
        i += 1
    return data_array[0:i]


def list2ArrayIO(input_path, output_path):
    import pandas as pd
    import pickle
    origin_list = pd.read_pickle(input_path)
    array_data = list2Array(origin_list)
    with open(output_path, 'wb') as f:
        pickle.dump(array_data, f)
    print("已成功存入指定文件！")
    return 0

# input_path = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
# output_path = "D:/桌面/relatedFile/cat_and_dog/训练集2/train_cat_imgs"
# list2ArrayIO(input_path, output_path)

#将darry文件转换为可直接训练的Tensor张量的batch形式(如文件为单一存储，可输入sheet = -1)
def darryFileToTfFile(path, batchSize):
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    dataList = pd.read_pickle(path)
    tf_data = tf.cast(dataList, dtype=tf.int32)
    train_db = tf.data.Dataset.from_tensor_slices(tf_data)
    train_db = train_db.shuffle(batchSize * 5).batch(batchSize)
    return train_db


def darryFileToTfFileAngGiveTag(path, batchSize):
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    dataList = pd.read_pickle(path)
    tagList = np.full((len(dataList), 2), [1, 0])
    tf_data = tf.cast(dataList, dtype=tf.int32)
    tf_tag = tf.cast(tagList, dtype=tf.int32)
    train_db = tf.data.Dataset.from_tensor_slices((tf_data, tf_tag))
    train_db = train_db.shuffle(batchSize * 5).batch(batchSize)
    return train_db

#将darry文件转换为可直接训练的Tensor张量的batch形式(如文件为单一存储，可输入sheet = -1)
def mergeDiffDataAndGiveTags(pathList, batchSize):
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    tf.random.set_seed(44)
    n = len(pathList)
    dataList = []
    tagList = []
    for i in range(len(pathList)):
        newTag = [0] * n
        newTag[i] = 1
        if i == 0:
            dataList = pd.read_pickle(pathList[i])
            tagList = np.full((len(dataList), n), newTag)
        else:
            dataTmp = pd.read_pickle(pathList[i])
            dataList = np.vstack((dataList, dataTmp))
            tagList = np.vstack((tagList, np.full((len(dataTmp), n), newTag)))
    tf_data = tf.cast(dataList, dtype=tf.int32)
    tf_tag = tf.cast(tagList, dtype=tf.float32)
    train_db = tf.data.Dataset.from_tensor_slices((tf_data, tf_tag))
    train_db = train_db.shuffle(batchSize * 5).batch(batchSize)
    return train_db


