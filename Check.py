# 进行测试，看模型准确率
def darryFileToTfFile(path, modelPath):
    import pandas as pd
    import tensorflow as tf
    dataList = pd.read_pickle(path)
    tf_data = tf.cast(dataList, dtype=tf.int32)
    model = tf.saved_model.load(modelPath)
    result = model(tf_data)
    t = 8
    return 0

path = "D:/桌面/relatedFile/cat_and_dog/测试集/test_cat_imgs"
model_path = "D:/桌面/relatedFile/cat_model"
darryFileToTfFile(path, model_path)


