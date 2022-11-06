# 进行测试，看模型准确率
def darryFileToTfFile(path, modelPath):
    import pandas as pd
    import tensorflow as tf
    dataList = pd.read_pickle(path)
    tf_data = tf.cast(dataList, dtype=tf.float32)
    tf_data = 2 * tf.convert_to_tensor(tf_data, dtype=tf.float32) / 255. - 1
    model = tf.saved_model.load(modelPath)
    result = model(tf_data)
    t = 8
    return 0

path = "D:/桌面/relatedFile/cat_and_dog/测试集/test_dog_imgs"
model_path = "D:/桌面/relatedFile/cat_model"
darryFileToTfFile(path, model_path)


