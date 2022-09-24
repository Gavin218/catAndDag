import tensorflow as tf


def alexNet_train():
    from Net import AlexNet
    import pandas as pd
    import numpy as np
    model = AlexNet()
    model.build(input_shape=(None, 227, 227, 3))
    model.encoder.summary()

    train_file_path = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
    dataset = pd.read_pickle(train_file_path)
    # test1_img = dataset[0]
    test1_img = dataset[0]
    # tf_img = tf.convert_to_tensor(test1_img)
    # 转换语句出现了问题！
    # tf2_img = tf.reshape(tf_img, [None, 227, 227, 3])
    tf2_img = tf.reshape(test1_img, [None, 227, 227, 3])
    y = model(tf2_img)
    t = 34
    return 0

alexNet_train()
# import pandas as pd
# train_file_path = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
# dataset = pd.read_pickle(train_file_path)
# img = dataset[0]
# tf_img = tf.convert_to_tensor(img)
# t = 3