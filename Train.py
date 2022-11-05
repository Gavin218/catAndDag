


def alexNet_train():
    import tensorflow as tf
    from tensorflow import keras
    from Net import AlexNet
    from Net import AlexNetTest
    from Preprocessing_Functions import mergeDiffDataAndGiveTags
    tf.random.set_seed(44)
    model = AlexNetTest()
    model.build(input_shape=(None, 227, 227, 3))
    model.encoder.summary()
    optimizer = keras.optimizers.Adam(lr=0.001)
    train_file_path1 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_cat_imgs"
    train_file_path2 = "D:/桌面/relatedFile/cat_and_dog/训练集/train_dog_imgs"
    path_list = [train_file_path1, train_file_path2]
    tf_data = mergeDiffDataAndGiveTags(path_list, 1)

    for epoch in range(2):
        for step, (x, y_real) in enumerate(tf_data):
            with tf.GradientTape() as tape:
                y_pre = model(x)
                ori_loss = tf.losses.categorical_crossentropy(y_real, y_pre, from_logits=True)
                # ori_loss = tf.losses.mean_squared_error(y_real, y_pre)
                rec_loss = tf.reduce_mean(ori_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                # 间隔性打印训练误差
                print(epoch, step, float(rec_loss))
    model.save("D:/桌面/relatedFile/cat_model")
    return 0

alexNet_train()
# import pandas as pd
