
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
        pickle.dump(new_file_list, f)
    print("已成功存入指定文件！")
    return 0

input_path = "D:/桌面/relatedFile/狗狗/"
suffix = ["jpg", "jpeg", "png"]
X_size = 227
Y_size = 227
output_path = "D:/桌面/relatedFile/cat_and_dog/dog_imgs"
readPicturesAndResizeThenOutput(input_path, suffix, X_size, Y_size, output_path)