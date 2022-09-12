
# input_path为文件夹 后缀需为/，suffix为后缀，需为".jpg"等格式
def readPicturesAndResizeThenOutput(input_path, suffix, X_size, Y_size, output_name, output_path):
    import os
    import cv2
    import pickle
    from CV_Functions import cv_imread
    new_file_list = []
    for root, sub_folders, file_list in os.walk(input_path):
        for file in file_list:
            if file.endswith(suffix):
                img = cv_imread(input_path + file)
                new_img = cv2.resize(img, (X_size, Y_size), interpolation=cv2.INTER_CUBIC)
                new_file_list.append(new_img)
    with open(output_path, 'wb') as f:
        pickle.dump(new_file_list, f)
    print("已成功存入指定文件！")
    return 0

input_path = "D:/桌面/test/"
suffix = ".jpg"
X_size = 400
Y_size = 400
output_name = 0
output_path = "D:/桌面/test2/resize_imgs"
readPicturesAndResizeThenOutput(input_path, suffix, X_size, Y_size, output_name, output_path)