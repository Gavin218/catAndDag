import cv2
import numpy as np

def cv_imread(file_path):
    """可读取图片（路径为中文）

    :param file_path: 图片路径
    :return:
    """
    # 可以使用中文路径读取图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

file_path = "D:/桌面/cat.jpg"
# img = cv2.imread("D:/桌面/cat.jpg")
img = cv_imread(file_path)
print(img.shape)