

"""可读取图片（路径为中文）
:param file_path: 图片路径
"""
def cv_imread(file_path):
    import cv2
    import numpy as np
    # 可以使用中文路径读取图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

def cv_show(file_path):
    import cv2
    import pandas as pd
    img = pd.read_pickle(file_path)[87]  # 随便读取一个
    cv2.imshow("img", img)
    cv2.waitKey(0)
    return 0

# file_path = "D:/桌面/relatedFile/cat_and_dog/测试集/test_cat_imgs"
# cv_show(file_path)