

"""可读取图片（路径为中文）
:param file_path: 图片路径
"""
def cv_imread(file_path):
    import cv2
    import numpy as np
    # 可以使用中文路径读取图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img