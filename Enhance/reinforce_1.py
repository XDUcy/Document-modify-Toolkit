import sys
sys.path.append('/home/aistudio/external-libraries')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage import morphology,filters,feature
from scipy import ndimage


def enhance_image(input_path, output_path):
    image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_equalized_image = clahe.apply(gray_image)
    blurred_gray_image = cv2.GaussianBlur(adaptive_equalized_image, (3, 3), 0)
    _, binary_image_1 = cv2.threshold(blurred_gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.subtract(binary_image_1, blurred_gray_image)
    sharpened_image = cv2.add(binary_image_1, mask)
    denoised = filters.rank.median(sharpened_image,morphology.disk(2))
    markers = filters.rank.gradient(denoised,morphology.disk(5))<10
    markers = ndimage.label(markers)[0]

    gradient = filters.rank.gradient(denoised,morphology.disk(2))
    labels = watershed(gradient, markers, mask = sharpened_image)
    kernel = np.ones((5,5), np.uint8)
    eroded_labels = cv2.erode(labels.astype(np.uint8), kernel, iterations=1)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_binary_image = cv2.bitwise_not(binary_image)
    eroded_black_points = cv2.erode(inverted_binary_image, kernel, iterations=1)

    combined_labels = cv2.add(eroded_labels, eroded_black_points)
    region_areas = np.bincount(combined_labels.flatten())
    threshold_area = 10

    mask = np.zeros_like(combined_labels)
    for region_label, region_area in enumerate(region_areas):
        if region_area >= threshold_area:
            mask[combined_labels == region_label] = 255

    masked_binary_image = cv2.bitwise_and(binary_image, mask)
    smooth_image = cv2.GaussianBlur(masked_binary_image,(5,5),0)
    edges = cv2.bitwise_not(cv2.Canny(smooth_image,1,25))
    mask_bitwise = np.ones_like(masked_binary_image)
    bitwise_and_result = cv2.bitwise_and(masked_binary_image,edges,mask = mask_bitwise)
    canny_image = cv2.addWeighted(bitwise_and_result,0.7,masked_binary_image,0.3,0)
    cv2.imwrite(output_path, canny_image)
    return 


if __name__ == "__main__":
    # 读取图像
    image = cv2.imread("example_3.png")

    #转换为灰度图处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #自适应均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_equalized_image = clahe.apply(gray_image)

    # 高斯模糊
    blurred_gray_image = cv2.GaussianBlur(adaptive_equalized_image, (3, 3), 0)

    #二值化去噪
    _, binary_image_1 = cv2.threshold(blurred_gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 计算锐化的掩码
    mask = cv2.subtract(binary_image_1, blurred_gray_image)

    # 将原始灰度图像与掩码相加，得到锐化后的图像-我猜应该放在最后
    sharpened_image = cv2.add(binary_image_1, mask)

    #分水岭算法-ostu算法计算连通区域
    # 过滤噪声
    denoised = filters.rank.median(sharpened_image,morphology.disk(2))
    markers = filters.rank.gradient(denoised,morphology.disk(5))<10
    markers = ndimage.label(markers)[0]

    gradient = filters.rank.gradient(denoised,morphology.disk(2))
    labels = watershed(gradient, markers, mask = sharpened_image)

    # 腐蚀操作
    kernel = np.ones((5,5), np.uint8)
    eroded_labels = cv2.erode(labels.astype(np.uint8), kernel, iterations=1)

    # 找到图像中的小黑点，并将其标记为前景区域
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_binary_image = cv2.bitwise_not(binary_image)
    eroded_black_points = cv2.erode(inverted_binary_image, kernel, iterations=1)

    # 将小黑点标记和分割结果合并
    combined_labels = cv2.add(eroded_labels, eroded_black_points)

    # 计算区域面积
    region_areas = np.bincount(combined_labels.flatten())

    # 设定阈值，过滤小区域
    threshold_area = 10

    # 创建掩码
    mask = np.zeros_like(combined_labels)
    for region_label, region_area in enumerate(region_areas):
        if region_area >= threshold_area:
            mask[combined_labels == region_label] = 255

    # 应用掩码
    masked_binary_image = cv2.bitwise_and(binary_image, mask)
    # canny算法提取边缘
    smooth_image = cv2.GaussianBlur(masked_binary_image,(5,5),0)
    edges = cv2.bitwise_not(cv2.Canny(smooth_image,1,25))
    mask_bitwise = np.ones_like(masked_binary_image)
    bitwise_and_result = cv2.bitwise_and(masked_binary_image,edges,mask = mask_bitwise)
    canny_image = cv2.addWeighted(bitwise_and_result,0.7,masked_binary_image,0.3,0)


    #展示
    cv2.imshow("Original Image", image)
    cv2.imshow("Canny Edges",edges)
    cv2.imshow("Masked Binary Image",masked_binary_image)
    cv2.imshow("Canny Images",canny_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()