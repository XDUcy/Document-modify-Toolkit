import sys
sys.path.append('/home/aistudio/external-libraries')

import gradio as gr
from PIL import Image
import Transform
from Enhance import reinforce_1
import os
from datetime import datetime

def transform_image(image):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = "uploads"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input_image_path = os.path.join(save_path, f"{timestamp}.png")
    image.save(input_image_path)

    output_path = "processed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image_path = os.path.join(output_path, f"transformed_{timestamp}.png")

    Transform.predict.run(input_image_path, output_image_path)
    processed_image = Image.open(output_image_path)
    return processed_image

def enhance_image(image):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = "uploads"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    input_image_path = os.path.join(save_path, f"{timestamp}.png")
    image.save(input_image_path)

    output_path = "processed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image_path = os.path.join(output_path, f"enhanced_{timestamp}.png")

    reinforce_1.enhance_image(input_image_path, output_image_path)
    processed_image = Image.open(output_image_path)
    return processed_image

def Option(image, operation):
    if operation == "矫正":
        return transform_image(image)
    elif operation == "增强":
        return enhance_image(image)
    elif operation == "先矫正后增强":
        transformed_image = transform_image(image)
        return enhance_image(transformed_image)

title = "图片矫正&增强小工具 - 拯救看不清的PPT"
description = "这是一个图片矫正&增强小工具，可以帮助你矫正图片倾斜，增强图片清晰度。并且可以（可选地）对图片进行二值化增强处理，使得图片更加清晰。"
article="""
    ## 说明
    这是一个简单的图像处理应用。您可以上传一张图片，应用将对其进行处理，并返回处理后的图片。
    - 上传图片的格式可以是 PNG、JPG 等。
    - 图片处理过程可能包括一些基本的操作，例如调整大小、旋转等。
    - 处理后的图片将显示在页面上。

    ### 使用步骤
    1. 点击下方的“选择文件”按钮上传图片。
    2. 等待图片处理完成。
    3. 查看处理后的图片。
    """


iface = gr.Interface(
    fn = Option,
    inputs = [
        gr.Image(type='pil', label="上传图片"),
        gr.Radio(["矫正", "增强", "先矫正后增强"], label="操作类型")
    ],
    outputs="image",
    live = True,
    title = title,
    description=description
)

iface.launch()