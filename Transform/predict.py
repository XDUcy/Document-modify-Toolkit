import cv2
import paddle

# from GeoTr import GeoTr
from .GeoTr import GeoTr
from .utils import to_image, to_tensor



def run(input_path, output_path):
    # 配置输入文件和模型
    # image_path = input_img
    model_path = "/home/aistudio/Transform/best.ckpt"
    # 配置输出文件
    # base_output_path = "./doc/output/"
    # t = str(time.time())
    # output_path = base_output_path + t + ".jpg"

    checkpoint = paddle.load(model_path)
    state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    img_org = cv2.imread(input_path)
    img = cv2.resize(img_org, (288, 288))
    x = to_tensor(img)
    y = to_tensor(img_org)
    bm = model(x)
    bm = paddle.nn.functional.interpolate(
        bm, y.shape[2:], mode="bilinear", align_corners=False
    )
    bm_nhwc = bm.transpose([0, 2, 3, 1])
    out = paddle.nn.functional.grid_sample(y, (bm_nhwc / 288 - 0.5) * 2)
    out_image = to_image(out)
    cv2.imwrite(output_path, out_image)
    return 