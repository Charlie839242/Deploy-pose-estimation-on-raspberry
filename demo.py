import cv2
import numpy as np
import tensorflow as tf

import visualize
from decode import person_from_keypoints_with_scores



def img_pre(img):
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)       # 双线性插值得到192*192尺寸                                                      # 归一化
    input_data = np.asarray(img).astype(dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)                         # 扩充1个维度，得到1*192*192*3
    return input_data


PATH_TO_TFLITE = 'movenet_lightning.tflite'

interpreter = tf.lite.Interpreter(model_path=PATH_TO_TFLITE)
print('model loaded successfully')

interpreter.allocate_tensors()

# 获取模型的详细数据
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

camera = cv2.VideoCapture(0)
_,img  = camera.read()         # 先获取一次图片，以获取长宽
width  = img.shape[1]
height = img.shape[0]
while True:
    _,img = camera.read()
    print('The original img shape is', img.shape)
    input_data = img_pre(img)
    print('The resized img shape is', input_data.shape)

    # 设置模型输出
    interpreter.set_tensor(input_details[0]['index'],input_data)
    print('input data set successfully')

    # 运行模型
    print('start running model')
    interpreter.invoke()
    print('running model done')

    # 获取模型输出
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('The shape of the output_data is ', output_data.shape)
    output_data = np.squeeze(output_data)           # 删除单维度
    print('The shape of the output_data after squeeze is ', output_data.shape)
    decode_data = [person_from_keypoints_with_scores(output_data, height, width)]
    print('The decode version of the output data is ', decode_data)

    img = visualize.visualize(img, decode_data)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break






















