import cv2
import numpy as np
import time
import visualize
from decode import person_from_keypoints_with_scores
import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--model_path',
    help='Path of estimation model.',
    required=False,
    type=str,
    default='./model/movenet_lightning_float16.tflite')
parser.add_argument(
    '--video', help='Using camera or video', type=str, required=False, default='0')
parser.add_argument(
    '--platform', help='Run on PC or respberry.', type=str, required=False, default='PC')
parser.add_argument(
    '--classifier', help='Path of classification model.', required=False)
args = parser.parse_args()


if (args.platform == 'PC'):
    import tensorflow
    Interpreter = tensorflow.lite.Interpreter
    print('Using full tensorflow pkg')
elif (args.platform == 'raspberry'):
    from tflite_runtime.interpreter import Interpreter
    print('Using tflite_runtime')

def img_pre(img):
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)       # 双线性插值得到192*192尺寸
    input_data = np.asarray(img).astype(dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)                         # 扩充1个维度，得到1*192*192*3
    return input_data


if args.classifier:
    classify_pose = ['chair', 'cobra', 'dog', 'tree', 'warrior']
    classifier = Interpreter(model_path=args.classifier, num_threads=4)
    print('classify model loaded successfully')

    classifier.allocate_tensors()
    classify_input_details = classifier.get_input_details()
    classify_output_details = classifier.get_output_details()



interpreter = Interpreter(model_path=args.model_path, num_threads=4)
print('pose estimation model loaded successfully')

interpreter.allocate_tensors()

# 获取模型的详细数据
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

try:
    camera = cv2.VideoCapture(int(args.video))
except ValueError:
    camera = cv2.VideoCapture(args.video)

_,img  = camera.read()         # 先获取一次图片，以获取长宽
width  = img.shape[1]
height = img.shape[0]
while True:
    time_start = time.time()
    _,img = camera.read()

    # 图像预处理
    input_data = img_pre(img)

    # 设置模型输出
    interpreter.set_tensor(input_details[0]['index'],input_data)

    # 运行模型
    interpreter.invoke()

    # 获取模型输出
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 模型输出解码
    output_data = np.squeeze(output_data)           # 删除单维度
    decode_data = [person_from_keypoints_with_scores(output_data, height, width)]

    if args.classifier:
        input_tensor = [[
            keypoint.coordinate.y, keypoint.coordinate.x, keypoint.score
        ] for keypoint in decode_data[0].keypoints]
        input_tensor = np.array(input_tensor)
        if(min(input_tensor[:,2]) > 0.1):
            input_tensor = input_tensor.flatten().astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            classifier.set_tensor(classify_input_details[0]['index'], input_tensor)

            classifier.invoke()

            output_tensor = classifier.get_tensor(classify_output_details[0]['index'])
            print(classify_pose[np.argmax(output_tensor)])



    # visualize
    img = visualize.visualize(img, decode_data)
    time_end = time.time()
    FPS = 'FPS = ' + str(1/(time_end - time_start))
    cv2.putText(img, FPS, (24, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)      # 显示FPS

    cv2.imshow('img', img)
    if cv2.waitKey(1) == 27:
        break






















