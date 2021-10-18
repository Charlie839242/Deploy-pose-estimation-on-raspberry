# Deploy-pose-estimation-on-raspberry  
This project deploys pose estimation model--movenet on raspberry  
该项目推理的pose estimation模型是谷歌的movenet。  
感谢[tensorflow-example](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi)。
我在学习和理解movenet相关内容的时候从这个仓库学到了很多。  

## 如何运行
### 在电脑端测试：
```
pip3 install requirements.txt
使用摄像头：  python3 demo.py --model_path="./model/movenet_lightning_float16.tflite" --video="0" --			             platform="PC"
使用本地视频：python3 demo.py --model_path="./model/movenet_lightning_float16.tflite" --video="test.mp4" 
			--platform="PC"
			
```
### 在树莓派测试：
```
pip3 install requirements.txt
使用摄像头：  python3 demo.py --model_path="./model/movenet_lightning_float16.tflite" --video="0" --			             platform="raspberry"
使用本地视频：python3 demo.py --model_path="./model/movenet_lightning_float16.tflite" --video="test.mp4" 
			--platform="raspberry"
```

## 对movenet的一些理解

### ①Bottom-up

movenet是一个bottom-up的单人姿态检测模型，即movenet根据特征图先输出关键点，再根据这些点的相对位置来筛选。这样就免去了部署如YOLO的一些人体检测器。缺点是bottom-up模型的精度不如top-down的模型。

Openpose就是一个top-down的模型，需要额外的人体检测器，加大了部署成本。

### ②输入和预处理

模型输入的图片尺寸是1×192×192×3.

可以直接用opencv-python对图像做预处理：

```
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_LINEAR)       
    input_data = np.asarray(img).astype(dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)                         
```

### ③输出和解码

模型的输出是一个1×1×17×3的张量。

先得到17×3的矩阵：

```
	output_data = np.squeeze(output_data)     
```

在得到的17×3矩阵中:

```
    for i in range[17]：
     print (output_data[i])
```

对于打印出来的十七个numpy数组，每个数组包含一个关键点的信息，output_data[i , 0], output_data[i , 1], output_data[i , 2]分别表示该关键点的y轴坐标，x轴坐标和置信度，其中y轴坐标和x轴坐标的大小在（0，1）之间，需要根据显示的图片进行处理，得到真实图片。

例如，我们的图片是显示的图片是640×480，而得到的output_data[i , 0] = 0.5，则真实坐标应该是480×0.5=240。



这十七个关键点定义在BodyPart这个class中：

```
    class BodyPart(enum.Enum):
      NOSE = 0
      LEFT_EYE = 1
      RIGHT_EYE = 2
      LEFT_EAR = 3
      RIGHT_EAR = 4
      LEFT_SHOULDER = 5
      RIGHT_SHOULDER = 6
      LEFT_ELBOW = 7
      RIGHT_ELBOW = 8
      LEFT_WRIST = 9
      RIGHT_WRIST = 10
      LEFT_HIP = 11
      RIGHT_HIP = 12
      LEFT_KNEE = 13
      RIGHT_KNEE = 14
      LEFT_ANKLE = 15
      RIGHT_ANKLE = 16
```



接下来就是对shape为17×3的output_data进行decode。

在decode之前，先定义几个类来辅助解码：

```
    class Point(NamedTuple):
      x: float
      y: float
    
    class KeyPoint(NamedTuple):
      body_part: BodyPart
      coordinate: Point
      score: float
    
    class Rectangle(NamedTuple):
      start_point: Point
      end_point: Point
      
    class Person(NamedTuple):
      keypoints: List[KeyPoint]
      bounding_box: Rectangle
      score: float
```

对于每个关键点，我们有三个信息要储存：

​		①：关键点的类别。即这个点是头的还是手的。

​		②：关键点的坐标。

​		③：关键点的置信度。即衡量这个关键点是不是误检测。

因此，用KeyPoint和Point两个类可以储存关键点的信息。因为有十七个关键点，所以建立一个包含17个KeyPoint对象的List：

```
	keypoints = []
    for i in range(scores.shape[0]):
      keypoints.append(
         KeyPoint(
              BodyPart(i),
              Point(int(kpts_x[i] * image_width), int(kpts_y[i] * image_height)),
              scores[i]))
```



此外通过得到的17个点的信息得到这个人的检测框，这个信息储存在Rectangle类中。

这里，我们通过选取17个关键点中最左下角和最右上角的坐标来确定检测框：

```
    start_point = Point(
        int(np.min(kpts_x) * image_width), int(np.min(kpts_y) * image_height))
    end_point = Point(
        int(np.max(kpts_x) * image_width), int(np.max(kpts_y) * image_height))
    bounding_box = Rectangle(start_point, end_point)
```



再者，通过每个关键点的置信度，我们可以设定一个阈值，小于这个阈值代表该关键点不存在。把所有大于阈值的关键点置信度取平均，可以用来推测这个人是否存在，若平均值太低，则代表不存在。该平均值储存在Person类的score属性中。

这里通过lambda和filter函数实现。

filter函数接收两个参数，第一个是函数，第二个是序列。filter()会把序列所有的元素传给函数，返回True和False，然后将返回True的元素放在一个新列表中返回。lambda创建了一个匿名函数，入口是x，返回的是x > keypoint_score_threshold。

```
    scores_above_threshold = list(
    	filter(lambda x: x > keypoint_score_threshold, scores))
    person_score = np.average(scores_above_threshold)
```

~~***这一步有一个bug，当检测到的所有关键点都不合格时，scores_above_threshold可能是一个空列表，这时np.average返回的是nan值，电脑会报一个警告***~~

**已解决，解决方法：**

```
    scores_above_threshold = list(
    	filter(lambda x: x > keypoint_score_threshold, scores))
    scores_above_threshold = np.append(scores_above_threshold, 0)
    # 手动添加一个0来确保不会出现空列表
    person_score = np.average(scores_above_threshold)
```



最终，返回的应该是一个Person类，Person类包含了上述keypoints，bounding_box和person_score：

```
	return Person(keypoints, bounding_box, person_score)
```



### ④可视化

首先判断这个人是否存在，即是否应该显示检测框和关键点。通过Person的score属性判断，若太小，则代表这个人不存在。

```
    for person in list_persons:
    	if person.score < instance_threshold:
    		break; 
```



然后把合格的关键点画出来，'合格' 指的是该关键点大于设置的阈值：

```
    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # 根据关键点的score把所有合格的关键点画出
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, (0, 255, 0), 4)
```



下一步是在画出来的关键点之间连线。这一步是预先设定了哪些关键点之间应该连接，比如设定了身体和手相连。然后根据要连接的关键点的score来判断是否相连，以身体和手为例，需要身体关键点和手关键点的score同时达到一个阈值，才能对它们进行连线：

```
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, color, 2)
```

KEYPOINT_EDGE_INDS_TO_COLOR是一个字典类数据，其中预先保存了关键点之间的连接关系和关键点之间连线的颜色：

```
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): (147, 20, 255),
        (0, 2): (255, 255, 0),
        (1, 3): (147, 20, 255),
        (2, 4): (255, 255, 0),
        (0, 5): (147, 20, 255),
        (0, 6): (255, 255, 0),
        (5, 7): (147, 20, 255),
        (7, 9): (147, 20, 255),
        (6, 8): (255, 255, 0),
        (8, 10): (255, 255, 0),
        (5, 6): (0, 255, 255),
        (5, 11): (147, 20, 255),
        (6, 12): (255, 255, 0),
        (11, 12): (0, 255, 255),
        (11, 13): (147, 20, 255),
        (13, 15): (147, 20, 255),
        (12, 14): (255, 255, 0),
        (14, 16): (255, 255, 0)
    }
```



最后一步，把Person.bounding_box的检测框画出来：

```
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, (0, 255, 0), 1)
```



显示图片：

```
	cv2.imshow('image', image)
	if cv2.waitKey(1) == 27:
        break
```











