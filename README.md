# Deploy-pose-estimation-on-raspberry  
This project deploys pose estimation model--movenet on raspberry  
该项目推理的pose estimation模型是谷歌的movenet。  
感谢tensorflow[tensorflow-example](https://github.com/tensorflow/examples/tree/master/lite/examples/pose_estimation/raspberry_pi)。
我在学习和理解movenet相关内容的时候从这个仓库学到了很多。  
## 如何运行
### 在电脑端测试：
```
pip3 install requirements.txt
python3 demo_pc.py
```
### 在树莓派测试：
```
pip3 install requirements.txt
python3 demo_raspberry.py
```