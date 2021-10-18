import math
from typing import List, Tuple

import cv2
from decode import Person
import numpy as np


# (0, 1)pair是相邻的关键点编号
# (147, 20, 255)是连接关键点的线的颜色
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


def visualize(image: np.ndarray,
              list_persons: List[Person],
              keypoint_color: Tuple[int, ...] = (0, 255, 0),
              keypoint_threshold: float = 0.25,
              instance_threshold: float = 0.3):

  for person in list_persons:
    if person.score < instance_threshold:
      break;        # 因为该模型只检测单人，所以直接break

    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # 根据关键点的score把所有合格的关键点画出
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, keypoint_color, 4)

    # 根据相邻关键点的score判断是否把他们连上
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, color, 2)

    # 把整个框画出来
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, keypoint_color, 1)

  return image


