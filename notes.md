# 了解
自动驾驶方向：
+ 算法：感知、规划、控制
  + 感知算法工程师：传统ML、CV、NLP
    + SLAM算法工程师：simultaneous localization and mapping，根据传感器“点云”数据对周围环境进行建模
      + > 点云是某个坐标系下的点的数据集。点包含了丰富的信息，包括三维坐标X，Y，Z、颜色、分类值、强度值、时间等等。点云可以将现实世界原子化，通过高精度的点云数据可以还原现实世界。   
      + 激光SLAM：Lidar SLAM
      + 视觉SLAM：VSLAM
  + 决策算法工程师：利用感知模块的信息进行汽车加速、减速、左转、右转、换道、超车等决策。
  + 规划算法工程师：包括路径规划和速度规划。
+ 仿真
+ 测试