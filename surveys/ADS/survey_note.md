# 论文笔记
## A review: The Self-Driving Car’s Requirements and The Challenges it Faces
本研究的重点是着眼于并结合当前所有关注自动驾驶汽车（AVs）制造和使用中安全性的方法，特别关注软件和系统的要求。
+ 基本技术：
  + 传感器融合 sensor fusion：camera，RADAR，LiDAR
  + 计算机视觉 computer vision
  + 定位 localization
  + 测绘 mapping
+ 主要研究方向：
  + 减少交通事故
  + 减少交通拥塞
  + 减少有害排放
  + 避免网络安全攻击

基本流程：
1. Sensing：传感器收集数据
2. Perception：感知周围环境和位置
3. Planning：规划路线、任务、动作
4. Control：控制车身

> 已经提出了许多创新的方法来解决自动化车辆系统的软件设计方面，其主要目标是在这些车辆的独特领域以及更广泛的道路生态系统内增强安全性。本文的主要目的是对在这个特定领域内提出的各种方法进行分类。

故本文重点在于自动驾驶技术栈安全性。

## A Systematic Review of Perception System and Simulators for Autonomous Vehicles Research
> This paper presents a systematic review of the perception systems and simulators for autonomous vehicles (AV).

文章重点在于传感器的物理性质：
> 本文介绍了用于操作感知系统中最常用的传感器（超声波、雷达、激光雷达、激光雷达、摄像机、IMU、GNSS、RTK等）的物理基本原理、原理功能和电磁波谱。此外，还显示了它们的优缺点，并使用蜘蛛图对其特征进行量化。

## Applications of Computer Vision in Autonomous Vehicles: Methods, Challenges and Future Directions
>Benefit from the recent advances in computer vision, the perception task can be achieved by using sensors, such as camera, LiDAR, radar, and ultrasonic sensor. This paper reviews publications on computer vision and autonomous driving
that are published during the last ten years. 

1. 自动驾驶系统发展总结
2. 常用传感器和基准数据集
3. CV在自动驾驶中的应用：
   1. depth estimation
   2. object detection
   3. lane detection
   4. traffic sign recognition
4. 公众见解调查
5. 技术挑战

### Introduction
`Advanced Driver Assistance System (ADAS)`分为被动的和主动的，passive ADAS识别危险并提醒；active ADAS主动操作车辆做出动作。

列出了一些前人的survey：
+ E. Arnold, O. Y. Al-Jarrah, M. Dianati, S. Fallah, D. Oxtoby, and A. Mouzakitis, **“A survey on 3D object detection methods for autonomous driving applications,”** IEEE Trans. Intell. Transp. Syst., vol. 20, no.10, pp. 3782–3795, 2019.
+ Y. Zhang, Z. Lu, X. Zhang, J.-H. Xue, and Q. Liao, **“Deep learning in lane marking detection: A survey,”** IEEE Trans. Intell. Transp. Syst., vol. 23, no. 7, pp. 5976–5992, 2021.
+ Z. Wang, J. Zhan, C. Duan, X. Guan, P. Lu, and K. Yang, **“A review of vehicle detection techniques for intelligent vehicles,”** IEEE Trans. Neural Netw. Learn. Syst., 2022.
+ B. Ranft and C. Stiller, **“The role of machine vision for intelligent vehicles,”** IEEE Trans. Intell. Veh., vol. 1, no. 1, pp. 8–19, 2016.
+ F. Manfio Barbosa and F. Santos Os´orio, **“Camera-radar perception for autonomous vehicles and ADAS: Concepts, datasets and metrics,”** arXiv e-prints, pp. arXiv–2303, 2023.
+ S. Grigorescu, B. Trasnea, T. Cocias, and G. Macesanu, **“A survey of deep learning techniques for autonomous driving,”** J. Field Robot., vol. 37, no. 3, pp. 362–386, 2020.

SAE(Society of Automo￾tive Engineers)将自动驾驶系统等级分为六级：
+ Level 0 (no driving automation)
+ Level 1 (driver assistance): 最低级的辅助，驾驶员仍需全程参与。
+ Level 2 (partial driving automation): ADAS进行一定的控制。驾驶员仍要注意。
+ Level 3 (conditional driving automation): 特定情况下完全自动驾驶。驾驶员可以偶尔不参与。
+ Level 4 (high driving automation): 在合适的情况下汽车完全自动驾驶。特定情况下提醒驾驶员参与。
+ Level 5 (full driving automation): 完全自动驾驶。

论文结构：
+ 第二节：选择论文的标准
+ 第三节：简要概述自动驾驶系统的发展情况
+ 第四节：总结自动驾驶中常用的传感器和数据集
+ 第五节：描述自动驾驶汽车环境感知的计算机视觉任务
+ 第六节：公众对自动驾驶汽车的看法
+ 第七节：讨论挑战和未来的发展方向

### III. BRIEF OVERVIEW OF THE DEVELOPMENT OF
AUTONOMOUS DRIVING SYSTEMS
> Driving automation includes both advanced driver assistance systems (ADAS) and automated driving systems (ADS).

流行的ADAS功能：
+ Adaptive Cruise Control **(ACC)**: 自适应巡航控制。Automatically adjusting the vehicle’s speed.
+ High Beam Assist **(HBA)**: 自动切换远近光灯。Automatically switching the headlamp range between high beam and low beam.
+ Lane Departure Warning **(LDW)**: 车道偏离警告。Using
cameras to monitor and warns the driver if the vehicle is leaving its lane with.
+ Lane Keep Assist **(LKA)**: 车道保持辅助。Using cameras to automatically provide corrective steering to help keep the car securely in the detected lane.
+ Pre-Collision Warning **(PCW)**: 碰撞前警告。Using camera
or radar to detect potential collisions with vehicles or
pedestrians in front of the vehicle.
+ Traffic Sign Recognition **(TSR)**: 识别交通标志。
+ Driver Attention Monitor **(DAM)**: 驾驶员注意力监视器。A camera-based technology that tracks driver alertness.
+ Traffic Jam Assist **(TJA)**: 交通堵塞辅助系统。a feature using cameras to monitor lane markings and vehicles ahead. TJA combines features of ACC and LKA to automatically brake and steer if the driver does not take action in time.

### IV. SENSORS AND DATA SETS
传感器：
+ 相机：
  + 便宜、提供直接的二维信息
  + 易受雾霾等天气因素影响；高分辨率下数据量太大，几十MB/s
+ LiDAR：
  + 直接测量距离；构建精确和高精度的地图
  + 不适合检测小目标；高成本；受天气影响；数据量大，几十MB/s
+ Radar：
  + 检测距离、相对运动；相对适应不同的天气；数据量小，100KB/s
+ Ultrasonic sensor：超声波传感器
  + 易使用、高精度、检测小物体变化；数据量小，100KB/s
  + 测量距离小，不灵活

### V. ENVIRONMENT PERCEPTION FOR AUTONOMOUS VEHICLES
四个CV任务已经应用到了自动驾驶：
1. Depth Estimation：使用单眼RGB相机，利用DL从中得到深度图像
2. Object Detection：
   1. 通用对象检测：找到预定义的类别的object。R-CNN、SPPNet、Faster R-CNN、RPN。YOLO、SSD
   2. 特定目标检测：专门检测车辆、自行车、行人等。对每一个特定类别有其专门的检测方法。
      + 例子：B. Wu, F. Iandola, P. H. Jin, and K. Keutzer, **“SqueezeDet: Unified, small, low power fully convolutional neural networks for real-time object detection for autonomous driving,”** in Proc. CVPR Workshops,2017, pp. 129–137.
      + 人类会进行推断，例如夜间迎面而来的大灯；CV任务却一般只依赖清晰的轮廓信息。因此有这方面的融合工作。
3. Lane Detection：检测车道区域或车道标记。车道检测帮助车辆在道路车道内正确地定位，最大限度地减少碰撞的可能性。
   + 基于摄像机、激光雷达、多模态……
   + 也可以将其视为实例分割任务。 
     + > 语义分割：像素级别，每一个像素都划分一个对应的类
     + > 实例分割：将划分为同一类的不同实例进一步分割
     + D. Neven, B. De Brabandere, S. Georgoulis, M. Proesmans, and L. Van Gool, **“Towards end-to-end lane detection: an instance segmentation approach,”** in Proc. IV, 2018, pp. 286–291.
   + 用到BEV鸟瞰图
     + L. Caltagirone, S. Scheidegger, L. Svensson, and M. Wahde, **“Fast LIDAR-based road detection using fully convolutional neural networks,”** in Proc. IV. IEEE, 2017, pp. 1019–1024.
     + M. Bai, G. Mattyus, N. Homayounfar, S. Wang, S. K. Lakshmikanth, and R. Urtasun, **“Deep multi-sensor lane detection,”** in Proc. IROS. IEEE, 2018, pp. 3102–3109.
   + Attention-Base：
     + X. Zhang, Z. Li, X. Gao, D. Jin, and J. Li, **“Channel attention in LiDAR-camera fusion for lane line segmentation,”** Pattern Recognit., vol. 118, p. 108020, 2021.
4. Traffic Sign Recognition：识别交通标志，提高ADAS安全性、舒适性。
  + 数据集GTSRB，reflection-backdoor也测试了。
  + W. Min, R. Liu, D. He, Q. Han, Q. Wei, and Q. Wang, **“Traffic sign recognition based on semantic scene understanding and structural traffic sign location,”** IEEE Trans. Intell. Transp. Syst., vol. 23, no. 9, pp. 15 794–15 807, 2022.

### VII. CHALLENGES AND FUTURE DIRECTIONS
challenges：
+ sun glare
+ adverse weather
+ failure detection
+ Data size, storage capability, and real-time processing speed

未来方向：
+ 长时自动驾驶的通用数据集
+ 边缘算力
+ 实时性、轻量级网络

## A survey on deep learning approaches for data integration in autonomous driving system
> Abstract—The perception module of self-driving vehicles relies on a multi-sensor system to understand its environment. Recent advancements in deep learning have led to the rapid development of approaches that integrate multi-sensory measurements to enhance perception capabilities. This paper surveys the latest deep learning integration techniques applied to the perception module in autonomous driving systems, categorizing integration approaches based on “what, how, and when to integrate.” A new taxonomy of integration is proposed, based on three dimensions: multi-view, multi-modality, and multi-frame. The integration operations and their pros and cons are summarized, providing new insights into the properties of an “ideal” data integration approach that can alleviate the limitations of existing methods. After reviewing hundreds of relevant papers, this survey concludes with a discussion of the key features of an optimal data integration approach.

本文研究应用于自动驾驶系统中感知模块的最新深度学习集成技术
+ 基于what、how、when对集成方法进行了分类。
+ 提出了一种基于多视图、多模态和多框架等三维空间的新的集成分类方法。 `multi-view, multi-modality, and multi-frame`
+ 总结了集成操作及其优缺点，为“理想的”数据集成方法的特性提供了新的见解，可以缓解现有方法的局限性。

文章题目中的`data integration`，也就是`data fusion`、`sensor fusion`。

### I. INTRODUCTION
+ What to integrate?
  + What is the content to integrate?
    + multi-view：集成来自相同类型传感器的不同视角。
    + multi-modality：集成不同类型传感器的数据。
    + multi-frame：来自不同时间的数据。
  + What is the order to integrate?
    + 以上有三个数据集成的维度，按什么维度顺序进行？
    + 先维度一；先维度二；多维度深度融合。
+ When to integrate?
  + 与数据的抽象级别相关
    + 数据级集成
    + 特征级集成
    + 决策级集成
    + 多级混合集成
  + 暂没有绝对的优劣之分，但是各有优缺点
+ How to integrate?
  + projection
  + concatenation
  + addition/average mean/weighted summation
  + probabilisitc method
  + rule-based transcation
  + temporal integration approach
  + neural network/encoder-decoder structure

Table 1：统计了最近关于ADS感知中深度学习的综述文章。

文章组织：
+ 第二节：ADS中常用传感器介绍
+ 三四五：What、How、When
+ 第六节：案例研究

### II. SENSING MODALITIES AND PRE-PROCESSING
传感器分为外部和内部，外部如摄像头获取外部信息，内部如GPS获取车辆自身的信息。主要关注外传感器。

+ camera
  + 一般讲的是RGB相机。
    + monocular：单眼的。多个单眼相机根据已知的方向和相对位置来还原三维信息。
    + stereo：立体的。根据内置算法得到三维信息。
  + 获取的信息有较高的时间分辨率（每秒几十帧）和空间分辨率
  + 不同的表示方法：
    + 像素表示：正常的HWC表示
    + point or voxel 表示：
      + 点云
      + 体素：类似我的世界。对3D空间进行网格划分，并赋予每个网格特征。
  + 制造成本低
  + 受天气影响大
  + 重构三维信息复杂
  + 没有被遮挡物体的检测能力
+ LiDAR：激光雷达
  + 有不同维度
    + 一维：距离测量
    + 二维：通过水平旋转，获取水平面上的X-Y坐标
    + 三维：通过竖直方向多个激光雷达，获取X-Y-Z坐标
  + 高价
  + 可以在低能见度的时候使用（可见光低能见度），但是雨雪雾霾会对激光产生散射的时候就不好用
  + 目标物体颜色影响：黑色物体吸收更多的光
  + 所得数据比图像稀疏，是点云数据
    + 点云 point：保存原始信息；数据量大，难以集成
    + 体素 voxel：相当于对点云进行降采样，损失精度，减小数据量
    + 像素/视图 pixel/view：将三维点云转换为二维视图BEV、range view范围视图、perspective view透视视图。与相机数据集成时常用。
    + 集成表示：多种方式结合。
+ Millimeter wave radar (MMW-radar)
  + 在极端天气和昏暗环境下表现更好
  + 低价
  + 利用多普勒效应检测物体速度
  + 表示方法：
    + 点云
    + 地图表示：在时间戳上累计雷达数据，形成BEV网格地图
  + 缺乏纹理、语义信息
  + 角度分辨率相对低
  + 无用电磁波干扰

### III. DATA INTEGRATION: WHAT TO INTEGRATE
集成什么？ -> 集成顺序？

在多视角、多模态、多帧三个维度下，探讨一维集成、二维集成；针对二维集成探讨集成顺序。

+ multi-view：
  + camera：给定摄像机的相对参数，通过计算从多个二维图像重建三维信息。
    + 内部参数：二维摄像机坐标到二维图像坐标的映射
    + 外部参数：三维世界坐标到二维摄像机坐标的映射
    + [知乎--相机内参外参](https://zhuanlan.zhihu.com/p/389653208)
    + [CSDN--相机内参外参](https://blog.csdn.net/weixin_43206570/article/details/84797361)
    + [外参矩阵](https://zhuanlan.zhihu.com/p/405306563)
  + 输入：
    + 多视角图像
    + 深度图像：只有一个通道，每个像素表示距离
    + 热图像：只有一个通道，表示红外辐射强度
  + 很少有用雷达的
+ multi-modality：
  + camera & LiDAR：一个二维一个三维信息，需要转换为相同的维度然后集成
  + camera & RADAR
  + LiDAR & RADAR
  + camera & LiDAR & RADAR
+ multi-frame：
  + camera
    + 两阶段，先对每一帧提取特征图；然后对特征图序列操作
    + 利用图像对(pairs)
  + LiDAR
    + 点云序列
    + 视角序列
+ multi-view & multi-modality：
  + 先view后modality
    + 集成LiDAR与多视图相机图像
    + 集成多视图LiDAR与单眼相机
  + 先modality后view
    + 在每个视图上进行模态融合，然后形成多视图信息
+ multi-view & multi-frame：
  + **BEVFormer**
  + 大多关注将时间序列应用到多视角相机中
  + 先将多视图融合为BEV，然后应用时序信息
+ multi-modality & multi-frame

### IV. DATA INTEGRATION: WHEN TO INTEGRATE
+ 数据级集成：如利用帧之间的光流信息
+ 特征级集成：广泛使用；有稀释单一模式的优势的不利影响。
+ 决策级集成
+ 多级集成

### V. DATA INTEGRATION: HOW TO INTEGRATE
+ projection：可以在数据级、特征级、决策级操作
  + 需要对齐坐标
  + 分辨率等因素会显著影响性能
+ concatenation：直接叠加特征图；要求一些维度对其（如HW，C可以不一样）
+ Addition-similar operations：加权求和等；要求形状完全一致或者可broadcast
+ probabilistic method：概率方法增加鲁棒性，但是要求学习更多的参数、数据
+ Rule-based transaction：人为制定规则
+ Temporal integration approaches
+ Encoder-decoder methods：
  + **BEVFormer**
  + >Theoretically, Transformer have unlimited receptive fields and can obtain global information with the Q, K, and V structure. BEVFormer integrate camera image multi-view and multi-frame information with Transformer architecture. Decoder of BEVFormer has a temporal self-attention layer where BEV queries can interact with historical BEV features, and a spatial crossattention layer where BEV queries interact with features of other camera views.

