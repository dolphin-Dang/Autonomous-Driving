# End-to-End Object Detection with Transformers
[DETR-github](https://github.com/facebookresearch/detr)

## 概述
重点：
+ end-to-end：大多数以前的操作需要hand-designed components，比如设置anchor、NMS。其中NMS训练、部署困难。
+ transformer应用到目标检测

pipeline：
+ CNN抽取特征
+ transformer encoder学习全局特征
+ transformer decoder生成预测框
+ 将预测框和GroundTruth框进行匹配，匹配上的计算类别loss和框loss，未匹配上的记为背景类。推理时用阈值选取置信度高的框。

避免冗余框问题：
+ transformer全局建模能力使得不会出现过多冗余框
+ 别的深度学习目标检测模型都是把目标检测任务转化为回归或者分类问题，而DETR将其视为集合预测问题。这样可以避免冗余框的问题。

## Preliminaries
### 集合预测
使用基于匈牙利算法的loss来进行预测框和真实框的最佳匹配。

### 目标检测
+ two-stage：提出proposal；从中选择；NMS。
+ one-stage：根据anchor得到结果；NMS。

以上方法性能极大依赖于初始猜测，因此后处理十分重要。

## DETR model
### 基于集合预测的目标函数
一次推理出的框个数固定（N=100）。预测结果就是N维向量，ground-truth也是，其中没有物体的框用“背景类”表示。

每一个元素$y_i=(c_i,b_i)$，其中前者是类别，后者是一个4维tuple分别代表框中心和框的宽高占全图的比例。

$L_{match}=L_{class}+L_{box}$

$L_{box}=L_{iou}+L1_{loss}$


### DETR结构细节
+ object query相当于可学习的anchor，每一个负责图像一部分区间里的查询（原文Fig.7）


# DEFORMABLE DETR: DEFORMABLE TRANSFORMERS FOR END-TO-END OBJECT DETECTION
[Deformable DETR-github](https://github.com/fundamentalvision/Deformable-DETR)

传统的attention需要更新权重，学习某个点对应哪些位置比较重要，这个学习过程开销很大；而deformable attention则通过一个全连接层来找到对应点，并在此基础上进行attention计算。
