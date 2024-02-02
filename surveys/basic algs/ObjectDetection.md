# 综述
[Object Detection in 20 Years: A Survey阅读总结--知乎](https://zhuanlan.zhihu.com/p/192362333)

## 发展历程
+ 传统方法
  + HOG：基于网格；缩放输入图像，保持检测窗口不变
  + DPM：分而治之，将一个大物体理解为其部分的组合
+ 两阶段CNN：框内目标分类、框边界回归
  + R-CNN：
    + 在对不同的候选框进行特征提取
  + SPPNet：
    + 对整个图片进行特征提取，在feature map上找到每个候选框的区域（将候选框从原图映射到特征图），再对各个ROI采用SPP层池化，提取出固定长度的特征向量。
    + 缺点：多阶段训练；只对最后的FC层微调
  + Fast R-CNN：
    + ROI pooling层实际上是简化版SPP层，都是为了获取相同维度的特征
    + 相比SPPNet，最后把特征同时交给两个分类器，一个检测目标类别，一个精修边界框
  + Faster R-CNN：
    + 端到端的RCNN，引入了RPN
+ 一阶段CNN
  + YOLO：
    + 全部视为回归任务
  + SSD：
    + 多分辨率检测不同尺度的对象，对小物体友好。
  + DETR：
    + 应用transformer到目标检测
+ 数据集
  + Pascal VOC
  + MS-COCO
+ 效果衡量方法
  + 0.5 IoU的mAP
  + MS-COCO AP：0.5-0.95的AP平均值


# 目标检测算法
### R-CNN
+ 通过`selective search`找到候选框
  + >Selective search是一种比较好的数据筛选方式，首先对图像进行过分割切成很多很多小块，然后根据小块之间的颜色直方图、梯度直方图、面积和位置等基本特征，把相近的相邻对象进行拼接，从而选出画面中有一定语义的区域。
+ 候选框都被调整为相同大小
+ 用预训练好的CNN模型抽特征
+ 用SVM分类器预测每个框里的分类
+ bounding-box regression：
  + >首先用NMS对同个类的region proposals进行合并，因为SVM会给我们的每个ROI一个分数，因此我们选取得分最高而且与ground-truth重叠度超过阈值的ROI，记为P，然后丢弃与该被选框重叠度较高的那些。对剩下的高分ROI，每个P与其对应的ground-truth组成一个样本对，我们使用他们作为样本让线性回归学习一个从预测框到ground-truth映射关系。
  + 这个映射关系先进行线性映射，然后进行平移和缩放。其中只有线性映射是需要学习的。
+ 缺点：对需要对不同候选框提取特征，重复计算

### Faster R-CNN
+ [一文读懂Faster R-CNN -- 知乎](https://zhuanlan.zhihu.com/p/31426458)
+ [pytorch Faster-RCNN代码导读 -- 知乎](https://zhuanlan.zhihu.com/p/145842317)
+ 引入`Region Proposal Network (RPN)`，几乎零开销的区域提案
+ 首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
+ RPN通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
+ Roi Pooling收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
+ 最后利用上述proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。
+ Feature Pyramid Networks FPN：
+ 使用多层不同分辨率的特征图，综合高层低分辨率高语义信息和低层高分辨率低语义信息。
+ bottom-up和top-down两步。其中top-down先上采样之后与前一层特征图相加，然后进行卷积形成新的特征图。

### YOLO
+ [YOLO -- 知乎](https://zhuanlan.zhihu.com/p/32525231)
+ 将图像划分为多个区域，并同时预测每个区域的边界框和概率。
+ 检测小物体能力较差。
+ 预测边框位置+置信度+类别
  + 位置由左上角右下角坐标，共四个值组成
  + 置信度为 I(含有物体)*IOU
  + 类别数为C
  + 因此一个区域要预测 4+1+C 个值