# EdgeAttnSwin-Edge-Aware-and-Attention-Augmented-SwinFusion-for-High-Quality-Multi-Modal-Image-Fusion
图像融合是融合多源图像信息、生成高质量图像的重要技术。针对现有方法在边缘细节、特征提取与多尺度融合上的不足，本文基于SwinFusion提出改进框架EdgeAttnSwin。该框架结合CNN‑Transformer架构，引入EAM、CSEAM、MSFM三个模块，分别用于增强边缘特征、强化关键信息、聚合多尺度特征，并通过优化损失函数平衡结构与纹理。在可见光‑红外、医学图像等多模态融合任务中，EdgeAttnSwin在指标与视觉效果上均优于现有算法，能更好保留细节、突出目标，提升融合质量。
# 各场景下的模型使用说明
Visible and Infrared Image Fusion (VIF，可见光与红外图像融合)
To Train（训练）：
数据集：需下载 MSRS 数据集（https://github.com/Linfeng-Tang/MSRS），放置路径为 ./Dataset/trainsets/MSRS/；
训练命令：指定分布式训练的进程数、端口号，加载 VIF 场景的训练配置文件，开启分布式训练。
To Test（测试）：
数据集：同训练（MSRS），放置路径为 ./Dataset/testsets/MSRS/；
测试命令：指定模型路径、迭代次数、数据集名称、红外 / 可见光图像目录。

Medical Image Fusion (Med，医学图像融合)
To Train：
数据集：Harvard 医学数据集（已经提供完整代码压缩包"EdgeAttnSwin_Med.zip"），分 PET-MRI/CT-MRI 两类，放置对应路径；
训练命令：加载医学图像融合的训练配置文件。
To Test：
数据集：（文档笔误，应为 “测试数据集”）同训练，放置对应测试路径；
测试命令：分 PET-MRI/CT-MRI 两种场景，分别指定 MRI/PET、MRI/CT 目录。
