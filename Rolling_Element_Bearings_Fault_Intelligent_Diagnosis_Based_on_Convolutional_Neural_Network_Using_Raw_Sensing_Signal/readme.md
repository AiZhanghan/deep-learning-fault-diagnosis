# Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal

## 数据

12 Drive End Fault

DE - 驱动端加速度数据

外圈故障选6点钟方向数据

数据集A： 负载1 hp

数据集B： 负载2 hp

数据集C： 负载3 hp

数据集D：数据集A + 数据集B + 数据集C

测试集大小 = 25 * 2048 = 51200

训练集大小 = 2048 + 64 * 659 = 44224

所需总长度 = 测试集大小 + 训练集大小 = 95424，所以去头部多余数据点。