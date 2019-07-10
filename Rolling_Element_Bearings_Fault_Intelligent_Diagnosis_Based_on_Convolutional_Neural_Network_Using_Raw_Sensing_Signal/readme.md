# Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal

## 数据

（可能有问题，数据集里1750rpm的Normal数据可能取错了）

12 Drive End Fault

DE - 驱动端加速度数据，只取了一个传感器数据

外圈故障选6点钟方向数据

数据集A： 负载1 hp

数据集B： 负载2 hp

数据集C： 负载3 hp

数据集D：数据集A + 数据集B + 数据集C

测试集大小 = 25 * 2048 = 51200

训练集大小 = 2048 + 64 * 659 = 44224

验证集大小 = 7 * 2048  = 14336

所需总长度 = 测试集大小 + 训练集大小 + 验证集大小 = 109760，所以去头部多余数据点。

## 数据预处理

* 归一化的方式？

  数据为（N， d）的二维矩阵，N为样本数，d为样本特征维度。归一化应该沿0轴，还是1轴？

## 训练

* he_normal初始化

* 通道数为1

  ```python
  x_train = np.expand_dims(x_train, axis = 2)
  input_shape = (sample_size, 1)
  ```

* 保存与读取模型

  ```python
  model.save(file_path)
  
  from keras.models import load_model
  model_A = load_model(file_path)
  ```

* history

* 加入验证集

* Keras + SKlearn

## 模型可视化

`keras.utils.vis_utils` 模块提供了一些绘制 Keras 模型的实用功能(使用 `graphviz`)。

以下实例，将绘制一张模型图，并保存为文件：

```python
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes = True)
```

`plot_model` 有 4 个可选参数:

- `show_shapes` (默认为 False) 控制是否在图中输出各层的尺寸。
- `show_layer_names` (默认为 True) 控制是否在图中显示每一层的名字。
- `expand_dim`（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
- `dpi`（默认为 96）控制图像 dpi。

## 训练历史可视化

Keras `Model` 上的 `fit()` 方法返回一个 `History` 对象。`History.history` 属性是一个记录了连续迭代的训练/验证（如果存在）损失值和评估值的字典。这里是一个简单的使用 `matplotlib` 来生成训练/验证集的损失和准确率图表的例子：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

## Model_A

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model.png)



![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_A_acc.png)

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_A_loss.png)

* 复现精度：98.4%

## Model_B

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_B_acc.png)

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_B_loss.png)

* 复现精度：99.2%

## Model_C

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_C_acc.png)

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_C_loss.png)

* 复现精度：99.2%

## Model_D

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_D_acc.png)

![](https://raw.githubusercontent.com/AiZhanghan/deep-learning-fault-diagnosis/master/Rolling_Element_Bearings_Fault_Intelligent_Diagnosis_Based_on_Convolutional_Neural_Network_Using_Raw_Sensing_Signal/model_D_loss.png)

* 复现精度：99.1%