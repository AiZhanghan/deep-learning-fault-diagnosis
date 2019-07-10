# A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals

智能故障诊断技术能在取代费时费力而又不可靠的人工故障分析的同时，提高故障诊断的有效性。深度学习得益于其模型的多层非线性映射能力，能提高智能故障诊断的准确性。本文提出了一种新型的第一层宽卷积核的卷积神经网络（WDCNN）。该方法将原始振动信号作为输入，同时，通过数据增强方法得到更多的训练集数据，用宽卷积核的第一卷积层提取特征，并抑制高频噪声。后续层均使用小卷积核来进行多层非线性映射。并使用AdaBN提高模型的域适应能力。该模型解决了目前基于CNN故障诊断方法精度不够高的问题。WDCNN不仅能对常见数据集（CWRU Bearing Data）达到100%的分类精度，而且在不同工况和噪声环境条件下，其表现也优于目前最高水准的基于频率特征的DNN模型。

* 与计算机视觉领域CNN设计标准不同，第一层宽卷积核，抑制高频噪声，最后一个池化层单元在原始信号的感受野大于信号周期；
* Overlap for Data Augmentation
* AdaBN 域适应能力

## 验证数据集大小的作用

验证数据集大小为90， 300， 3000， 19800时的模型精度，其中大小为90， 300是不使用数据数据增强，3000， 19800使用数据增强（Overlap）。训练迭代次数4000。

### t-SNE可视化

取最后一层隐含层对测试集的激活值（特征图）

```python
from keras.models import Model

dense_layer_model = Model(inputs = model.input, output = model.get_layer('activation_6').output)

dense_output = dense_layer_model.predict(x_test)
```

```python
from sklearn.manifold import TSNE

tsne = TSNE()
x_tsne = tsne.fit_transform(dense_output)
font = {"color": "darkred",
        "size": 13, 
        "family" : "serif"}
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_label, alpha=0.6, 
            cmap=plt.cm.get_cmap('rainbow', 10))
plt.title("t-SNE", fontdict=font)
cbar = plt.colorbar(ticks=range(10)) 
cbar.set_label(label='digit value', fontdict=font)
plt.clim(-0.5, 9.5)
```



### 数据集大小90

![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_90_acc.png)

![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_90_loss.png)

* 复现精度：65.6%

* 分析：数据量不够，存在明显过拟合现象。

* t-SNE可视化：

  ![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_90_t_SNE.png)

### 数据集大小300

![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_300_acc.png)

![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_300_loss.png)

复现精度：93.2%

* t-SNE可视化：

  ![](G:\Huang_Zhenkai\workspace\github\deep-learning-fault-diagnosis\A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals\effect_of_data_number\data_number_300_t_SNE.png)

  分析：较好分开，第三类和第一类混叠，Ball-0.07，Ball-0.21

## 数据集大小 3000

![](.\effect_of_data_number\data_number_3000_acc.png)

![](.\effect_of_data_number\data_number_3000_loss.png)

* 复现精度：100%

* t-SNE可视化

![](.\effect_of_data_number\data_number_3000_t_SNE.png)
