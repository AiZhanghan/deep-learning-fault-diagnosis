# A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals

智能故障诊断技术能在取代费时费力而又不可靠的人工故障分析的同时，提高故障诊断的有效性。深度学习得益于其模型的多层非线性映射能力，能提高智能故障诊断的准确性。本文提出了一种新型的第一层宽卷积核的卷积神经网络（WDCNN）。该方法将原始振动信号作为输入，同时，通过数据增强方法得到更多的训练集数据，用宽卷积核的第一卷积层提取特征，并抑制高频噪声。后续层均使用小卷积核来进行多层非线性映射。并使用AdaBN提高模型的域适应能力。该模型解决了目前基于CNN故障诊断方法精度不够高的问题。WDCNN不仅能对常见数据集（CWRU Bearing Data）达到100%的分类精度，而且在不同工况和噪声环境条件下，其表现也优于目前最高水准的基于频率特征的DNN模型。

