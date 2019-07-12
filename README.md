# deep learning fault diagnosis

## 高被引论文模型复现

* [《Rolling Element Bearings Fault Intelligent Diagnosis Based on Convolutional Neural Network Using Raw Sensing Signal》][1]

* [《A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain AdaptationAbility on Raw Vibration Signals》][2]

  

  [1]: https://link.springer.com/chapter/10.1007/978-3-319-50212-0_10

  [2]: http://apps.webofknowledge.com/full_record.do?product=UA&amp;search_mode=GeneralSearch&amp;qid=1&amp;SID=7BQxbNTFkuf8oUqC9BD&amp;page=1&amp;doc=1

  

## 可选研究方向

* 域适应问题：某一工况下训练，提高其在另一工况下的诊断精度。模型的泛化能力。由于工作任务的变化，机器工作负载也会随之改变，如何利用在一个负载下的数据进行训练，对另一个负载下的信号进行诊断。
* 抗噪声：提高模型抗噪声能力，工业现场的噪声无法避免，使用加速度计测得的振动信号易被污染，研究如何从含有噪声的信号中诊断出轴承的故障
* 类别不平衡问题：健康数据>>故障数据
* 能不能找到新的损失函数，不是Pointwise，考虑之前状态？

