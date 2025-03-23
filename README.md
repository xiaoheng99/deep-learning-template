### 安装依赖
1. pip install requirements.txt

### 运行代码
1. python3 train.py 或者在window直接运行train.py
2. 在cnn里可以运行测试模型

### 在tensorboard里查看模型参数
输入: tensorboard --logdir outputs/ --port 6006

### 修改模型
1.可以在model里新开一个文件夹，然后写下你的模型代码即可

### 修改配置
可以在yaml文件里修改配置的参数

#### 模版
configs: 表示超参数的配置
model: 神经网络的模型
outputs: 表示输出的log和图像
utils： 工具文件
dataset: 加载数据和数据的预处理
train: 训练模型并评估