# EMNIST数据集识别

此存储库包含使用卷积神经网络（CNN）在EMNIST数据集上训练的预测系统的代码。该系统包括两个主要部分：模型训练脚本（`train.py`）和预测脚本（`predict.py`）。

## 先决条件

在运行此项目之前，请确保已安装以下内容：

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pillow（PIL Fork）

## 安装

安装所需的包：

```bash
pip install tensorflow numpy pillow
```

## 训练模型

要在EMNIST数据集上训练模型，请运行`train.py`脚本。这将下载MNIST数据集，训练一个CNN模型，并将模型权重保存在`./ckpt`目录中。可以通过执行以下命令启动训练过程：

```bash
python train.py
```

训练过程将在指定的间隔保存检查点。这些检查点包含训练好的模型权重，允许恢复训练或进行预测而无需重新训练。

## 预测数字图像

训练模型后，可以使用`predict.py`来预测存储在`./test_images/`目录中的图像的数字。确保图像为28x28像素，灰度，并且与EMNIST数据集格式相似。

要运行预测，请执行：

```bash
python predict.py
```

该脚本将处理`./test_images/`目录中的每个图像，预测并打印它代表的数字。

## 添加测试图像

要添加更多的测试图像以进行预测：

1. 将图像放置在`./test_images/`目录中。
2. 确保图像为灰度，28x28像素。

预测脚本将自动处理目录中的所有图像。

## 作者

- YunPeng Zhu (3137168510@qq.com)

如果对此项目有任何问题或反馈，请随时联系作者。