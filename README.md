# GAN-图片生成器

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.15%2B-green)](https://pytorch.org/vision/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-yellow)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-red)](https://matplotlib.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.65%2B-purple)](https://github.com/tqdm/tqdm)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

本项目实现了经典的生成对抗网络（Generative Adversarial Networks, GAN），用于生成逼真的手写数字图像。GAN由两个神经网络组成：

- **生成器（Generator）**：接收随机噪声向量，输出生成的假图像
- **判别器（Discriminator）**：判断输入图像是真实的还是生成的

通过对抗训练，生成器逐渐学会生成以假乱真的手写数字图像。

本项目使用MNIST是为了方便演示，实际使用可以使用其他数据集。

## 使用方法

### 方法一：使用Jupyter Notebook

```bash
jupyter notebook gan-mnist.ipynb
```

按顺序运行每个单元格即可。

## 主要功能

### 1. 数据处理

- 自动下载MNIST数据集
- 数据归一化和标准化处理
- 高效的数据加载器（支持多进程和GPU加速）

### 2. 模型架构

#### 生成器（Generator）

- 输入：100维随机噪声向量
- 输出：28×28灰度图像
- 结构：全连接层 + 转置卷积层

#### 判别器（Discriminator）

- 输入：28×28灰度图像
- 输出：真假概率（0-1）
- 结构：卷积层 + 全连接层

### 3. 训练特性

- ✅ 支持单GPU/多GPU训练（自动检测）
- ✅ 支持断点续训（设置`continue_train = True`）
- ✅ 学习率动态调整
- ✅ 训练进度可视化（tqdm进度条）
- ✅ 定期保存模型和生成样本

### 4. 效果测试

- 生成样本网格可视化
- 潜在空间插值展示

## 超参数配置

| 参数         | 默认值 | 说明         |
| ------------ | ------ | ------------ |
| `batch_size` | 256    | 批次大小     |
| `epochs`     | 25     | 训练轮数     |
| `lr_g`       | 0.0002 | 生成器学习率 |
| `lr_d`       | 0.0002 | 判别器学习率 |
| `latent_dim` | 100    | 噪声向量维度 |
| `img_size`   | 28     | 图像尺寸     |
| `channels`   | 1      | 图像通道数   |

## GPU支持

本项目针对GPU T4 x2进行了优化：

- 使用`nn.DataParallel`实现多GPU并行
- 较大的batch_size充分利用显存
- `pin_memory`加速数据传输
- 异步数据加载

## 断点续训

如需从已有模型继续训练，修改以下配置：

```python
continue_train = True
continue_model_path = "./models/generator_epoch_20.pth"
```

## 生成效果

训练完成后，生成的样本将保存在`./samples/`目录（请打开相应的注释）下：

- `real_samples_preview.png`：真实数据预览
- `epoch_*_batch_*.png`：训练过程中的生成样本
- `generated_grid_8x8.png`：8×8生成样本网格
- `interpolation_samples.png`：潜在空间插值效果

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

## 作者信息

**谙弆悕博士（Ailan Anjuxi）**

- 📧 邮箱：[anjuxi.ME@outlook.com](mailto:anjuxi.ME@outlook.com)
- 📞 SIP电话：[sip:anjuxi@sip.linphone.org](sip:anjuxi@sip.linphone.org)
