# Deep_learning_tutorial
## 深度学习教程
### 前言
主要是练习李沐老师教程中的源码，我在每一章中容易有疑惑的地方进行了更细致的补充，另外补充了练习题的答案。所用的深度学习框架为Pytorch。
### TODO
- [ ] 参加Kaggle竞赛
- [ ] 完成教程中的概念源码学习，以及练习课后题
- [X] 把每章中的每节进行编号
### 章节
#### 01.preliminarier
* [01.数据操作。讲解pytorch中的张量(tensor)的一些使用方法；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/01.ndarray.ipynb)
* [02.数据预处理。讲解使用pandas对数据进行一系列的预处理操作；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/02.pandas.ipynb)
* [03.线性代数。讲解标量、向量、矩阵、张量的基本概念和性质；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/03.linear-algebra.ipynb)
* [04.微积分。讲解导数和微分、偏导数、梯度、链式法则等基本概念性质，以及怎么使用matplotlib画图；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/04.calculus.ipynb)
* [05.自动微分(autograd)。Pytorch可以自动求导。Pytorch函数表达式是隐式构造的，求解导数时是反向累积的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/01.preliminaries/05.autograd.ipynb)
#### 02.linear-networks
* [01.线性回归。讲解线性模型，它是一个单层神经网络；机器学习模型中最关键要素是训练数据、损失函数、优化算法，还有模型本身；使用最小二乘法和极大似然估计这两种方法去求解 w 和 b 是等价的；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/01.linear-regression.ipynb)
* [02.线性回归的源码实现。讲解线性回归的具体源码实现，包括初始化模型参数、定义模型、定义损失函数、定义优化算法、训练；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/02.linear-regression-scratch.ipynb)
* [03.线性回归的框架实现。讲解线性回归的框架实现，在pytorch中，data模块提供了数据处理工具，nn模块定义了大量的神经网络层和常见损失函数，optim模块有实现优化算法；使用深度学习框架可以高效地搭建神经网络；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/03.linear-regression-concise.ipynb)
* [04.图像分类数据集。本节介绍了一个服装分类数据集Fashion-MNIST，它由10个类别的图像组成；同时实现了一个函数用于获取读取数据集，函数返回训练集和验证集的数据迭代器；](https://github.com/lixinjie97/Deep_learning_tutorial/blob/main/02.linear-networks/01.linear-regression.ipynb)